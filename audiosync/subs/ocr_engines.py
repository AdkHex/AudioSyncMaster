"""The OCR engines behind Subsync's OCR task.

Each engine takes a list of prepared line images (black text on white PNGs,
one text line each) and returns, per image, what it read: observations with
text, confidence and -- where the engine knows them -- boxes. Turning those
into cue text (furigana, joining, fixes) is ``ocr.py``'s job, so every engine
is judged by the same rules.

* **vision** -- Apple Vision through ``native/vision-ocr``, a small Swift
  helper run once per batch. Found via ``AUDIOSYNC_VISION_OCR``, a bundled
  copy (``resources/tools`` beside the bundled FFmpeg), or compiled from
  source with ``swiftc`` into the user cache on first use (dev machines).
* **tesseract** -- the ``tesseract`` CLI on PATH, one process per line in a
  small thread pool, TSV output for word confidences.
* **rapidocr** -- PaddleOCR models on ONNX Runtime, in the OCR pack's own
  Python (``workers/ocr_rapid_worker.py``).
* **claude** -- Claude vision over the Messages API (stdlib HTTP), several
  line images per request, transcription only.
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .. import media
from ..media import Cancelled, CancellationToken
from . import languages
from .tasks import TaskError

ENGINE_LABELS = {
    "vision": "Apple Vision",
    "tesseract": "Tesseract",
    "rapidocr": "RapidOCR",
    "claude": "Claude vision",
}

RAPID_PACK = "ocr-rapidocr"
RAPID_WORKER = "ocr_rapid_worker.py"

#: Claude reads lines in batches: enough to amortise the prompt, few enough
#: that one refused or truncated answer costs little to redo.
CLAUDE_BATCH = 24
#: Claude's confidence is not reported by the API; a line it marks as
#: uncertain gets the low value so it is flagged for review.
CLAUDE_CONFIDENCE = 0.95
CLAUDE_UNCERTAIN = 0.4

Progress = Callable[[int, int], None]


@dataclass
class LineRead:
    """What one engine read in one line image."""

    #: ``{"text", "confidence", "box": [x, y, w, h] | None}``; boxes are
    #: normalised with a top-left origin.
    observations: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def empty(self) -> bool:
        return not any((o.get("text") or "").strip() for o in self.observations)


# ------------------------------------------------------------ subprocesses


def _popen(command: List[str], token: Optional[CancellationToken], env: Optional[dict] = None) -> subprocess.Popen:
    kwargs = media._popen_kwargs()
    kwargs["stdin"] = subprocess.PIPE
    if env is not None:
        kwargs["env"] = env
    try:
        process = subprocess.Popen(command, **kwargs)
    except OSError as exc:
        raise TaskError(f"Could not start {os.path.basename(command[0])}: {exc}") from exc
    if token:
        token.register(process)
    return process


def _run(command: List[str], timeout: float, token: Optional[CancellationToken], env: Optional[dict] = None) -> bytes:
    """Run to completion; stdout on success, TaskError with stderr's last
    line otherwise. Cancellation kills the process group."""
    if token:
        token.raise_if_cancelled()
    process = _popen(command, token, env)
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            media._terminate(process)
            process.communicate()
            raise TaskError(f"{os.path.basename(command[0])} timed out after {timeout:.0f} s")
        if token and token.cancelled:
            raise Cancelled("operation cancelled")
        if process.returncode != 0:
            lines = stderr.decode("utf-8", "replace").strip().splitlines()
            raise TaskError(f"{os.path.basename(command[0])} failed: {lines[-1] if lines else process.returncode}")
        return stdout
    finally:
        if token:
            token.unregister(process)
        if process.poll() is None:
            media._terminate(process)


# ----------------------------------------------------------------- vision

_VISION_LOCK = threading.Lock()
_VISION_LANGS: Dict[str, List[str]] = {}

#: Preferred Vision languages per ISO 639-1 code, most wanted first. English
#: rides along for CJK scripts because credits, signs and names in Latin
#: letters are common in those subtitles.
_VISION_PREFERRED: Dict[str, List[str]] = {
    "ja": ["ja-JP", "en-US"],
    "zh": ["zh-Hans", "zh-Hant", "en-US"],
    "zh-hant": ["zh-Hant", "zh-Hans", "en-US"],
    "yue": ["yue-Hant", "yue-Hans", "zh-Hant", "en-US"],
    "ko": ["ko-KR", "en-US"],
    "en": ["en-US"],
    "fr": ["fr-FR", "en-US"],
    "it": ["it-IT", "en-US"],
    "de": ["de-DE", "en-US"],
    "es": ["es-ES", "en-US"],
    "pt": ["pt-BR", "en-US"],
    "ru": ["ru-RU", "en-US"],
    "uk": ["uk-UA", "ru-RU", "en-US"],
    "vi": ["vi-VT", "en-US"],
    "ar": ["ar-SA", "ars-SA"],
    "no": ["nb-NO", "no-NO", "en-US"],
    "nn": ["nn-NO", "nb-NO", "en-US"],
}


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def user_cache_dir() -> str:
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "AudioSyncMaster")


def tools_dirs() -> List[str]:
    """Where a bundled helper can live: ``tools`` beside the bundled
    ``ffmpeg`` folder, and the same candidates media.py uses for FFmpeg."""
    out: List[str] = []
    bundled = media._bundled_dir()
    if bundled:
        out.append(os.path.join(os.path.dirname(bundled), "tools"))
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        out += [
            os.path.join(exe_dir, "resources", "tools"),
            os.path.join(exe_dir, "tools"),
            os.path.join(os.path.dirname(exe_dir), "tools"),
        ]
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            out.append(os.path.join(meipass, "resources", "tools"))
    out.append(os.path.join(_repo_root(), "src-tauri", "resources", "tools"))
    seen: List[str] = []
    for d in out:
        if d not in seen:
            seen.append(d)
    return seen


def vision_source() -> Optional[str]:
    path = os.path.join(_repo_root(), "native", "vision-ocr", "main.swift")
    return path if os.path.isfile(path) else None


def _compiled_vision_path(source: str) -> str:
    with open(source, "rb") as handle:
        digest = hashlib.sha1(handle.read()).hexdigest()[:12]
    return os.path.join(user_cache_dir(), "vision-ocr", digest, "vision-ocr")


def vision_helper(build: bool = True, token: Optional[CancellationToken] = None,
                  log: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Path of the Vision helper, compiling it from source if allowed."""
    if sys.platform != "darwin":
        return None
    override = os.environ.get("AUDIOSYNC_VISION_OCR")
    if override and os.path.isfile(override):
        return override
    for directory in tools_dirs():
        candidate = os.path.join(directory, "vision-ocr")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    source = vision_source()
    if not source:
        return None
    target = _compiled_vision_path(source)
    if os.path.isfile(target):
        return target
    swiftc = shutil.which("swiftc")
    if not build or not swiftc:
        return None
    with _VISION_LOCK:
        if os.path.isfile(target):
            return target
        os.makedirs(os.path.dirname(target), exist_ok=True)
        staging = f"{target}.{os.getpid()}.tmp"
        if log:
            log("Building the Apple Vision OCR helper (first use only)...")
        _run([swiftc, "-O", "-swift-version", "5", "-o", staging, source], 900, token)
        os.replace(staging, target)
    return target


def vision_supported_languages(helper: str, token: Optional[CancellationToken] = None) -> List[str]:
    cached = _VISION_LANGS.get(helper)
    if cached is None:
        try:
            data = json.loads(_run([helper, "--list-languages"], 60, token).decode("utf-8"))
            cached = list(data.get("accurate") or [])
        except (TaskError, ValueError):
            cached = []
        _VISION_LANGS[helper] = cached
    return cached


def vision_languages(language: Optional[str], supported: Sequence[str]) -> List[str]:
    """Vision recognitionLanguages for a subtitle language, limited to what
    this macOS supports. Empty means "detect automatically"."""
    raw = (language or "").strip().lower().replace("_", "-")
    code = languages.normalize(raw)
    if code == "zh" and any(tag in raw for tag in ("hant", "tw", "hk", "mo")):
        wanted = _VISION_PREFERRED["zh-hant"]
    elif code:
        wanted = list(_VISION_PREFERRED.get(code, []))
        wanted += [s for s in supported if s.split("-")[0].lower() == code and s not in wanted]
    else:
        return []
    if supported:
        wanted = [w for w in wanted if w in supported]
    primary = [w for w in wanted if w != "en-US"]
    if code != "en" and not primary:
        return []  # this macOS cannot read the language; let Vision detect
    return wanted


def run_vision(paths: List[str], language: Optional[str], options: dict, ctx: Any,
               progress: Optional[Progress] = None) -> List[LineRead]:
    helper = vision_helper(build=True, token=ctx.token, log=ctx.log)
    if not helper:
        raise TaskError(_vision_reason() or "The Apple Vision OCR helper is not available.")
    langs = vision_languages(language, vision_supported_languages(helper, ctx.token))
    if language and not langs:
        ctx.log(f"Apple Vision on this Mac cannot read {languages.name_of(language)}; using automatic detection.")
    request = {
        "images": paths,
        "languages": langs,
        "languageCorrection": bool(options.get("languageCorrection", True)),
        "jobs": max(1, min(4, os.cpu_count() or 2)),
    }
    results: List[LineRead] = [LineRead(error="no result") for _ in paths]
    process = _popen([helper, "--stdin"], ctx.token)
    errors: List[bytes] = []
    drain = threading.Thread(target=lambda: errors.append(process.stderr.read()), daemon=True)
    drain.start()
    done = 0
    try:
        process.stdin.write(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        process.stdin.close()
        for raw in process.stdout:
            try:
                item = json.loads(raw.decode("utf-8"))
            except ValueError:
                continue
            index = item.get("index")
            if not isinstance(index, int) or not 0 <= index < len(paths):
                continue
            if item.get("error"):
                results[index] = LineRead(error=str(item["error"]))
            else:
                results[index] = LineRead(
                    observations=[
                        {"text": o.get("text", ""), "confidence": float(o.get("confidence", 0.0)), "box": o.get("box")}
                        for o in item.get("observations") or []
                    ]
                )
            done += 1
            if progress:
                progress(done, len(paths))
        process.wait()
        drain.join(timeout=5)
        if ctx.token.cancelled:
            raise Cancelled("operation cancelled")
        if process.returncode != 0 and done < len(paths):
            tail = b"".join(errors).decode("utf-8", "replace").strip().splitlines()
            raise TaskError(f"Apple Vision OCR failed: {tail[-1] if tail else process.returncode}")
    finally:
        ctx.token.unregister(process)
        if process.poll() is None:
            media._terminate(process)
    return results


def _vision_reason() -> Optional[str]:
    if sys.platform != "darwin":
        return "Apple Vision is only available on macOS."
    if vision_helper(build=False):
        return None
    if vision_source() and shutil.which("swiftc"):
        return None
    return "The Apple Vision helper is missing from this build (and swiftc is not installed to build it)."


# -------------------------------------------------------------- tesseract

_TESSERACT_CODES: Dict[str, str] = {
    "ja": "jpn", "zh": "chi_sim", "yue": "chi_tra", "ko": "kor", "en": "eng", "fr": "fra", "es": "spa",
    "it": "ita", "de": "deu", "pt": "por", "ru": "rus", "nl": "nld", "sv": "swe", "da": "dan", "no": "nor",
    "fi": "fin", "pl": "pol", "cs": "ces", "sk": "slk", "hu": "hun", "ro": "ron", "el": "ell", "tr": "tur",
    "ar": "ara", "he": "heb", "hi": "hin", "th": "tha", "vi": "vie", "id": "ind", "ms": "msa", "uk": "ukr",
    "bg": "bul", "hr": "hrv", "sr": "srp", "sl": "slv", "lt": "lit", "lv": "lav", "et": "est", "ca": "cat",
    "fa": "fas", "ta": "tam", "te": "tel", "bn": "ben", "ka": "kat", "is": "isl", "eu": "eus", "gl": "glg",
}  # fmt: skip

_INSTALL_HINT = (
    "Install Tesseract with its language data: macOS 'brew install tesseract tesseract-lang', "
    "Debian/Ubuntu 'apt install tesseract-ocr tesseract-ocr-<lang>', Windows the UB Mannheim installer "
    "(tick the languages you need)."
)


def tesseract_path() -> Optional[str]:
    override = os.environ.get("AUDIOSYNC_TESSERACT")
    if override and os.path.isfile(override):
        return override
    return shutil.which("tesseract")


def tesseract_code(language: Optional[str]) -> str:
    raw = (language or "").lower()
    code = languages.normalize(raw)
    if code == "zh" and any(tag in raw for tag in ("hant", "tw", "hk")):
        return "chi_tra"
    return _TESSERACT_CODES.get(code or "en", "eng")


def tesseract_languages(binary: str, token: Optional[CancellationToken] = None) -> List[str]:
    try:
        out = _run([binary, "--list-langs"], 30, token).decode("utf-8", "replace")
    except TaskError:
        return []
    return [line.strip() for line in out.splitlines()[1:] if line.strip()]


def parse_tesseract_tsv(tsv: str) -> List[Dict[str, Any]]:
    """Word rows (level 5) of Tesseract's TSV as observations with boxes in
    pixels (normalised later by the caller if needed)."""
    words = []
    lines = tsv.splitlines()
    if not lines:
        return words
    header = lines[0].split("\t")
    col = {name: i for i, name in enumerate(header)}
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < len(header):
            continue
        try:
            if int(parts[col["level"]]) != 5:
                continue
            conf = float(parts[col["conf"]])
        except (ValueError, KeyError):
            continue
        text = parts[col["text"]].strip()
        if not text or conf < 0:
            continue
        box = [int(parts[col[k]]) for k in ("left", "top", "width", "height")]
        words.append({"text": text, "confidence": conf / 100.0, "box": box})
    return words


def run_tesseract(paths: List[str], language: Optional[str], options: dict, ctx: Any,
                  progress: Optional[Progress] = None) -> List[LineRead]:
    binary = tesseract_path()
    if not binary:
        raise TaskError("Tesseract is not installed. " + _INSTALL_HINT)
    code = tesseract_code(language)
    installed = tesseract_languages(binary, ctx.token)
    if installed and code not in installed:
        raise TaskError(f"Tesseract has no {languages.name_of(language)} data ({code}). " + _INSTALL_HINT)
    lang_arg = code + ("+eng" if code != "eng" and "eng" in installed and languages.is_cjk(language) else "")
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    results: List[LineRead] = [LineRead() for _ in paths]
    done = 0
    lock = threading.Lock()

    def one(i: int) -> None:
        nonlocal done
        ctx.check()
        try:
            out = _run([binary, paths[i], "stdout", "-l", lang_arg, "--psm", "7", "tsv"], 120, ctx.token, env)
            words = parse_tesseract_tsv(out.decode("utf-8", "replace"))
            if words:
                joined = join_words([w["text"] for w in words], languages.is_cjk(language))
                conf = sum(w["confidence"] for w in words) / len(words)
                results[i] = LineRead(observations=[{"text": joined, "confidence": conf, "box": None}])
        except TaskError as exc:
            results[i] = LineRead(error=str(exc))
        with lock:
            done += 1
            if progress:
                progress(done, len(paths))

    workers = max(1, min(4, os.cpu_count() or 2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(one, i) for i in range(len(paths))]:
            future.result()
    return results


def _is_cjk_char(ch: str) -> bool:
    o = ord(ch)
    return (
        0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF
        or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF or 0x3000 <= o <= 0x303F
    )


def join_words(words: Sequence[str], cjk: bool) -> str:
    """Join recognised words: no space between two CJK characters."""
    out = ""
    for word in words:
        if out and not (cjk and _is_cjk_char(out[-1]) and _is_cjk_char(word[0])):
            out += " "
        out += word
    return out


# --------------------------------------------------------------- rapidocr


def run_rapidocr(paths: List[str], language: Optional[str], options: dict, ctx: Any,
                 progress: Optional[Progress] = None) -> List[LineRead]:
    from . import packs

    if not packs.python_for(RAPID_PACK):
        raise TaskError("Install the OCR (RapidOCR) pack first (Settings > Engine packs).")
    request: Dict[str, Any] = {"images": paths, "language": languages.normalize(language) or language}
    # The built-in model reads English and simplified Chinese; other scripts
    # need their own recogniser, downloaded on first use like a Whisper model.
    recognizer = packs.ocr_recognizer(language)
    if recognizer and recognizer.get("needed"):
        model_id = recognizer["needed"]
        ctx.log(f"Downloading the RapidOCR {model_id} model for {languages.name_of(language)}...")
        packs.install(RAPID_PACK, ctx.progress, ctx.log, ctx.token, model=model_id)
        recognizer = packs.ocr_recognizer(language)
    if recognizer and recognizer.get("recModel"):
        request["recModel"] = recognizer["recModel"]
        request["recKeys"] = recognizer.get("recKeys")
    data = packs.run_worker(RAPID_PACK, RAPID_WORKER, request, ctx)
    results: List[LineRead] = [LineRead(error="no result") for _ in paths]
    for item in data.get("results") or []:
        index = item.get("index")
        if isinstance(index, int) and 0 <= index < len(paths):
            if item.get("error"):
                results[index] = LineRead(error=str(item["error"]))
            else:
                results[index] = LineRead(observations=list(item.get("observations") or []))
    if progress:
        progress(len(paths), len(paths))
    return results


# ----------------------------------------------------------------- claude

_CLAUDE_SYSTEM = (
    "You transcribe subtitle images. Each image is one line of subtitle text cut from a film or TV "
    "episode: dark text on a white background. Write exactly the characters you see -- same "
    "language, same script, same punctuation and capitalisation. Never translate, correct grammar, "
    "complete words, or add anything that is not in the image. Ignore small ruby/furigana above "
    "kanji. If a line is empty or unreadable, return an empty string and mark it uncertain."
)

_CLAUDE_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "uncertain": {"type": "boolean"},
                },
                "required": ["id", "text", "uncertain"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}


class _ClaudeReader:
    """Messages API with image blocks. Structured output via
    ``output_config.format``; effort ``low`` because transcription needs
    eyes, not deliberation. Features an older model rejects (effort, schema)
    are dropped after the 400 that names them."""

    def __init__(self, key: str, model: str, ctx: Any):
        from . import translate_engines as te

        self.te = te
        self.key = key
        self.model = model
        self.transport = te.Transport("Anthropic", secrets={"anthropic": key}, check=ctx.check, log=ctx.log,
                                      timeout=180.0)
        family = model.split("-")[1] if model.startswith("claude-") and "-" in model[7:] else ""
        self.use_effort = family != "haiku"
        self.use_schema = True
        self.requests = 0

    def _body(self, content: List[dict]) -> dict:
        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": _CLAUDE_SYSTEM,
            "messages": [{"role": "user", "content": content}],
        }
        output: Dict[str, Any] = {}
        if self.use_schema:
            output["format"] = {"type": "json_schema", "schema": _CLAUDE_SCHEMA}
        if self.use_effort:
            output["effort"] = "low"
        if output:
            body["output_config"] = output
        return body

    def read(self, items: List[tuple], language: Optional[str]) -> Dict[str, tuple]:
        """``items`` = [(id, png_path)]; returns id -> (text, uncertain)."""
        name = languages.name_of(language) if language else "the original language"
        content: List[dict] = []
        for line_id, path in items:
            with open(path, "rb") as handle:
                data = base64.b64encode(handle.read()).decode("ascii")
            content.append({"type": "text", "text": f"Image {line_id}:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
        ids = ", ".join(i for i, _ in items)
        instruction = (
            f"Transcribe each image above exactly. The text is in {name}. Return one entry per image "
            f"with its id ({ids})."
        )
        if not self.use_schema:
            instruction += ' Answer with only a JSON object: {"lines": [{"id", "text", "uncertain"}]}.'
        content.append({"type": "text", "text": instruction})
        headers = {"x-api-key": self.key, "anthropic-version": self.te.ANTHROPIC_VERSION}
        for _ in range(3):
            try:
                data = self.transport.request("POST", f"{self.te.ENDPOINTS['anthropic']}/v1/messages",
                                              self._body(content), headers)
                break
            except self.te.ApiError as exc:
                message = exc.message.lower()
                if exc.status == 400 and self.use_effort and "effort" in message:
                    self.use_effort = False
                    continue
                if exc.status == 400 and self.use_schema and ("output_config" in message or "schema" in message):
                    self.use_schema = False
                    continue
                raise TaskError(f"Claude rejected the OCR request (HTTP {exc.status}): {exc.message}") from None
        else:
            raise TaskError("Claude kept rejecting the OCR request.")
        self.requests += 1
        stop = data.get("stop_reason")
        if stop == "refusal":
            return {}
        if stop == "max_tokens":
            raise self.te.Truncated()
        text = "".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text")
        parsed = self.te.parse_json_text(text)
        out: Dict[str, tuple] = {}
        for entry in parsed.get("lines") or []:
            if isinstance(entry, dict) and "id" in entry:
                out[str(entry["id"])] = (str(entry.get("text") or ""), bool(entry.get("uncertain")))
        return out


def run_claude(paths: List[str], language: Optional[str], options: dict, ctx: Any,
               progress: Optional[Progress] = None) -> List[LineRead]:
    key = (ctx.secrets or {}).get("anthropic")
    if not key:
        raise TaskError("Claude OCR needs an Anthropic API key (Settings > API keys).")
    reader = _ClaudeReader(key, str(options.get("claudeModel") or "claude-sonnet-5"), ctx)
    results: List[LineRead] = [LineRead(error="no result") for _ in paths]
    queue = [list(range(i, min(i + CLAUDE_BATCH, len(paths)))) for i in range(0, len(paths), CLAUDE_BATCH)]
    done = 0
    while queue:
        batch = queue.pop(0)
        ctx.check()
        items = [(f"L{i + 1}", paths[i]) for i in batch]
        try:
            answers = reader.read(items, language)
        except reader.te.Truncated:
            if len(batch) > 1:
                half = len(batch) // 2
                queue[:0] = [batch[:half], batch[half:]]
                continue
            answers = {}
        for i in batch:
            answer = answers.get(f"L{i + 1}")
            if answer is None:
                results[i] = LineRead(error="Claude returned no text for this line")
            else:
                text, uncertain = answer
                conf = CLAUDE_UNCERTAIN if uncertain or not text.strip() else CLAUDE_CONFIDENCE
                results[i] = LineRead(observations=[{"text": text, "confidence": conf, "box": None}])
        done += len(batch)
        if progress:
            progress(done, len(paths))
    ctx.log(f"Claude read {len(paths)} lines in {reader.requests} requests")
    return results


# ---------------------------------------------------------------- facade

RUNNERS = {
    "vision": run_vision,
    "tesseract": run_tesseract,
    "rapidocr": run_rapidocr,
    "claude": run_claude,
}


def recognize(engine: str, paths: List[str], language: Optional[str], options: dict, ctx: Any,
              progress: Optional[Progress] = None) -> List[LineRead]:
    runner = RUNNERS.get(engine)
    if runner is None:
        raise TaskError(f"Unknown OCR engine: {engine}")
    if not paths:
        return []
    started = time.monotonic()
    results = runner(paths, language, options, ctx, progress)
    ctx.log(f"{ENGINE_LABELS[engine]} read {len(paths)} line images in {time.monotonic() - started:.1f} s")
    return results


def engine_statuses(secrets: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """``EngineStatus`` for each OCR engine. Cheap: never compiles or spawns
    anything except ``tesseract --version``-free lookups on PATH."""
    out: List[Dict[str, Any]] = []
    reason = _vision_reason()
    out.append({"id": "vision", "label": ENGINE_LABELS["vision"], "available": reason is None, "reason": reason,
                "pack": None})
    binary = tesseract_path()
    out.append({"id": "tesseract", "label": ENGINE_LABELS["tesseract"], "available": bool(binary),
                "reason": None if binary else "Tesseract is not installed. " + _INSTALL_HINT, "pack": None})
    try:
        from . import packs

        rapid = bool(packs.python_for(RAPID_PACK))
    except Exception:  # noqa: BLE001 - a broken packs module must not hide the other engines
        rapid = False
    out.append({"id": "rapidocr", "label": ENGINE_LABELS["rapidocr"], "available": rapid,
                "reason": None if rapid else "Install the OCR (RapidOCR) pack.", "pack": RAPID_PACK})
    has_key = True if secrets is None else bool(secrets.get("anthropic"))
    out.append({"id": "claude", "label": ENGINE_LABELS["claude"], "available": has_key,
                "reason": None if has_key else "Add an Anthropic API key in Settings.", "pack": None})
    return out
