"""Subtitles for videos that have none: the speech-recognition front end.

Whisper itself runs in an engine pack (mlx-whisper on Apple Silicon,
faster-whisper anywhere) through ``workers/asr_worker.py``; this module does
everything around it in the engine's own numpy-only Python:

1.  extract the chosen audio stream as 16 kHz mono WAV;
2.  optionally isolate the voices with Demucs (demucs pack), which helps
    when a score or a song sits under the dialogue;
3.  find the speech (Silero from the asr-faster pack when installed, else the
    built-in detector in ``vad.py``) and hand Whisper only those stretches --
    Whisper invents text in long silences and music, and never in speech;
4.  recognise, with word timestamps;
5.  remove what Whisper is known to invent: the credits of the subtitle
    communities it was trained on ("Subtitles by the Amara.org community",
    「ご視聴ありがとうございました」, 明镜与点点栏目), loops of one phrase
    or one word, segments it itself scored as probably not speech or failed
    to decode, Latin words looping through a transcript in another script,
    and words that fall outside the detected speech. Short lines are not
    suspect for being short: a whispered name or a one-word shout is speech;
6.  cut the words into subtitles with the style rules for the language
    (``style.segment_words``).

SDH: sound descriptions that Whisper itself writes ("[Music]", 「(拍手)」, ♪)
are kept as their own subtitles when SDH is on and dropped otherwise. No
[music] cues are *invented* for stretches without speech: telling music from
other loud non-speech (rain, traffic, a crowd) needs a classifier this
engine does not have, and a wrong [music] label is worse than none.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..media import Cancelled, MediaError, _popen_kwargs, _terminate, ffmpeg_path, probe
from . import languages, packs
from .model import Cue, SubtitleDoc, Word
from .tasks import TaskContext, TaskError, TaskResult, preview_of, write_subtitle

SR = 16000

#: GenerateOptions defaults (types.ts DEFAULT_OPTIONS.generate).
DEFAULTS: Dict[str, Any] = {
    "engine": "mlx-whisper",
    "model": "large-v3",
    "language": "auto",
    "task": "transcribe",
    "vocalIsolation": False,
    "vad": True,
    "beamSize": 5,
    "initialPrompt": "",
    "suppressHallucinations": True,
    "stylePreset": "netflix",
    "sdh": False,
}

#: engine id -> (pack id, worker engine name)
ENGINES: Dict[str, Tuple[str, str]] = {
    "mlx-whisper": ("asr-mlx", "mlx"),
    "faster-whisper": ("asr-faster", "faster"),
}

# Speech regions. Silences shorter than MIN_SILENCE_S stay inside a region --
# Whisper punctuates better with natural pauses, and cutting them buys
# nothing -- while longer ones are removed. PAD_S keeps a region from
# clipping a soft first or last syllable. These are faster-whisper's own
# vad_filter defaults, which were tuned for exactly this job.
MIN_SILENCE_S = 2.0
PAD_S = 0.4
MIN_SPEECH_S = 0.1
SILERO_ON, SILERO_OFF = 0.5, 0.35

# Whisper's decoding thresholds (its defaults) and the silence after which
# a lone segment surrounded by silence is treated as a likely hallucination.
COMPRESSION_RATIO_THRESHOLD = 2.4
LOGPROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD = 0.6
HALLUCINATION_SILENCE_S = 2.0


# ---------------------------------------------------------------- statuses


def engine_statuses() -> List[dict]:
    out = []
    for engine_id, (pack_id, _worker) in ENGINES.items():
        spec = packs.PACKS[pack_id]
        available = packs.python_for(pack_id) is not None
        if available:
            reason = None
        elif not packs.supported(spec):
            reason = "Needs a Mac with Apple Silicon"
        else:
            reason = f"Install the {spec.label} pack"
        out.append({
            "id": engine_id,
            "label": "mlx-whisper (Apple Silicon GPU)" if engine_id == "mlx-whisper" else "faster-whisper (CPU / NVIDIA)",
            "available": available,
            "reason": reason,
            "pack": pack_id,
        })
    return out


def _options(raw: Optional[dict]) -> Dict[str, Any]:
    opts = dict(DEFAULTS)
    opts.update({k: v for k, v in (raw or {}).items() if v is not None})
    if opts["engine"] not in ENGINES:
        raise TaskError(f"Unknown speech recognition engine: {opts['engine']}")
    language = str(opts.get("language") or "auto")
    if language != "auto":
        code = languages.normalize(language)
        if not code:
            raise TaskError(f"Unknown language: {language}")
        language = code
    opts["language"] = language
    if opts["task"] not in ("transcribe", "translate"):
        raise TaskError(f"Unknown recognition task: {opts['task']}")
    opts["beamSize"] = max(1, min(10, int(opts.get("beamSize") or 1)))
    opts["initialPrompt"] = str(opts.get("initialPrompt") or "").strip()
    return opts


# ------------------------------------------------------------------- audio


def extract_audio(
    path: str,
    audio_track: int,
    out_path: str,
    ctx: TaskContext,
    rate: int = SR,
    channels: int = 1,
    stage: str = "Extracting audio",
) -> float:
    """Decode stream ``0:a:<audio_track>`` to a 16-bit PCM WAV; returns its
    duration in seconds. Progress comes from FFmpeg's own -progress output."""
    if not os.path.isfile(path):
        raise TaskError(f"Video not found: {path}")
    try:
        total = probe(path, ctx.token).duration or 0.0
    except MediaError:
        total = 0.0
    command = [
        ffmpeg_path(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", path, "-map", f"0:a:{max(0, int(audio_track))}", "-vn", "-sn", "-dn",
        "-ac", str(channels), "-ar", str(rate), "-c:a", "pcm_s16le",
        "-progress", "pipe:1", "-nostats", out_path,
    ]
    ctx.check()
    kwargs = _popen_kwargs()
    with tempfile.TemporaryFile() as err:
        kwargs["stderr"] = err
        try:
            process = subprocess.Popen(command, **kwargs)
        except OSError as exc:
            raise TaskError(f"Could not start FFmpeg: {exc}") from exc
        ctx.token.register(process)
        try:
            assert process.stdout is not None
            for raw in iter(process.stdout.readline, b""):
                key, _, value = raw.decode("ascii", "replace").strip().partition("=")
                if key in ("out_time_us", "out_time_ms") and total > 0 and value.isdigit():
                    ctx.progress(int(min(99, 100 * int(value) / 1e6 / total)), stage)
            process.wait()
        finally:
            ctx.token.unregister(process)
            _terminate(process)
        if ctx.token.cancelled:
            raise Cancelled("operation cancelled")
        if process.returncode != 0:
            err.seek(0)
            detail = err.read().decode("utf-8", "replace").strip().splitlines()
            tail = detail[-1] if detail else f"exit code {process.returncode}"
            if any("matches no streams" in line for line in detail):
                raise TaskError(f"{os.path.basename(path)} has no audio track {int(audio_track) + 1}.")
            raise TaskError(f"Could not read the audio of {os.path.basename(path)}: {tail}")
    ctx.progress(100, stage)
    return _wav_duration(out_path)


def _wav_duration(path: str) -> float:
    import wave

    with wave.open(path, "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def isolate_vocals(media: dict, ctx: TaskContext) -> str:
    """16 kHz mono WAV of the voices only, separated by Demucs."""
    pack_id = "demucs"
    if packs.python_for(pack_id) is None:
        raise TaskError("Voice isolation needs the Voice isolation (Demucs) pack. Install it, or turn voice isolation off.")
    model = packs.model_spec(pack_id, "htdemucs")
    if packs.model_path(model) is None:
        packs.install(pack_id, ctx.progress, ctx.log, ctx.token, model="htdemucs")
    stereo = os.path.join(ctx.workdir, "audio44k.wav")
    extract_audio(media["path"], int(media.get("audioTrack") or 0), stereo, ctx, rate=44100, channels=2,
                  stage="Extracting audio for Demucs")
    vocals = os.path.join(ctx.workdir, "vocals16k.wav")
    result = packs.run_worker(pack_id, "demucs_worker.py", {
        "audio": stereo, "out": vocals, "outRate": SR,
        "modelPath": packs.model_path(model), "model": "htdemucs",
    }, ctx)
    try:
        os.remove(stereo)
    except OSError:
        pass
    ctx.log(f"Voices isolated with Demucs on {result.get('device')} in {result.get('seconds')} s")
    return vocals


# ----------------------------------------------------------------- speech


def _hysteresis_runs(probs: np.ndarray, on: float, off: float) -> List[Tuple[int, int]]:
    """[start, end) frames of stretches above ``off`` that reach ``on``."""
    above = probs >= off
    if not above.any():
        return []
    edges = np.diff(np.concatenate([[0], above.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    hits = np.concatenate([[0], np.cumsum(probs >= on)])
    return [(int(s), int(e)) for s, e in zip(starts, ends) if hits[e] > hits[s]]


def speech_regions(
    probs: np.ndarray,
    hop_s: float,
    duration: float,
    on: float = 0.5,
    off: float = 0.5,
    min_silence_s: float = MIN_SILENCE_S,
    pad_s: float = PAD_S,
    min_speech_s: float = MIN_SPEECH_S,
) -> List[Tuple[float, float]]:
    """Padded, merged (start, end) seconds of speech from per-frame
    probabilities: runs shorter than ``min_speech_s`` are dropped, runs
    closer than ``min_silence_s`` joined, each padded by ``pad_s``."""
    runs = [(s * hop_s, e * hop_s) for s, e in _hysteresis_runs(np.asarray(probs, dtype=np.float32), on, off)]
    runs = [(s, e) for s, e in runs if e - s >= min_speech_s]
    merged: List[List[float]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] < min_silence_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out: List[Tuple[float, float]] = []
    for s, e in merged:
        s, e = max(0.0, s - pad_s), min(duration, e + pad_s)
        # Padding shrinks a gap; one still shorter than min_silence_s is a
        # pause like any other. Cutting it would only swap it for the
        # worker's one-second spacer and lose whatever quiet speech it held
        # (a whispered name between two louder ones).
        if out and s - out[-1][1] < min_silence_s:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


#: A gap between two speech regions that is at most LOUD_GAP_MAX_S long and
#: no more than LOUD_GAP_DB quieter than the louder of them is recognised
#: too. People pause in dialogue and the sound drops; where it stays as loud
#: as the voices, a score or effects are playing and the detector cannot hear
#: speech under them. In the Mugen Train station scene the energy detector
#: cut 220.9-228.8 s -- a train and music at -30 dBFS, as loud as the lines
#: around it -- and with it "W-wait up!" and "Tanjiro! Inosuke!". A quiet
#: gap is still cut: silence is where Whisper invents text.
LOUD_GAP_MAX_S = 10.0
LOUD_GAP_DB = 6.0


def _level_db(wav: str, start: float, end: float) -> float:
    """RMS level of ``wav`` between two times, in dBFS."""
    import wave

    with wave.open(wav, "rb") as handle:
        rate = handle.getframerate()
        first = max(0, min(handle.getnframes(), int(start * rate)))
        handle.setpos(first)
        raw = handle.readframes(max(0, int(end * rate) - first))
        width, channels = handle.getsampwidth(), handle.getnchannels()
    if width != 2 or not raw:
        return -120.0
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    return float(10 * np.log10(np.mean(x * x) + 1e-12))


def join_loud_gaps(
    regions: List[Tuple[float, float]],
    level: Any,
    max_gap_s: float = LOUD_GAP_MAX_S,
    drop_db: float = LOUD_GAP_DB,
) -> List[Tuple[float, float]]:
    """Join regions across short gaps that are about as loud as the speech
    on either side. ``level(start, end)`` gives the dBFS of a stretch."""
    if len(regions) < 2:
        return list(regions)
    levels = [level(s, e) for s, e in regions]
    out = [regions[0]]
    loudest = levels[0]
    for (s, e), db in zip(regions[1:], levels[1:]):
        gap = s - out[-1][1]
        if 0 < gap <= max_gap_s and level(out[-1][1], s) >= max(loudest, db) - drop_db:
            out[-1] = (out[-1][0], e)
            # The joined region's level is that of the voice just before
            # the next gap, not of everything since the first join.
            loudest = db
        else:
            out.append((s, e))
            loudest = db
    return out


class Speech:
    """What the speech detector found: padded regions for recognition, and
    the per-frame decision (``frames``, ``hop_s``) for snapping word edges."""

    def __init__(self, regions: List[Tuple[float, float]], frames: np.ndarray, hop_s: float, engine: str):
        self.regions = regions
        self.frames = frames
        self.hop_s = hop_s
        self.engine = engine


def detect_speech(wav: str, duration: float, ctx: TaskContext) -> Speech:
    """Speech in ``wav``.

    Silero (from the asr-faster pack) when it is installed: it is a trained
    network and holds up under music and whispering. Otherwise the built-in
    detector, whose runs are already smoothed and thresholded at 0.5.
    """
    if packs.python_for("asr-faster") is not None:
        try:
            result = packs.run_worker("asr-faster", "vad_worker.py", {"audio": wav, "outDir": ctx.workdir}, ctx)
            probs = np.fromfile(result["probsPath"], dtype="<f4")
            hop = float(result["hopS"])
            frames = np.zeros(len(probs), dtype=bool)
            for s, e in _hysteresis_runs(probs, SILERO_ON, SILERO_OFF):
                frames[s:e] = True
            regions = speech_regions(probs, hop, duration, on=SILERO_ON, off=SILERO_OFF)
            regions = join_loud_gaps(regions, lambda a, b: _level_db(wav, a, b))
            return Speech(regions, frames, hop, "silero")
        except TaskError as exc:
            ctx.log(f"Silero speech detection failed ({exc}); using the built-in detector")
    from . import vad

    probs, hop = vad.speech_activity(wav, 0, "energy", ctx.token, lambda p: ctx.progress(int(p), "Detecting speech"))
    regions = join_loud_gaps(speech_regions(probs, hop, duration), lambda a, b: _level_db(wav, a, b))
    return Speech(regions, np.asarray(probs) > 0.5, hop, "energy")


#: How far the last word before a pause may be stretched to where the
#: speech actually stops.
SNAP_EXTEND_S = 0.4


def snap_words(words: List[Word], speech: Speech) -> int:
    """Pull word edges onto the detected speech; returns how many moved.

    Whisper's word timings come from attention alignment and run early: on
    clips with known timing the first word of a line started 300-470 ms
    before the voice did, with or without speech detection, and a line's
    last word ended 50-450 ms early. So a word that starts in frames the
    detector heard as silence starts where the speech starts instead, and
    the last word before a pause ends where the speech ends (at most
    SNAP_EXTEND_S later). A word lying wholly in the silence just before
    the voice (the alignment squeezed a first syllable into the pause) moves
    onto the onset. Words only ever move toward detected speech.
    """
    frames, hop = speech.frames, speech.hop_s
    n = len(frames)
    if n == 0 or hop <= 0:
        return 0
    moved = 0
    for i, word in enumerate(words):
        a = max(0, min(n - 1, int(word.start / hop)))
        b = max(a + 1, min(n, int(np.ceil(word.end / hop))))
        span = frames[a:b]
        onset = None
        if span.any():
            if not span[0]:
                onset = (a + int(np.argmax(span))) * hop
        else:
            ahead = frames[b:min(n, b + int(np.ceil(0.6 / hop)))]
            if ahead.any():
                onset = (b + int(np.argmax(ahead))) * hop
        if onset is not None and onset > word.start:
            if onset <= word.end - 0.05:
                word.start = onset
            else:
                # The word sits (almost) wholly in the silence before the
                # voice -- the alignment squeezed a first syllable into
                # the pause: move it onto the onset, keeping its length.
                length = word.end - word.start
                cap = words[i + 1].end if i + 1 < len(words) else onset + length
                word.start = onset
                word.end = max(onset + 0.02, min(onset + length, cap))
            moved += 1
        following = words[i + 1].start if i + 1 < len(words) else float("inf")
        if following - word.end > 0.3:
            k = max(0, min(n - 1, int(word.end / hop)))
            limit = min(following - 0.05, word.end + SNAP_EXTEND_S)
            while k < n and frames[k] and (k + 1) * hop <= limit:
                k += 1
            if k * hop > word.end:
                word.end = k * hop
                moved += 1
    return moved


# ------------------------------------------------------------ hallucinations

#: What Whisper writes over silence and music: the sign-offs and credit lines
#: of the video subtitles it learned from. Patterns match the *normalised*
#: text (lower case, letters and digits only). ALWAYS patterns are credits
#: nobody says in a film and remove the segment wherever they appear. The
#: others remove it only when together they cover most of it (70%), so a
#: character who says "thank you for watching the kids tonight" keeps the line.
_ALWAYS = [
    r"amaraorg", r"castingwords", r"dimatorzok", r"明镜.{0,6}点点栏目", r"点点栏目",
    r"請不吝點贊|请不吝点赞", r"字幕志愿者|字幕志願者", r"untertitel(?:ung)?(?:imauftrag)?deszdf",
    r"soustitrage(?:société)?radiocanada", r"soustitragest501", r"acuradiqtss",
    r"mbc뉴스[가-힣]{2,4}입니다", r"자막제공및자막싱크",
]
_PHRASES: Dict[str, List[str]] = {
    "en": [r"thanks?(?:you)?(?:so?much|verymuch)?forwatching(?:thisvideo)?",
           r"(?:please)?(?:like(?:and)?)?(?:dont(?:forget)?to)?subscribe(?:tomychannel|tothechannel|andlike)?",
           r"subtitlesby\w*", r"captionsby\w*", r"seeyou(?:inthe)?next(?:video|time|one)",
           r"ifyouenjoyedthisvideo"],
    "ja": [r"(?:最後まで)?ご(?:視聴|覧)(?:いただき)?(?:本当に|誠に)?ありがとうございま(?:した|す)",
           r"(?:チャンネル登録|高評価|グッドボタン)(?:と|や)?(?:チャンネル登録|高評価|グッドボタン)?"
           r"(?:を)?(?:よろしく)?(?:お願いします|お願いいたします|してね|してください)?",
           r"次の動画でお会いしましょう", r"また次回の動画で"],
    "ko": [r"(?:영상을)?시청해주셔서감사합니다", r"구독(?:과|하고)?좋아요(?:부탁드립니다|눌러주세요)?",
           r"구독과알림설정(?:부탁드립니다)?", r"다음영상에서만나요"],
    "zh": [r"谢谢(?:大家)?(?:收看|观看)|謝謝(?:大家)?(?:收看|觀看)", r"感谢(?:您的)?观看|感謝(?:您的)?觀看",
           r"(?:请|請)?(?:订阅|訂閱).{0,6}(?:点赞|點贊|转发|轉發|打赏|打賞)", r"欢迎订阅\w*|歡迎訂閱\w*",
           r"字幕由\w{0,12}提供"],
    "fr": [r"merci(?:d|de)?avoir(?:regardé|visionné)(?:cettevidéo)?", r"(?:nhésitezpasà)?(?:vous)?abonnez(?:vous)?",
           r"soustitres?réalisés?par\w*", r"àbientôtpourunenouvellevidéo"],
    "es": [r"graciasporver(?:el|este)?(?:video|vídeo)?", r"suscríbete(?:alcanal)?",
           r"subtítulos(?:realizados)?por\w*", r"nosvemosenel(?:próximo|siguiente)(?:video|vídeo)"],
    "it": [r"grazie(?:mille)?perlavisione", r"sottotitoli(?:creati|e revisione)?(?:dalla|acura)\w*",
           r"iscrivitialcanale|iscrivetevi(?:alcanale)?"],
    "de": [r"(?:vielen)?dankfürszuschauen", r"untertitel(?:der|von)\w*community",
           r"abonniert(?:den|meinen)kanal", r"biszumnächstenmal"],
    "pt": [r"obrigad[oa]porassistir", r"inscrevase(?:nocanal)?", r"legendas?(?:pela|por)\w*"],
    "ru": [r"спасибозапросмотр", r"подписывайтесь(?:на(?:наш)?канал)?", r"редакторсубтитров\w*",
           r"субтитры(?:сделал|создавал)\w*", r"продолжениеследует"],
    "nl": [r"bedanktvoorhetkijken", r"ondertiteling(?:door)?\w*", r"ondertiteld(?:door)?\w*"],
}
_ALWAYS_RE = [re.compile(p) for p in _ALWAYS]
_PHRASE_RE = {k: [re.compile(p) for p in v] for k, v in _PHRASES.items()}
PHRASE_COVERAGE = 0.7

#: A segment that is only a sound description: [Music], (拍手), ♪ ... ♪
_SOUND_RE = re.compile(r"^\s*(?:[\[\(（【〔♪♫♬*].*[\]\)）】〕♪♫♬*]|[♪♫♬\s]+)\s*$")


def _norm(text: str) -> str:
    """Lower case, letters and digits only (every script)."""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LN")


def _known_hallucination(text: str, language: Optional[str]) -> Optional[str]:
    norm = _norm(text)
    if not norm:
        return None
    for rx in _ALWAYS_RE:
        if rx.search(norm):
            return "subtitle credit"
    patterns = list(_PHRASE_RE.get(language or "", []))
    if language != "en":
        patterns += _PHRASE_RE["en"]  # Whisper writes English sign-offs over any language
    covered = np.zeros(len(norm), dtype=bool)
    for rx in patterns:
        for match in rx.finditer(norm):
            covered[match.start():match.end()] = True
    if covered.mean() >= PHRASE_COVERAGE:
        return "known phrase"
    return None


def _collapse_loops(words: List[dict], min_repeats: int = 4) -> Tuple[List[dict], int]:
    """Cut a phrase repeated ``min_repeats`` or more times in a row down to
    two -- Whisper's decoding loop ("はい、はい、はい、はい…" for a minute)
    -- and return the kept words and how many loops were cut."""
    tokens = [_norm(w["text"]) for w in words]
    keep = [True] * len(words)
    loops = 0
    i = 0
    while i < len(tokens):
        best: Optional[Tuple[int, int]] = None
        for n in range(1, 9):
            unit = tokens[i:i + n]
            if len(unit) < n or not "".join(unit):
                break
            reps = 1
            while tokens[i + reps * n:i + (reps + 1) * n] == unit:
                reps += 1
            if reps >= min_repeats and (best is None or reps * n > best[0] * best[1]):
                best = (reps, n)
        if best:
            reps, n = best
            for k in range(i + 2 * n, i + reps * n):
                keep[k] = False
            loops += 1
            i += reps * n
        else:
            i += 1
    return [w for w, k in zip(words, keep) if k], loops


#: Languages whose transcripts are not written in Latin letters. A Latin
#: word in one of them is either something really said in English (a
#: brand, "OK") or Whisper drifting into English over noise, as in
#: 「いても行くしかないよProblem」 followed by "Problem Problem Problem"
#: over a train scene of Mugen Train.
NON_LATIN = {
    "ja", "zh", "yue", "ko", "ru", "uk", "be", "bg", "mk", "kk", "mn", "tt", "ba", "tg",
    "el", "he", "yi", "ar", "fa", "ur", "ps", "sd", "hi", "mr", "ne", "sa", "bn", "as", "pa",
    "gu", "ta", "te", "kn", "ml", "si", "th", "lo", "my", "km", "ka", "hy", "am", "bo",
}
#: A Latin word that comes back within LATIN_REPEAT_S seconds in such a
#: transcript is a loop, not code-switching. Short all-capital words (OK,
#: TV, NHK) are said repeatedly for real, so they need LATIN_ACRONYM_REPEATS.
LATIN_REPEAT_S = 30.0
LATIN_ACRONYM_REPEATS = 4
#: A Latin word glued to the end of a line with no space and less
#: confidence than this was not heard; it was appended.
LATIN_GLUED_PROB = 0.5
#: A segment of one word said three or more times with a mean probability
#: below this is a decoding loop ("Problem Problem Problem", 「5555」);
#: 「はいはいはい」 said for real is heard with confidence.
LOOP_WORD_PROB = 0.5


def _mean_probability(words: List[dict], default: float) -> float:
    probs = [float(w["probability"]) for w in words if w.get("probability") is not None]
    return sum(probs) / len(probs) if probs else default


def _is_latin(norm: str) -> bool:
    return bool(norm) and all(unicodedata.name(ch, "").startswith("LATIN") for ch in norm)


def _single_token_loop(words: List[dict], language: Optional[str]) -> bool:
    tokens = [_norm(w.get("text") or "") for w in words]
    tokens = [t for t in tokens if t]
    if len(tokens) < 3 or len(set(tokens)) != 1:
        return False
    if language in NON_LATIN and _is_latin(tokens[0]):
        return True
    return _mean_probability(words, 1.0) < LOOP_WORD_PROB


def _latin_words(segments: List[dict]) -> List[Tuple[int, int, str, float, bool]]:
    """(segment index, word index, token, start, is an acronym) of every
    Latin word, sound tags ("[Music]") aside."""
    found = []
    for si, seg in enumerate(segments):
        if seg.get("sound") or _SOUND_RE.match(seg.get("text") or ""):
            continue
        for wi, word in enumerate(seg.get("words") or []):
            norm = _norm(word.get("text") or "")
            if _is_latin(norm):
                bare = (word.get("text") or "").strip().strip(".,!?;:'\"")
                found.append((si, wi, norm, float(word["start"]), bare.isupper() and len(bare) <= 4))
    return found


def _drop_stray_latin(kept: List[dict], language: Optional[str], heard: List[dict]) -> List[dict]:
    """Remove implausible Latin words from a transcript in a non-Latin
    script, in place; returns the removed entries. Repeats are counted over
    ``heard`` -- everything Whisper wrote, removed segments included -- as a
    loop dropped as a whole ("Problem Problem Problem") is what shows the
    same word glued onto the line before it is part of it."""
    if language not in NON_LATIN:
        return []
    found = _latin_words(kept)
    everywhere = [(norm, start) for _s, _w, norm, start, _a in _latin_words(heard)]
    drop: Dict[int, Dict[int, str]] = {}
    for si, wi, norm, start, acronym in found:
        repeats = sum(1 for other, t in everywhere if other == norm and abs(t - start) <= LATIN_REPEAT_S)
        reason = None
        if repeats >= (LATIN_ACRONYM_REPEATS if acronym else 2):
            reason = "repeated Latin word"
        else:
            words = kept[si]["words"]
            word = words[wi]
            before = _norm(words[wi - 1]["text"]) if wi else ""
            glued = (wi == len(words) - 1 and before and not _is_latin(before[-1])
                     and not (word.get("text") or "")[:1].isspace())
            if glued and word.get("probability") is not None and word["probability"] < LATIN_GLUED_PROB:
                reason = "stray Latin word"
        if reason:
            drop.setdefault(si, {})[wi] = reason
    removed = []
    for si in sorted(drop):
        seg = kept[si]
        words = seg["words"]
        gone = [words[wi] for wi in sorted(drop[si])]
        removed.append({"start": round(gone[0]["start"], 2), "end": round(gone[-1]["end"], 2),
                        "text": "".join(w["text"] for w in gone).strip(), "reason": next(iter(drop[si].values()))})
        rest = [w for wi, w in enumerate(words) if wi not in drop[si]]
        seg["words"] = rest
        seg["text"] = "".join(w["text"] for w in rest)
        if rest:
            seg["start"], seg["end"] = rest[0]["start"], rest[-1]["end"]
    kept[:] = [seg for seg in kept if not (seg.get("words") == [] and not seg.get("sound"))]
    return removed


def clean_segments(
    segments: List[dict],
    language: Optional[str],
    regions: Optional[Sequence[Tuple[float, float]]],
    prompt: str = "",
    sdh: bool = False,
    suppress: bool = True,
) -> Tuple[List[dict], List[dict]]:
    """Drop or trim what Whisper invented. Returns (kept, removed), where each
    removed entry is ``{"start", "end", "text", "reason"}``."""
    kept: List[dict] = []
    removed: List[dict] = []
    prompt_norm = _norm(prompt)

    def drop(seg: dict, reason: str) -> None:
        removed.append({"start": round(seg["start"], 2), "end": round(seg["end"], 2),
                        "text": seg["text"].strip(), "reason": reason})

    for seg in segments:
        text = seg.get("text") or ""
        if _SOUND_RE.match(text):
            if sdh:
                kept.append(dict(seg, sound=True))
            continue  # a sound tag without SDH is not a hallucination, just not wanted
        if not suppress:
            kept.append(seg)
            continue
        reason = _known_hallucination(text, language)
        norm = _norm(text)
        if not reason and prompt_norm and norm and (norm == prompt_norm or (prompt_norm in norm and len(norm) <= len(prompt_norm) + 4)):
            reason = "repeated the prompt"
        if not reason and seg.get("noSpeechProb", 0) > NO_SPEECH_THRESHOLD and seg.get("avgLogprob", 0) < -0.8:
            reason = "scored as not speech"
        if not reason and seg.get("noSpeechProb", 0) > 0.5 and seg.get("avgLogprob", 0) < -0.8 \
                and _mean_probability(seg.get("words") or [], 1.0) < 0.6:
            # Just under the threshold, but with words Whisper doubted too:
            # large-v3 read the opening score as 「エヴィトレックス」 (no speech
            # 0.57, logprob -0.87, words 0.54). Shouts on the same clip scored
            # no speech 0.00; whispered names 0.49 at logprob -0.30.
            reason = "scored as not speech"
        logprob = float(seg.get("avgLogprob", 0))
        if not reason and (logprob != logprob or seg.get("compressionRatio", 0) > COMPRESSION_RATIO_THRESHOLD):
            # Every temperature failed Whisper's own checks and it kept the
            # last try anyway (「5555」 over the opening music, logprob NaN).
            reason = "failed decoding"
        temperature = float(seg.get("temperature", 0))
        if not reason and temperature >= 0.6 and logprob < -1.4:
            reason = "low confidence"
        if not reason and temperature > 0 and logprob < LOGPROB_THRESHOLD + 0.2 \
                and _mean_probability(seg.get("words") or [], 1.0) < 0.6:
            # Greedy decoding failed and a sampled retry only just passed,
            # with words Whisper itself doubted: noise read as words over a
            # score (「エンディボロックス」, logprob -0.97, words 0.53).
            reason = "low confidence"
        if reason:
            drop(seg, reason)
            continue
        words = seg.get("words") or []
        if regions:
            inside = [w for w in words if _overlaps(w, regions)]
            if words and not inside:
                drop(seg, "outside detected speech")
                continue
            words = inside
        if _single_token_loop(words, language):
            drop(seg, "repetition loop")
            continue
        words, loops = _collapse_loops(words)
        if loops:
            removed.append({"start": round(seg["start"], 2), "end": round(seg["end"], 2),
                            "text": seg["text"].strip(), "reason": "repetition loop"})
        seg = dict(seg, words=words)
        if words:
            seg["text"] = "".join(w["text"] for w in words)
        # The same line three or more times in a row is a loop across
        # segments; the first copy stays.
        if norm and len(kept) >= 2 and all(_norm(k["text"]) == norm for k in kept[-2:]) and len(norm) >= 2:
            drop(seg, "repeated line")
            continue
        kept.append(seg)
    if suppress:
        removed.extend(_drop_stray_latin(kept, language, segments))
        removed.sort(key=lambda r: r["start"])
    return kept, removed


def _overlaps(word: dict, regions: Sequence[Tuple[float, float]], margin: float = 0.2) -> bool:
    start, end = float(word["start"]), float(word["end"])
    for s, e in regions:
        if end >= s - margin and start <= e + margin:
            return True
    return False


# --------------------------------------------------------------- transcribe


def _ensure_model(pack_id: str, model_id: str, ctx: TaskContext) -> str:
    model = packs.model_spec(pack_id, model_id)
    path = packs.model_path(model)
    if path:
        return path
    ctx.log(f"The {model.label} model is not downloaded yet; downloading it ({model.download_bytes / 1e9:.1f} GB)")
    packs.install(pack_id, ctx.progress, ctx.log, ctx.token, model=model_id)
    path = packs.model_path(model)
    if not path:
        raise TaskError(f"The {model.label} model could not be downloaded.")
    return path


def transcribe(media: dict, options: dict, ctx: TaskContext) -> Tuple[List[Word], str, Dict[str, Any]]:
    """Recognise the speech of ``media`` (a MediaRef: path, audioTrack).

    Returns the words (Whisper's text, leading spaces included, on the
    file's own timeline), the language (ISO 639-1; "en" for translate) and a
    report of what was done.
    """
    opts = _options(options)
    pack_id, worker_engine = ENGINES[opts["engine"]]
    spec = packs.PACKS[pack_id]
    if packs.python_for(pack_id) is None:
        if not packs.supported(spec):
            raise TaskError(f"{opts['engine']} needs a Mac with Apple Silicon; choose faster-whisper instead.")
        raise TaskError(f"Install the {spec.label} pack first (Settings > Engine packs).")
    if not media or not media.get("path"):
        raise TaskError("Choose a video or audio file to make subtitles for.")
    started = time.monotonic()
    model_dir = _ensure_model(pack_id, opts["model"], ctx)

    wav = os.path.join(ctx.workdir, "audio16k.wav")
    duration = extract_audio(media["path"], int(media.get("audioTrack") or 0), wav, ctx)
    if duration < 0.1:
        raise TaskError("The audio track is empty.")
    asr_wav = isolate_vocals(media, ctx) if opts["vocalIsolation"] else wav

    regions: Optional[List[Tuple[float, float]]] = None
    speech: Optional[Speech] = None
    vad_engine = None
    if opts["vad"]:
        speech = detect_speech(asr_wav, duration, ctx)
        regions, vad_engine = speech.regions, speech.engine
        speech_s = sum(e - s for s, e in regions)
        ctx.log(f"Speech detected ({vad_engine}): {speech_s:.0f} s in {len(regions)} stretches of {duration:.0f} s")
        if not regions:
            ctx.log("No speech was detected; nothing to recognise.")

    ctx.check()
    request = {
        "engine": worker_engine,
        "modelPath": model_dir,
        "audio": asr_wav,
        "language": None if opts["language"] == "auto" else opts["language"],
        "task": opts["task"],
        "beamSize": opts["beamSize"],
        "initialPrompt": opts["initialPrompt"],
        "speech": [[round(s, 3), round(e, 3)] for s, e in regions] if regions is not None else None,
        "compressionRatioThreshold": COMPRESSION_RATIO_THRESHOLD,
        "logprobThreshold": LOGPROB_THRESHOLD,
        "noSpeechThreshold": NO_SPEECH_THRESHOLD,
        "hallucinationSilenceThreshold": HALLUCINATION_SILENCE_S if opts["suppressHallucinations"] else None,
    }
    if regions == []:
        result = {"segments": [], "language": request["language"], "languageProbability": None,
                  "engine": opts["engine"], "decoding": "", "device": "", "loadS": 0, "recogniseS": 0}
    else:
        result = packs.run_worker(pack_id, "asr_worker.py", request, ctx)

    spoken = languages.normalize(result.get("language")) or result.get("language") or request["language"]
    kept, removed = clean_segments(
        result.get("segments") or [], spoken, regions, opts["initialPrompt"], opts["sdh"], opts["suppressHallucinations"],
    )
    for item in removed:
        ctx.log(f"Removed ({item['reason']}) at {_clock(item['start'])}: {item['text'][:80]}")

    words: List[Word] = []
    sounds: List[Cue] = []
    for seg in kept:
        if seg.get("sound"):
            sounds.append(Cue(start=float(seg["start"]), end=float(seg["end"]), text=seg["text"].strip()))
            continue
        for w in seg.get("words") or []:
            if (w.get("text") or "").strip():
                words.append(Word(start=float(w["start"]), end=max(float(w["end"]), float(w["start"])),
                                  text=str(w["text"]), probability=w.get("probability")))
    words.sort(key=lambda w: (w.start, w.end))
    snapped = snap_words(words, speech) if speech is not None else 0
    language = "en" if opts["task"] == "translate" else (spoken or "und")
    elapsed = time.monotonic() - started
    info: Dict[str, Any] = {
        "language": language,
        "spokenLanguage": spoken,
        "languageProbability": result.get("languageProbability"),
        "durationS": round(duration, 2),
        "speechS": round(sum(e - s for s, e in regions), 2) if regions is not None else None,
        "elapsedS": round(elapsed, 2),
        "realtimeFactor": round(elapsed / duration, 3) if duration else None,
        "engine": opts["engine"],
        "model": opts["model"],
        "decoding": result.get("decoding"),
        "device": result.get("device"),
        "vad": vad_engine,
        "vocalIsolation": bool(opts["vocalIsolation"]),
        "segments": len(kept),
        "snappedWordEdges": snapped,
        "removed": removed,
        "removedHallucinations": len(removed),
        "soundCues": sounds,
        "loadS": result.get("loadS"),
        "recogniseS": result.get("recogniseS"),
    }
    return words, language, info


def _clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ------------------------------------------------------------------- cues


def make_cues(words: List[Word], language: Optional[str], opts: Dict[str, Any]) -> List[Cue]:
    """Words -> subtitles with the style rules for the language."""
    try:
        from . import style
    except ImportError:
        style = None  # type: ignore[assignment]
    if style is not None:
        rules = style.rules_for(language, {"preset": opts.get("stylePreset") or "netflix", "sdh": bool(opts.get("sdh"))})
        return style.segment_words(words, rules, language)
    return _fallback_cues(words, language)


def _fallback_cues(words: List[Word], language: Optional[str]) -> List[Cue]:
    """Plain segmentation for when the style module is unavailable: a cue
    ends at a pause, a sentence end, 7 seconds or two lines' worth of text."""
    cjk = language in ("ja", "zh", "yue")
    line = 16 if cjk else 42
    cues: List[Cue] = []
    group: List[Word] = []

    def flush() -> None:
        if not group:
            return
        text = "".join(w.text for w in group).strip()
        if len(text) > line:
            cut = len(text) // 2
            if not cjk:
                spaces = [i for i, ch in enumerate(text) if ch == " "]
                if spaces:
                    cut = min(spaces, key=lambda i: abs(i - len(text) / 2))
                    text = text[:cut] + "\n" + text[cut + 1:]
                    cut = -1
            if cut >= 0:
                text = text[:cut] + "\n" + text[cut:]
        probs = [w.probability for w in group if w.probability is not None]
        cues.append(Cue(start=group[0].start, end=group[-1].end + 0.5, text=text,
                        confidence=round(sum(probs) / len(probs), 3) if probs else None))
        group.clear()

    for word in words:
        if group:
            text = "".join(w.text for w in group + [word]).strip()
            if (word.start - group[-1].end > 1.0 or len(text) > 2 * line
                    or word.end - group[0].start > 7.0
                    or group[-1].text.strip()[-1:] in ".?!。？！"):
                flush()
        group.append(word)
    flush()
    for a, b in zip(cues, cues[1:]):
        a.end = min(a.end, b.start - 0.083)
    for cue in cues:
        cue.end = max(cue.end, cue.start + 0.833)
    return cues


def _merge_sounds(cues: List[Cue], sounds: List[Cue]) -> List[Cue]:
    """Add Whisper's own sound descriptions where no dialogue covers them."""
    out = list(cues)
    for sound in sounds:
        if not any(c.start < sound.end and sound.start < c.end for c in cues):
            out.append(sound)
    return sorted(out, key=lambda c: (c.start, c.end))


# -------------------------------------------------------------------- task

#: Most removed segments listed in a job report (the count is always exact).
REPORT_REMOVED_MAX = 50
#: A decoding loop can be hundreds of characters (「ビビビ…」 for 18 s).
REPORT_TEXT_MAX = 120


def run_task(job: dict, ctx: TaskContext) -> TaskResult:
    media = (job.get("input") or {}).get("video") or {}
    if not media.get("path"):
        raise TaskError("Choose a video or audio file to make subtitles for.")
    opts = _options(job.get("options"))
    warnings: List[str] = []
    if opts["task"] == "translate" and opts["model"] == "large-v3-turbo":
        warnings.append("large-v3-turbo was not trained to translate; large-v3 or medium translate far better.")
    if opts["engine"] == "mlx-whisper" and opts["beamSize"] > 1:
        ctx.log("mlx-whisper has no beam search; decoding greedily with best-of sampling on retries.")

    words, language, info = transcribe(media, opts, ctx)
    ctx.progress(0, "Making subtitles")
    cues = make_cues(words, language, opts)
    if opts["sdh"] and info.get("soundCues"):
        cues = _merge_sounds(cues, info["soundCues"])
    doc = SubtitleDoc(cues=cues, language=language, source_format="asr")
    doc.meta["generatedBy"] = f"{info['engine']} {info['model']}"
    output = write_subtitle(doc, {"path": media["path"]}, job, "", ctx, language=language)
    output.label = f"{languages.name_of(language)} (generated)"
    ctx.progress(100, "Making subtitles")

    removed = info["removedHallucinations"]
    report = {
        "language": language,
        "languageName": languages.name_of(language),
        "languageProbability": info.get("languageProbability"),
        "durationS": info["durationS"],
        "elapsedS": info["elapsedS"],
        "realtimeFactor": info["realtimeFactor"],
        "words": len(words),
        "cues": len(cues),
        "engine": info["engine"],
        "model": info["model"],
        "removedHallucinations": removed,
        # What was removed and why, so a missing line can be told from a
        # line Whisper never wrote. Capped: a film with a bad music bed can
        # lose hundreds, and the report travels to the UI as one event.
        "removed": [dict(r, text=r["text"] if len(r["text"]) <= REPORT_TEXT_MAX else r["text"][:REPORT_TEXT_MAX] + "…")
                    for r in info["removed"][:REPORT_REMOVED_MAX]],
        "speechS": info.get("speechS"),
        "vad": info.get("vad"),
        "decoding": info.get("decoding"),
        "device": info.get("device"),
    }
    if not words:
        warnings.append("No speech was recognised.")
    detected = ""
    if opts["language"] == "auto" and info.get("languageProbability") is not None:
        detected = f" (detected, {info['languageProbability']:.0%} sure)"
    spoken = languages.name_of(info.get("spokenLanguage"))
    what = f"Translated {spoken} speech into English" if opts["task"] == "translate" else f"Transcribed {spoken}{detected}"
    speed = info["durationS"] / max(info["elapsedS"], 1e-6)
    summary = (
        f"{what}: {len(cues)} subtitles from {_clock(info['durationS'])} of audio with "
        f"{info['engine']} {info['model']} in {_clock(info['elapsedS'])} "
        f"({speed:.1f}x real-time speed)"
        + (f"; removed {removed} likely hallucination{'s' if removed != 1 else ''}." if removed else ".")
    )
    return TaskResult(outputs=[output], report=report, summary=summary, warnings=warnings, preview=preview_of(doc))
