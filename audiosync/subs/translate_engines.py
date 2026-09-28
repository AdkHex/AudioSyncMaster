"""HTTP clients for the translation services, stdlib only.

The shipped engine is a PyInstaller build with numpy and nothing else, so
these talk to the services with ``urllib``. None of them uses a vendor SDK.
What they share lives here: retries that honour ``Retry-After``, errors
that name the service and never carry a key, and cancellation between
attempts. The pipeline itself (scenes, prompts, review, cache) is in
``translate.py``.

Keys: every client gets its key from ``ctx.secrets`` and registers it with
:func:`scrub`. Every message that leaves this module (TaskError, log line,
exception text) goes through ``scrub``, so a service that echoes the key
back in an error body, or Google's ``?key=`` URL in a urllib error, cannot
leak it into the job log or report.
"""

from __future__ import annotations

import json
import random
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .tasks import TaskError

#: Service endpoints. Tests point these at a local mock server; production
#: code never changes them. ``openai``/``ollama`` are only defaults: the
#: job's ``baseUrl`` overrides them.
ENDPOINTS: Dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
    "deepl_free": "https://api-free.deepl.com",
    "deepl_pro": "https://api.deepl.com",
    "google": "https://translation.googleapis.com",
}

ANTHROPIC_VERSION = "2023-06-01"
#: Server-side refusal fallback ("default" routes by refusal category).
FALLBACK_BETA = "server-side-fallback-2026-07-01"

#: Retries for 429 / 529 / 5xx / timeouts. The first wait is RETRY_BASE_S,
#: doubling up to RETRY_CAP_S. A Retry-After header wins when present.
MAX_ATTEMPTS = 6
RETRY_BASE_S = 2.0
RETRY_CAP_S = 60.0
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 529}

SERVICE_NAMES = {
    "claude": "Anthropic",
    "openai": "OpenAI",
    "ollama": "Ollama",
    "deepl": "DeepL",
    "google": "Google Translate",
}


# --------------------------------------------------------------- scrubbing


def scrub(text: Any, secrets: Any) -> str:
    """``text`` with every secret value (and any ``key=`` query value) masked."""
    out = str(text)
    values = secrets.values() if isinstance(secrets, dict) else (secrets or [])
    for value in values:
        if value and len(str(value)) >= 4:
            out = out.replace(str(value), "***")
            out = out.replace(urllib.parse.quote(str(value), safe=""), "***")
    return re.sub(r"([?&]key=)[^&\s\"']+", r"\1***", out)


# -------------------------------------------------------------- transport


class ApiError(Exception):
    """A non-retryable 4xx the caller may want to handle (e.g. degrade a
    request feature). ``message`` is already scrubbed."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


@dataclass
class Transport:
    """One service's HTTP settings plus the job's cancellation and log."""

    service: str
    secrets: Dict[str, str] = field(default_factory=dict)
    check: Callable[[], None] = lambda: None
    log: Callable[[str], None] = lambda _msg: None
    timeout: float = 300.0
    #: Whether this service authenticates with a key (for the 401 message).
    keyed: bool = True

    def _sleep(self, seconds: float) -> None:
        """Sleep in short slices so Cancel is honoured during a backoff."""
        end = time.monotonic() + max(0.0, seconds)
        while True:
            self.check()
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(0.2, left))

    def _error_message(self, raw: bytes) -> str:
        text = raw.decode("utf-8", "replace")
        try:
            data = json.loads(text)
        except ValueError:
            return scrub(text.strip()[:300], self.secrets)
        message = ""
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                message = str(err.get("message") or err.get("type") or "")
            elif isinstance(err, str):
                message = err
            message = message or str(data.get("message") or data.get("detail") or "")
        return scrub((message or text)[:300], self.secrets)

    def request(
        self,
        method: str,
        url: str,
        body: Optional[dict] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> dict:
        """Send JSON, return parsed JSON, retrying what is worth retrying."""
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        all_headers = {"Accept": "application/json"}
        if payload is not None:
            all_headers["Content-Type"] = "application/json"
        all_headers.update(headers or {})
        last = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.check()
            req = urllib.request.Request(url, data=payload, headers=all_headers, method=method)
            retry_after: Optional[float] = None
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                try:
                    return json.loads(raw.decode("utf-8"))
                except ValueError:
                    last = f"{self.service} sent a response that is not JSON"
            except urllib.error.HTTPError as exc:
                status = exc.code
                raw = exc.read() if exc.fp else b""
                message = self._error_message(raw)
                if status in (401, 403) or (status == 400 and "api key" in message.lower() and self.keyed):
                    if self.keyed:
                        raise TaskError(
                            f"The {self.service} API key was rejected (HTTP {status}): {message}. "
                            "Check the key in Settings."
                        ) from None
                    raise TaskError(f"{self.service} refused the request (HTTP {status}): {message}") from None
                if status == 456:
                    raise TaskError(f"The {self.service} quota for this billing period is used up.") from None
                if status not in RETRYABLE:
                    raise ApiError(status, message) from None
                last = f"{self.service} is busy (HTTP {status}): {message}"
                retry_after = _retry_after(exc.headers.get("Retry-After") if exc.headers else None)
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
                reason = getattr(exc, "reason", exc)
                last = f"Could not reach {self.service}: {scrub(reason, self.secrets)}"
            if attempt == MAX_ATTEMPTS:
                break
            wait = retry_after if retry_after is not None else min(RETRY_CAP_S, RETRY_BASE_S * 2 ** (attempt - 1))
            wait = min(wait, RETRY_CAP_S * 2)
            if retry_after is None:
                wait *= 0.75 + random.random() * 0.5
            self.log(f"{last}; retrying in {wait:.0f} s ({attempt}/{MAX_ATTEMPTS - 1})")
            self._sleep(wait)
        raise TaskError(scrub(last or f"{self.service} did not answer", self.secrets))


def _retry_after(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # an HTTP date; fall back to exponential backoff


# ------------------------------------------------------------ LLM clients


class Refused(Exception):
    """The model declined (stop_reason "refusal") even after fallback."""


class Truncated(Exception):
    """The answer hit max_tokens; the caller splits the work and retries."""


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    characters: int = 0


def parse_json_text(text: str) -> Any:
    """JSON from a model's text answer, tolerating code fences and prose."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except ValueError:
                pass
    raise ValueError("the model's answer was not valid JSON")


class LLMClient:
    """Common face of the chat engines: one system + one user message in,
    one JSON object matching ``schema`` out."""

    engine = ""
    model = ""
    usage: Usage

    def complete(self, system: str, user: str, schema: dict, name: str, effort: str = "high") -> dict:
        raise NotImplementedError

    @property
    def cache_identity(self) -> str:
        """What besides the prompt decides the answer (never the key)."""
        return f"{self.engine}|{self.model}"


def _claude_version(model: str) -> Tuple[str, float]:
    match = re.match(r"claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d{1,2}))?(?:$|-)", model)
    if not match:
        return "", 0.0
    minor = match.group(3) or "0"
    return match.group(1), float(f"{match.group(2)}.{minor}")


class ClaudeClient(LLMClient):
    """Anthropic Messages API, raw HTTP.

    Structured output uses ``output_config.format`` (json_schema), not
    forced tool use: Claude Opus 5.5 and Fable 5.1 reject ``tool_choice``
    any/tool with a 400. Thinking is left to the model (adaptive, always on
    for Opus 5.5), so ``max_tokens`` leaves room for it; depth is set with
    ``output_config.effort``. Features an older or unusual model rejects are
    dropped after the first 400 that names them, so any Claude id works.
    """

    engine = "claude"

    def __init__(self, key: str, model: str, transport: Transport, base_url: Optional[str] = None):
        self.key = key
        self.model = model
        self.t = transport
        self.base = (base_url or ENDPOINTS["anthropic"]).rstrip("/")
        self.usage = Usage()
        family, version = _claude_version(model)
        #: effort: Opus 4.5+, Sonnet/Fable/Mythos 4.6+; Haiku 4.5 rejects it.
        self.use_effort = (family == "opus" and version >= 4.5) or (
            family in ("sonnet", "fable", "mythos") and version >= 4.6
        )
        self.use_schema = True
        #: Server-side refusal fallback where Anthropic offers it.
        self.use_fallback = (family == "opus" and version >= 5) or (family in ("fable", "mythos") and version >= 5)

    def _body(self, system: str, user: str, schema: dict, effort: str) -> Tuple[dict, Dict[str, str]]:
        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            # The system prompt is identical for every scene of a job, so it
            # is the cacheable prefix.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
        }
        output_config: Dict[str, Any] = {}
        if self.use_schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if self.use_effort and effort:
            output_config["effort"] = effort
        if output_config:
            body["output_config"] = output_config
        headers = {"x-api-key": self.key, "anthropic-version": ANTHROPIC_VERSION}
        if self.use_fallback:
            body["fallbacks"] = "default"
            headers["anthropic-beta"] = FALLBACK_BETA
        if not self.use_schema:
            body["messages"][0]["content"] = user + "\n\nAnswer with only the JSON object, no prose."
        return body, headers

    def complete(self, system: str, user: str, schema: dict, name: str, effort: str = "high") -> dict:
        for _ in range(4):
            body, headers = self._body(system, user, schema, effort)
            try:
                data = self.t.request("POST", f"{self.base}/v1/messages", body, headers)
            except ApiError as exc:
                if not self._degrade(exc):
                    raise _api_task_error(self.t.service, exc, self.model) from None
                continue
            break
        else:
            raise TaskError(f"{self.t.service} kept rejecting the request")
        self.usage.requests += 1
        usage = data.get("usage") or {}
        self.usage.input_tokens += int(usage.get("input_tokens") or 0) + int(
            usage.get("cache_creation_input_tokens") or 0
        ) + int(usage.get("cache_read_input_tokens") or 0)
        self.usage.output_tokens += int(usage.get("output_tokens") or 0)
        stop = data.get("stop_reason")
        if stop == "refusal":
            raise Refused(str((data.get("stop_details") or {}).get("category") or "refused"))
        if stop == "max_tokens":
            raise Truncated()
        text = "".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text")
        return parse_json_text(text)

    def _degrade(self, exc: ApiError) -> bool:
        """Drop a request feature this model rejected; False if nothing to drop."""
        if exc.status != 400:
            return False
        message = exc.message.lower()
        if self.use_fallback and "fallback" in message:
            self.use_fallback = False
            return True
        if self.use_effort and "effort" in message:
            self.use_effort = False
            return True
        if self.use_schema and ("output_config" in message or "json_schema" in message or "structured output" in message):
            self.use_schema = False
            return True
        return False


class OpenAICompatibleClient(LLMClient):
    """Chat Completions: OpenAI, Ollama, LM Studio, vLLM, OpenRouter...

    Asks for ``response_format`` json_schema (strict); a server that does
    not know it gets json_object, and then a plain prompt.
    """

    def __init__(self, engine: str, key: Optional[str], model: str, base_url: str, transport: Transport):
        self.engine = engine
        self.key = key
        self.model = model
        self.base = base_url.rstrip("/")
        self.t = transport
        self.usage = Usage()
        self.format_mode = "json_schema"

    @property
    def cache_identity(self) -> str:
        return f"{self.engine}|{self.model}|{self.base}"

    def complete(self, system: str, user: str, schema: dict, name: str, effort: str = "high") -> dict:
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        while True:
            body: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user if self.format_mode == "json_schema" else
                     user + "\n\nAnswer with only a JSON object matching this schema:\n" + json.dumps(schema)},
                ],
            }
            if self.format_mode == "json_schema":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": name, "strict": True, "schema": schema},
                }
            elif self.format_mode == "json_object":
                body["response_format"] = {"type": "json_object"}
            if self.engine == "ollama":
                body["temperature"] = 0.2
            try:
                data = self.t.request("POST", f"{self.base}/chat/completions", body, headers)
                break
            except ApiError as exc:
                if exc.status in (400, 422) and self.format_mode != "none":
                    self.format_mode = "json_object" if self.format_mode == "json_schema" else "none"
                    continue
                raise _api_task_error(self.t.service, exc, self.model) from None
        self.usage.requests += 1
        usage = data.get("usage") or {}
        self.usage.input_tokens += int(usage.get("prompt_tokens") or 0)
        self.usage.output_tokens += int(usage.get("completion_tokens") or 0)
        choices = data.get("choices") or [{}]
        choice = choices[0]
        message = choice.get("message") or {}
        if message.get("refusal"):
            raise Refused(str(message["refusal"])[:200])
        if choice.get("finish_reason") == "length":
            raise Truncated()
        return parse_json_text(message.get("content") or "")


def _api_task_error(service: str, exc: ApiError, model: str = "") -> TaskError:
    if exc.status == 404 and model:
        return TaskError(f"{service} does not know the model '{model}' ({exc.message})")
    return TaskError(f"{service} rejected the request (HTTP {exc.status}): {exc.message}")


# ------------------------------------------------------------- MT engines


_ALLOWED_TAG = re.compile(r"&lt;(/?)(i|b|u)&gt;", re.I)


def _to_markup(text: str, br: str) -> str:
    """Plain subtitle text -> XML/HTML the services keep tags in."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    escaped = _ALLOWED_TAG.sub(lambda m: f"<{m.group(1)}{m.group(2).lower()}>", escaped)
    return escaped.replace("\n", br)


def _from_markup(text: str) -> str:
    import html

    out = re.sub(r"\s*<br\s*/?>\s*", "\n", text, flags=re.I)
    out = re.sub(r"<(i|b|u)>\s+", r"<\1>", out)
    out = re.sub(r"\s+</(i|b|u)>", r"</\1>", out)
    return html.unescape(out).strip()


def deepl_target(code: str) -> str:
    raw = code.replace("_", "-")
    lowered = raw.lower()
    if lowered in ("en", "en-us"):
        return "EN-US"
    if lowered == "en-gb":
        return "EN-GB"
    if lowered in ("pt", "pt-br"):
        return "PT-BR"
    if lowered == "pt-pt":
        return "PT-PT"
    if lowered in ("zh", "zh-cn", "zh-hans"):
        return "ZH-HANS"
    if lowered in ("zh-tw", "zh-hk", "zh-hant"):
        return "ZH-HANT"
    return lowered.split("-")[0].upper()


class DeepLClient:
    """DeepL v2 /translate: one request per scene of up to 50 cues.

    ``context`` carries the neighbouring source lines (not translated, not
    billed); ``tag_handling=xml`` keeps ``<i>`` and the ``<br/>`` that
    stands in for a line break, and ``br`` is non-splitting so a sentence
    wrapped over two lines is translated as one sentence.
    """

    engine = "deepl"
    batch_limit = 50

    def __init__(self, key: str, transport: Transport):
        self.key = key
        self.t = transport
        self.base = (ENDPOINTS["deepl_free"] if key.strip().endswith(":fx") else ENDPOINTS["deepl_pro"]).rstrip("/")
        self.usage = Usage()
        self.model = "deepl"
        #: Source language the service detected (first answer), lower case.
        self.detected: Optional[str] = None

    def translate(self, texts: List[str], source: Optional[str], target: str, context: str, formality: str) -> List[str]:
        body: Dict[str, Any] = {
            "text": [_to_markup(t, "<br/>") for t in texts],
            "target_lang": deepl_target(target),
            "tag_handling": "xml",
            "non_splitting_tags": ["br", "i", "b", "u"],
            "preserve_formatting": True,
            "model_type": "prefer_quality_optimized",
        }
        if source:
            body["source_lang"] = source.split("-")[0].upper()
        if context:
            body["context"] = context
        if formality in ("more", "less"):
            body["formality"] = f"prefer_{formality}"
        try:
            data = self.t.request(
                "POST", f"{self.base}/v2/translate", body, {"Authorization": f"DeepL-Auth-Key {self.key}"}
            )
        except ApiError as exc:
            raise _api_task_error(self.t.service, exc) from None
        self.usage.requests += 1
        self.usage.characters += sum(len(t) for t in texts)
        items = data.get("translations") or []
        if items and not self.detected and items[0].get("detected_source_language"):
            self.detected = str(items[0]["detected_source_language"]).lower()
        out = [_from_markup(item.get("text", "")) for item in items]
        if len(out) != len(texts):
            raise TaskError("DeepL returned a different number of lines than it was sent.")
        return out


class GoogleClient:
    """Cloud Translation v2 (basic) with an API key, HTML mode so line
    breaks and italics survive."""

    engine = "google"
    batch_limit = 100

    def __init__(self, key: str, transport: Transport):
        self.key = key
        self.t = transport
        self.usage = Usage()
        self.model = "nmt"
        self.detected: Optional[str] = None

    def translate(self, texts: List[str], source: Optional[str], target: str, context: str, formality: str) -> List[str]:
        body: Dict[str, Any] = {"q": [_to_markup(t, "<br>") for t in texts], "target": target, "format": "html"}
        if source:
            body["source"] = source
        url = f"{ENDPOINTS['google'].rstrip('/')}/language/translate/v2?key={urllib.parse.quote(self.key, safe='')}"
        try:
            data = self.t.request("POST", url, body)
        except ApiError as exc:
            raise _api_task_error(self.t.service, exc) from None
        self.usage.requests += 1
        self.usage.characters += sum(len(t) for t in texts)
        items = (data.get("data") or {}).get("translations") or []
        if items and not self.detected and items[0].get("detectedSourceLanguage"):
            self.detected = str(items[0]["detectedSourceLanguage"]).lower()
        out = [_from_markup(item.get("translatedText", "")) for item in items]
        if len(out) != len(texts):
            raise TaskError("Google Translate returned a different number of lines than it was sent.")
        return out


def ollama_root(base_url: Optional[str] = None) -> str:
    """``http://host:port`` of an Ollama base URL (which ends in /v1)."""
    parsed = urllib.parse.urlparse(base_url or ENDPOINTS["ollama"])
    return f"{parsed.scheme}://{parsed.netloc}"


def ollama_models(base_url: Optional[str] = None, timeout: float = 0.3) -> Optional[List[str]]:
    """Installed Ollama models, or None when Ollama does not answer."""
    try:
        with urllib.request.urlopen(f"{ollama_root(base_url)}/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return [m.get("name") or m.get("model") for m in data.get("models") or [] if m.get("name") or m.get("model")]
