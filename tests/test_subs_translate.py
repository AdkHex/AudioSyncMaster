"""Tests for subtitle translation (audiosync/subs/translate.py).

No network: a local HTTP server in a thread stands in for Anthropic, OpenAI,
Ollama, DeepL and Google. The engines are pointed at it through
``translate_engines.ENDPOINTS`` (and ``baseUrl``), so these tests check
the exact requests we send (paths, headers, bodies) and how the pipeline
reacts to what comes back: missing ids, review corrections, over-budget
lines, 401 / 429 / 529, truncation, and a re-run served from the cache.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.media import CancellationToken  # noqa: E402
from audiosync.subs import translate, translate_engines  # noqa: E402
from audiosync.subs.model import Cue, SubtitleDoc  # noqa: E402
from audiosync.subs.tasks import TaskContext, TaskError  # noqa: E402

ANTHROPIC_KEY = "sk-ant-TESTKEY-0123456789"
OPENAI_KEY = "sk-openai-TESTKEY-abcdef"
DEEPL_FREE_KEY = "deepl-TESTKEY-1234:fx"
DEEPL_PRO_KEY = "deepl-TESTKEY-pro-5678"
GOOGLE_KEY = "AIza-TESTKEY-google-999"
ALL_KEYS = [ANTHROPIC_KEY, OPENAI_KEY, DEEPL_FREE_KEY, DEEPL_PRO_KEY, GOOGLE_KEY]


# ------------------------------------------------------------ mock server


class Mock:
    """Records every request; answers with scripted handlers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = []
        #: Optional per-test overrides: fn(payload, request) -> answer dict,
        #: or a (status, body, headers) tuple to send an HTTP error.
        self.translate_fn = None
        self.review_fn = None
        self.condense_fn = None
        self.detect_language = "fr"
        #: Responses to send before the normal ones: list of (status, body, headers).
        self.queue = []
        self.ollama_models = ["qwen3:14b"]
        server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        server.daemon_threads = True
        self.server = server
        self.base = f"http://127.0.0.1:{server.server_address[1]}"
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def reset(self):
        with self.lock:
            self.requests = []
        self.translate_fn = self.review_fn = self.condense_fn = None
        self.detect_language = "fr"
        self.queue = []

    def llm_requests(self):
        return [r for r in self.requests if r["path"].endswith(("/v1/messages", "/chat/completions"))]

    # -- the model's behaviour

    def llm_answer(self, system, user, request):
        payload = json.loads(user.split("\n\nAnswer with only")[0])
        if "reviewing editor" in system:
            fn = self.review_fn or (lambda p, r: {"corrections": []})
        elif system.startswith("You shorten"):
            fn = self.condense_fn or (lambda p, r: {"translations": [{"id": c["id"], "text": c["text"][:5]} for c in p["cues"]]})
        elif "identify the language" in system:
            fn = lambda p, r: {"language": self.detect_language}  # noqa: E731
        else:
            fn = self.translate_fn or default_translate
        return fn(payload, request)

    def _handler(self):
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body, headers=None):
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._record(None)
                if self.path == "/api/tags":
                    return self._send(200, {"models": [{"name": m} for m in mock.ollama_models]})
                self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else None
                request = self._record(body)
                with mock.lock:
                    queued = mock.queue.pop(0) if mock.queue else None
                if queued:
                    return self._send(*queued)
                path = urllib.parse.urlparse(self.path).path
                if path.endswith("/v1/messages"):
                    system = body["system"][0]["text"]
                    answer = mock.llm_answer(system, body["messages"][0]["content"], request)
                    if isinstance(answer, tuple):
                        return self._send(*answer)
                    stop = answer.pop("_stop", "end_turn")
                    return self._send(200, {
                        "id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
                        "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": json.dumps(answer, ensure_ascii=False)}],
                        "stop_reason": stop, "usage": {"input_tokens": 100, "output_tokens": 20,
                                                       "cache_read_input_tokens": 50, "cache_creation_input_tokens": 0},
                    })
                if path.endswith("/chat/completions"):
                    system = body["messages"][0]["content"]
                    answer = mock.llm_answer(system, body["messages"][1]["content"], request)
                    if isinstance(answer, tuple):
                        return self._send(*answer)
                    return self._send(200, {
                        "choices": [{"message": {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 80, "completion_tokens": 10},
                    })
                if path.endswith("/v2/translate"):
                    return self._send(200, {"translations": [
                        {"detected_source_language": "JA", "text": "T:" + t} for t in body["text"]]})
                if path.endswith("/language/translate/v2"):
                    return self._send(200, {"data": {"translations": [
                        {"translatedText": "G:" + q, "detectedSourceLanguage": "ja"} for q in body["q"]]}})
                self._send(404, {"error": {"message": "no route " + path}})

            def _record(self, body):
                entry = {"method": self.command, "path": self.path,
                         "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body}
                with mock.lock:
                    mock.requests.append(entry)
                return entry

        return Handler


def default_translate(payload, request):
    return {"translations": [{"id": c["id"], "text": "EN:" + strip(c["text"])} for c in payload["cues"]]}


def strip(text):
    return text.replace("<i>", "").replace("</i>", "").replace("\n", " ")


MOCK = Mock()


class Env:
    """Points every engine at the mock and gives the job a fresh cache."""

    def __init__(self, secrets=None):
        self.secrets = secrets if secrets is not None else {
            "anthropic": ANTHROPIC_KEY, "openai": OPENAI_KEY, "deepl": DEEPL_FREE_KEY, "google": GOOGLE_KEY}

    def __enter__(self):
        MOCK.reset()
        self.saved = dict(translate_engines.ENDPOINTS)
        self.saved_retry = (translate_engines.RETRY_BASE_S, translate_engines.MAX_ATTEMPTS)
        translate_engines.ENDPOINTS.update({
            "anthropic": MOCK.base,
            "openai": MOCK.base + "/v1",
            "ollama": MOCK.base + "/ollama/v1",
            "deepl_free": MOCK.base + "/free",
            "deepl_pro": MOCK.base + "/pro",
            "google": MOCK.base + "/google",
        })
        translate_engines.RETRY_BASE_S = 0.01
        self.root = tempfile.mkdtemp(prefix="subsync-translate-")
        self.saved_env = os.environ.get("AUDIOSYNC_CACHE_DIR")
        os.environ["AUDIOSYNC_CACHE_DIR"] = os.path.join(self.root, "cache")
        self.logs = []
        self.progress = []
        self.ctx = TaskContext(token=CancellationToken(), progress=lambda p, s: self.progress.append((p, s)),
                               log=self.logs.append, workdir=self.root, secrets=dict(self.secrets))
        return self

    def __exit__(self, *exc):
        translate_engines.ENDPOINTS.clear()
        translate_engines.ENDPOINTS.update(self.saved)
        translate_engines.RETRY_BASE_S, translate_engines.MAX_ATTEMPTS = self.saved_retry
        if self.saved_env is None:
            os.environ.pop("AUDIOSYNC_CACHE_DIR", None)
        else:
            os.environ["AUDIOSYNC_CACHE_DIR"] = self.saved_env
        shutil.rmtree(self.root, ignore_errors=True)


def ja_doc(n=4, gap_after=None, start=1.0):
    """``n`` Japanese cues of 2.5 s; a 10 s gap after cue index ``gap_after``."""
    lines = ["おはよう、田中さん。", "今日は<i>学校</i>に行くの？", "うん、行くよ。", "じゃあ、一緒に行こう。",
             "待って！", "遅れるぞ。", "ごめん。", "行こう。", "早く！", "はい。"]
    cues, t = [], start
    for i in range(n):
        cues.append(Cue(t, t + 2.5, lines[i % len(lines)]))
        t += 3.0 + (10.0 if gap_after is not None and i == gap_after else 0.0)
    return SubtitleDoc(cues=cues, language=None, source_format="srt")


def opts(**changes):
    base = dict(translate.DEFAULTS)
    base.update({"engine": "claude", "model": "claude-opus-5-5", "target": "en", "quality": "fast",
                 "fitReadingSpeed": False})
    base.update(changes)
    return base


def payload_of(request):
    body = request["body"]
    if "system" in body:
        return json.loads(body["messages"][0]["content"])
    return json.loads(body["messages"][1]["content"].split("\n\nAnswer with only")[0])


# ------------------------------------------------------------------ tests


def test_claude_request_shape():
    with Env() as env:
        out, report = translate.translate_doc(ja_doc(3), opts(), env.ctx)
        reqs = MOCK.llm_requests()
        assert len(reqs) == 1, reqs
        req = reqs[0]
        assert req["path"] == "/v1/messages"
        h = req["headers"]
        assert h["x-api-key"] == ANTHROPIC_KEY
        assert h["anthropic-version"] == "2023-06-01"
        assert h["content-type"] == "application/json"
        assert h["anthropic-beta"] == "server-side-fallback-2026-07-01"
        body = req["body"]
        assert body["model"] == "claude-opus-5-5"
        assert body["max_tokens"] >= 8000
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert body["output_config"]["format"]["schema"]["required"] == ["translations"]
        assert body["output_config"]["effort"] == "high"
        assert body["fallbacks"] == "default"
        for banned in ("thinking", "temperature", "tool_choice", "tools"):
            assert banned not in body, banned
        assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert "Japanese" in body["system"][0]["text"] and "English" in body["system"][0]["text"]
        assert "Tanaka-san" in body["system"][0]["text"]  # honorifics: keep
        cues = payload_of(req)["cues"]
        assert [c["id"] for c in cues] == [1, 2, 3]
        assert cues[0]["seconds"] == 2.5 and cues[0]["max_chars"] > 0
        assert out.cues[0].text.startswith("EN:")
        assert out.language == "en"
        assert report["source"] == "ja" and report["requests"] == 1
        assert report["inputTokens"] == 150 and report["outputTokens"] == 20


def test_claude_older_model_omits_effort_and_fallback():
    with Env() as env:
        translate.translate_doc(ja_doc(2), opts(model="claude-haiku-4-5"), env.ctx)
        body = MOCK.llm_requests()[0]["body"]
        assert "fallbacks" not in body
        assert "effort" not in body["output_config"]
        assert "anthropic-beta" not in MOCK.llm_requests()[0]["headers"]


def test_claude_drops_rejected_feature_and_retries():
    with Env() as env:
        MOCK.queue = [(400, {"type": "error", "error": {"type": "invalid_request_error",
                                                        "message": "fallbacks: not permitted for this model"}}, {})]
        out, _ = translate.translate_doc(ja_doc(2), opts(), env.ctx)
        reqs = MOCK.llm_requests()
        assert len(reqs) == 2
        assert "fallbacks" in reqs[0]["body"] and "fallbacks" not in reqs[1]["body"]
        assert out.cues[0].text.startswith("EN:")


def test_openai_request_shape():
    with Env() as env:
        out, report = translate.translate_doc(ja_doc(3), opts(engine="openai", model="gpt-5"), env.ctx)
        req = MOCK.llm_requests()[0]
        assert req["path"] == "/v1/chat/completions"
        assert req["headers"]["authorization"] == f"Bearer {OPENAI_KEY}"
        body = req["body"]
        assert body["model"] == "gpt-5"
        assert body["messages"][0]["role"] == "system" and body["messages"][1]["role"] == "user"
        rf = body["response_format"]
        assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"]["properties"]["translations"]
        assert out.cues[2].text.startswith("EN:")
        assert report["inputTokens"] == 80 and report["model"] == "gpt-5"


def test_openai_compatible_falls_back_to_json_object():
    with Env() as env:
        MOCK.queue = [(400, {"error": {"message": "response_format json_schema is not supported"}}, {})]
        out, _ = translate.translate_doc(ja_doc(2), opts(engine="openai", model="local-model",
                                                          baseUrl=MOCK.base + "/lmstudio/v1"), env.ctx)
        reqs = MOCK.llm_requests()
        assert reqs[0]["path"] == "/lmstudio/v1/chat/completions"
        assert reqs[1]["body"]["response_format"] == {"type": "json_object"}
        assert out.cues[1].text.startswith("EN:")


def test_ollama_needs_no_key_and_picks_installed_model():
    with Env(secrets={}) as env:
        out, report = translate.translate_doc(ja_doc(2), opts(engine="ollama", model="claude-opus-5-5"), env.ctx)
        req = MOCK.llm_requests()[0]
        assert req["path"] == "/ollama/v1/chat/completions"
        assert "authorization" not in req["headers"]
        assert req["body"]["model"] == "qwen3:14b"
        assert report["model"] == "qwen3:14b"
        assert out.cues[0].text.startswith("EN:")


def test_deepl_request_shape_free_and_pro_hosts():
    with Env() as env:
        doc = SubtitleDoc(cues=[Cue(1, 3, "前の行"), Cue(4, 6, "<i>こんにちは</i>\n元気？"), Cue(7, 9, "次の行")])
        out, report = translate.translate_doc(doc, opts(engine="deepl", source="ja", formality="more"), env.ctx)
        req = [r for r in MOCK.requests if r["path"].endswith("/v2/translate")][0]
        assert req["path"] == "/free/v2/translate", req["path"]
        assert req["headers"]["authorization"] == f"DeepL-Auth-Key {DEEPL_FREE_KEY}"
        body = req["body"]
        assert body["target_lang"] == "EN-US" and body["source_lang"] == "JA"
        assert body["tag_handling"] == "xml" and "br" in body["non_splitting_tags"]
        assert body["formality"] == "prefer_more"
        assert "<i>こんにちは</i><br/>元気？" in body["text"]
        assert "context" not in body  # all three lines are in this batch; no neighbours
        assert report["characters"] > 0 and report["requests"] == 1
        assert out.cues[1].text.startswith("T:")
        assert "<br" not in out.cues[1].text
    with Env(secrets={"deepl": DEEPL_PRO_KEY}) as env:
        translate.translate_doc(ja_doc(2), opts(engine="deepl", source="ja", batchSize=1), env.ctx)
        reqs = [r for r in MOCK.requests if r["path"].endswith("/v2/translate")]
        assert all(r["path"] == "/pro/v2/translate" for r in reqs)
        assert len(reqs) == 2
        # batch 2's context carries its neighbour (the previous source line)
        assert "おはよう" in reqs[1]["body"]["context"]


def test_google_request_shape_and_key_in_query_only():
    with Env() as env:
        out, report = translate.translate_doc(ja_doc(2), opts(engine="google", target="en"), env.ctx)
        req = [r for r in MOCK.requests if "/language/translate/v2" in r["path"]][0]
        parsed = urllib.parse.urlparse(req["path"])
        assert parsed.path == "/google/language/translate/v2"
        assert urllib.parse.parse_qs(parsed.query)["key"] == [GOOGLE_KEY]
        assert req["body"]["format"] == "html" and req["body"]["target"] == "en"
        assert len(req["body"]["q"]) == 2
        assert out.cues[0].text.startswith("G:")
        assert report["source"] == "ja"
        assert GOOGLE_KEY not in json.dumps(report)


def test_google_bad_key_error_is_scrubbed():
    with Env() as env:
        MOCK.queue = [(400, {"error": {"code": 400, "message": f"API key not valid: {GOOGLE_KEY}. Please pass a valid API key."}}, {})]
        try:
            translate.translate_doc(ja_doc(2), opts(engine="google"), env.ctx)
        except TaskError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected TaskError")
        assert "Google Translate API key was rejected" in message, message
        assert GOOGLE_KEY not in message


def test_scene_chunking_and_context():
    with Env() as env:
        doc = ja_doc(7, gap_after=2)  # cues 1-3, 10 s pause, cues 4-7
        _, report = translate.translate_doc(doc, opts(batchSize=4), env.ctx)
        reqs = MOCK.llm_requests()
        assert report["scenes"] == 2 and len(reqs) == 2
        first, second = payload_of(reqs[0]), payload_of(reqs[1])
        assert [c["id"] for c in first["cues"]] == [1, 2, 3]
        assert [c["id"] for c in second["cues"]] == [4, 5, 6, 7]
        assert "previous_scene" not in first
        assert len(first["next_lines"]) == 4 and first["next_lines"][0] == doc.cues[3].text
        prev = second["previous_scene"]
        assert [p["source"] for p in prev] == [c.text for c in doc.cues[:3]]
        assert all(p["translation"].startswith("EN:") for p in prev)
        assert any("scene 2/2" in stage for _, stage in env.progress)
    # A long scene with no pauses is cut into batches no larger than asked.
    items = [translate.Item(i + 1, i, i * 2.0, i * 2.0 + 1.8, "x", 10) for i in range(10)]
    batches = translate.split_scenes(items, 4)
    assert all(len(b) <= 4 for b in batches) and sum(len(b) for b in batches) == 10
    assert [it.id for b in batches for it in b] == list(range(1, 11))


def test_missing_and_empty_ids_are_retried():
    calls = []

    def flaky(payload, request):
        calls.append(payload)
        cues = payload["cues"]
        if len(calls) == 1:  # drop the last id, blank the second, add a stray id
            answer = [{"id": c["id"], "text": "EN:" + c["text"]} for c in cues[:-1]]
            answer[1]["text"] = ""
            answer.append({"id": 99, "text": "stray"})
            answer.append({"id": cues[0]["id"], "text": "duplicate"})
            return {"translations": answer}
        return default_translate(payload, request)

    with Env() as env:
        MOCK.translate_fn = flaky
        out, _ = translate.translate_doc(ja_doc(4), opts(), env.ctx)
        assert len(calls) == 2
        assert [c["id"] for c in calls[1]["cues"]] == [2, 4]
        assert [a["id"] for a in calls[1]["already_translated"]] == [1, 3]
        assert all(c.text.startswith("EN:") for c in out.cues)
        assert "duplicate" not in out.cues[0].text


def test_truncated_answer_splits_the_scene():
    calls = []

    def truncating(payload, request):
        calls.append([c["id"] for c in payload["cues"]])
        if len(payload["cues"]) > 2:
            return {"translations": [], "_stop": "max_tokens"}
        return default_translate(payload, request)

    with Env() as env:
        MOCK.translate_fn = truncating
        out, report = translate.translate_doc(ja_doc(4), opts(), env.ctx)
        assert calls == [[1, 2, 3, 4], [1, 2], [3, 4]], calls
        assert all(c.text.startswith("EN:") for c in out.cues)
        assert report["untranslated"] == 0


def test_review_pass_applies_corrections():
    def review(payload, request):
        cues = payload["cues"]
        assert {"id", "source", "draft", "max_chars"} <= set(cues[0])
        return {"corrections": [
            {"id": 1, "text": "Morning, Tanaka-san.", "issue": "unnatural phrasing"},
            {"id": 2, "text": cues[1]["draft"], "issue": "no change"},
            {"id": 42, "text": "stray", "issue": "unknown id"},
        ]}

    with Env() as env:
        MOCK.review_fn = review
        out, report = translate.translate_doc(ja_doc(3), opts(quality="accurate"), env.ctx)
        systems = [r["body"]["system"][0]["text"] for r in MOCK.llm_requests()]
        assert len(systems) == 2 and "reviewing editor" in systems[1]
        assert out.cues[0].text == "Morning, Tanaka-san."
        assert report["reviewedChanges"] == 1
        assert report["changes"] == [{"id": 1, "issue": "unnatural phrasing"}]


def test_condense_only_over_budget_cues():
    long_text = "This is a very long translated line that nobody could read in half a second on screen"

    def translate_fn(payload, request):
        return {"translations": [{"id": c["id"], "text": long_text if c["id"] == 2 else "Hi."} for c in payload["cues"]]}

    seen = []

    def condense_fn(payload, request):
        seen.append([c["id"] for c in payload["cues"]])
        return {"translations": [{"id": c["id"], "text": "Too fast."} for c in payload["cues"]]}

    with Env() as env:
        MOCK.translate_fn = translate_fn
        MOCK.condense_fn = condense_fn
        doc = ja_doc(3)
        doc.cues[1].end = doc.cues[1].start + 0.5
        out, report = translate.translate_doc(doc, opts(fitReadingSpeed=True), env.ctx)
        assert seen == [[2]], seen
        assert out.cues[1].text == "Too fast."
        assert report["condensed"] == 1


def test_401_names_the_service_without_the_key():
    with Env() as env:
        MOCK.queue = [(401, {"type": "error", "error": {"type": "authentication_error",
                                                        "message": f"invalid x-api-key {ANTHROPIC_KEY}"}}, {})]
        try:
            translate.translate_doc(ja_doc(2), opts(), env.ctx)
        except TaskError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected TaskError")
        assert "The Anthropic API key was rejected" in message, message
        assert ANTHROPIC_KEY not in message
        assert len(MOCK.llm_requests()) == 1  # not retried


def test_429_and_529_are_retried():
    with Env() as env:
        MOCK.queue = [
            (429, {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}, {"Retry-After": "0"}),
            (529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}, {}),
        ]
        out, report = translate.translate_doc(ja_doc(2), opts(), env.ctx)
        assert len(MOCK.llm_requests()) == 3
        assert report["requests"] == 1
        assert sum("retrying" in line for line in env.logs) == 2, env.logs
        assert out.cues[0].text.startswith("EN:")


def test_missing_key_is_a_clear_error():
    with Env(secrets={}) as env:
        try:
            translate.translate_doc(ja_doc(2), opts(), env.ctx)
        except TaskError as exc:
            assert "Anthropic API key" in str(exc)
        else:
            raise AssertionError("expected TaskError")
        assert MOCK.requests == []


def test_cache_hit_on_rerun():
    with Env() as env:
        doc = ja_doc(6, gap_after=2)
        first, report1 = translate.translate_doc(doc, opts(batchSize=3, quality="accurate"), env.ctx)
        sent = len(MOCK.llm_requests())
        assert sent == 4 and report1["cached"] == 0
        second, report2 = translate.translate_doc(doc, opts(batchSize=3, quality="accurate"), env.ctx)
        assert len(MOCK.llm_requests()) == sent, "re-run must not call the API"
        assert report2["cached"] == 4 and report2["requests"] == 0
        assert [c.text for c in first.cues] == [c.text for c in second.cues]
        # A different model is a different answer: no cache hit.
        translate.translate_doc(doc, opts(batchSize=3, model="claude-sonnet-5"), env.ctx)
        assert len(MOCK.llm_requests()) == sent + 2


def test_keys_never_in_logs_reports_or_cache():
    with Env() as env:
        MOCK.queue = [(529, {"type": "error", "error": {"type": "overloaded_error",
                                                        "message": f"Overloaded for key {ANTHROPIC_KEY}"}}, {})]
        doc = ja_doc(3)
        doc.cues[1].end = doc.cues[1].start + 0.4
        MOCK.translate_fn = lambda p, r: {"translations": [{"id": c["id"], "text": "A fairly long line of text " * 3} for c in p["cues"]]}
        out, report = translate.translate_doc(doc, opts(quality="accurate", fitReadingSpeed=True), env.ctx)
        for engine in ("openai", "deepl", "google"):
            translate.translate_doc(doc, opts(engine=engine, model="gpt-5"), env.ctx)
        blobs = [json.dumps(report), "\n".join(env.logs), json.dumps([s for _, s in env.progress])]
        for root, _, files in os.walk(os.environ["AUDIOSYNC_CACHE_DIR"]):
            for name in files:
                with open(os.path.join(root, name), encoding="utf-8") as fh:
                    blobs.append(fh.read())
        assert len(blobs) > 5
        assert any("retrying" in line for line in env.logs)
        for blob in blobs:
            for key in ALL_KEYS:
                assert key not in blob, f"key leaked into: {blob[:200]}"


def test_sdh_dropped_when_keep_sdh_is_false():
    doc = SubtitleDoc(cues=[
        Cue(1, 3, "[door slams]"),
        Cue(4, 6, "JOHN: Hello there."),
        Cue(7, 9, "- (sighs) Fine.\n- Good."),
        Cue(10, 12, "（ドアの音）"),
        Cue(13, 15, "[laughs] - Wait.\n- [gasps]"),
    ], language="en")
    assert translate.strip_sdh("JOHN: Hello there.") == "Hello there."
    assert translate.strip_sdh("- (sighs) Fine.\n- Good.") == "- Fine.\n- Good."
    assert translate.strip_sdh("[laughs] - Wait.\n- [gasps]") == "Wait."
    with Env() as env:
        out, report = translate.translate_doc(doc, opts(target="ja", keepSdh=False), env.ctx)
        sent = json.dumps(payload_of(MOCK.llm_requests()[0]), ensure_ascii=False)
        assert "door slams" not in sent and "JOHN" not in sent and "sighs" not in sent
        assert report["sdhRemoved"] == 2 and len(out.cues) == 3
        system = MOCK.llm_requests()[0]["body"]["system"][0]["text"]
        assert "SDH" not in system
    with Env() as env:
        translate.translate_doc(doc, opts(target="ja", keepSdh=True), env.ctx)
        assert "SDH" in MOCK.llm_requests()[0]["body"]["system"][0]["text"]


def test_source_auto_detection():
    assert translate.detect_script(["おはよう、田中さん。"]) == "ja"
    assert translate.detect_script(["안녕하세요"]) == "ko"
    assert translate.detect_script(["Bonjour"]) is None
    with Env() as env:
        doc = SubtitleDoc(cues=[Cue(1, 3, "Bonjour, ça va ?"), Cue(4, 6, "Oui, merci.")])
        _, report = translate.translate_doc(doc, opts(), env.ctx)
        reqs = MOCK.llm_requests()
        assert "identify the language" in reqs[0]["body"]["system"][0]["text"]
        assert report["source"] == "fr"
        assert "French" in reqs[1]["body"]["system"][0]["text"]


def test_glossary_context_and_policies_reach_the_prompt():
    with Env() as env:
        translate.translate_doc(ja_doc(2), opts(glossary="源氏 = Genji\n# comment\n鬼 -> oni",
                                                context="A goblin and a bride.", honorifics="localize",
                                                formality="less"), env.ctx)
        system = MOCK.llm_requests()[0]["body"]["system"][0]["text"]
        assert "源氏 → Genji" in system and "鬼 → oni" in system
        assert "A goblin and a bride." in system
        assert "do not keep romanized suffixes" in system
        assert "lean casual" in system


def test_italics_and_layout():
    assert translate.fix_tags("<i>学校</i>", "School") == "<i>School</i>"
    assert translate.fix_tags("学校", "<i>School") == "School"
    rules = translate.Rules(42, 17, 2, "-")
    assert translate.layout("-Where?\n-Here.", rules, "en") == "-Where?\n-Here."
    long = "I told you already that I am not going to the school festival this year"
    laid = translate.layout(long, rules, "en")
    assert "\n" in laid and all(len(line) <= 42 for line in laid.split("\n")), laid


def test_bilingual_doc_stacks_source_above_translation():
    with Env() as env:
        out, _ = translate.translate_doc(ja_doc(2), opts(), env.ctx)
        both = translate.bilingual_doc(out)
        assert both.cues[0].text.split("\n")[0] == "おはよう、田中さん。"
        assert both.cues[0].text.split("\n")[-1].startswith("EN:")


def test_run_task_writes_target_language_file():
    try:
        from audiosync.subs import formats
    except ImportError:
        print("        (skipped: formats.py not available yet)")
        return
    from audiosync.subs import tasks

    saved = tasks.load_subtitle
    try:
        from audiosync.subs import tracks  # noqa: F401
    except ImportError:
        # load_subtitle needs tracks.py only for tracks inside videos.
        tasks.load_subtitle = lambda ref, ctx, fps=None: formats.read(ref["path"], encoding=ref.get("encoding"), fps=fps)
    try:
        with Env() as env:
            src = os.path.join(env.root, "movie.ja.srt")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write("1\n00:00:01,000 --> 00:00:03,500\nおはよう、田中さん。\n\n"
                         "2\n00:00:04,000 --> 00:00:06,500\n今日は学校に行くの？\n")
            job = {"id": "j1", "task": "translate", "input": {"subtitle": {"path": src}},
                   "options": opts(bilingual=True), "output": {"format": "srt"}}
            result = translate.run_task(job, env.ctx)
            names = [os.path.basename(o.path) for o in result.outputs]
            assert names == ["movie.en.srt", "movie.bilingual.en.srt"], names
            assert result.outputs[0].language == "en"
            assert result.report["target"] == "en" and "Japanese" in result.summary, result.summary
            assert result.preview and result.preview["count"] == 2
            with open(result.outputs[0].path, encoding="utf-8") as fh:
                assert "EN:" in fh.read()
            with open(result.outputs[1].path, encoding="utf-8") as fh:
                both = fh.read()
            assert "おはよう" in both and "EN:" in both
            json.dumps(result.to_dict())  # the bridge must be able to send it
    finally:
        tasks.load_subtitle = saved


def test_engine_statuses():
    with Env():
        statuses = {s["id"]: s for s in translate.engine_statuses({"anthropic": "k"})}
        assert set(statuses) == {"claude", "openai", "deepl", "google", "ollama"}
        assert statuses["claude"]["available"] is True
        assert statuses["deepl"]["available"] is False and statuses["deepl"]["reason"]
        assert statuses["ollama"]["available"] is True  # the mock answers /api/tags
        assert all(set(s) == {"id", "label", "available", "reason", "pack"} for s in statuses.values())
        translate_engines.ENDPOINTS["ollama"] = "http://127.0.0.1:9/v1"  # nothing listens on port 9
        statuses = {s["id"]: s for s in translate.engine_statuses({})}
        assert statuses["ollama"]["available"] is False
        assert statuses["claude"]["available"] is False


def test_cancel_stops_between_requests():
    from audiosync.media import Cancelled

    with Env() as env:
        calls = []

        def cancel_after_first(payload, request):
            calls.append(1)
            env.ctx.token.cancel()
            return default_translate(payload, request)

        MOCK.translate_fn = cancel_after_first
        try:
            translate.translate_doc(ja_doc(6, gap_after=2), opts(batchSize=3), env.ctx)
        except Cancelled:
            pass
        else:
            raise AssertionError("expected Cancelled")
        assert len(calls) == 1


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {test.__name__}\n        {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
