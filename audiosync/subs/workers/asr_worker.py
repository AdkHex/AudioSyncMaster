"""Speech recognition worker: Whisper through mlx-whisper or faster-whisper.

Runs inside the asr-mlx or asr-faster pack's Python (never the engine's), so
it imports only the standard library, numpy and the pack's own libraries.

Request (JSON on stdin)::

    {"engine": "mlx" | "faster", "modelPath": "<local snapshot folder>",
     "audio": "<16 kHz mono 16-bit WAV>", "language": "ja" | null,
     "task": "transcribe" | "translate", "beamSize": 5, "initialPrompt": "",
     "speech": [[start, end], ...] | null,      # seconds, already padded
     "compressionRatioThreshold": 2.4, "logprobThreshold": -1.0,
     "noSpeechThreshold": 0.6, "hallucinationSilenceThreshold": 2.0 | null}

Result: {"language", "languageProbability", "duration", "speechDuration",
"segments": [{"start", "end", "text", "avgLogprob", "noSpeechProb",
"compressionRatio", "temperature", "words": [{"start", "end", "text",
"probability"}]}], "engine", "decoding", "device", "loadS", "recogniseS"}.
All times are on the original audio's timeline.

Speech regions are cut out and concatenated (with a second of silence
between them, see SPACER_S) before recognition, then every timestamp is
mapped back -- what faster-whisper's own ``vad_filter`` does.
It is done here for both engines, rather than with mlx-whisper's
``clip_timestamps``, because mlx-whisper 0.4.3 never seeks to the start of
the second and later clips (its loop continues from where the previous clip
ended), so the silences between clips were transcribed anyway -- exactly the
stretches where Whisper invents "Thanks for watching".

The concatenation is not decoded as one stream but in chunks of nearby
regions (see ``_chunks``), each starting its own 30-second window.
"""

from __future__ import annotations

import bisect
import dataclasses
import os
import sys
import time
import types
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol as proto  # noqa: E402

# Models arrive as local snapshot folders; never let a library phone home.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SR = 16000
STAGE = "Recognising speech"
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


#: Silence put between two speech stretches in the concatenated audio.
#: Whisper's word timings run about a third of a second early, so without it
#: the first word after a join was aligned *before* the join and mapped onto
#: the end of the previous stretch -- seconds away from where it was said.
#: Anything aligned inside the spacer belongs to the stretch after it.
SPACER_S = 1.0


class TimeMap:
    """Maps times on the concatenated speech audio back to the original."""

    def __init__(self, regions: List[Tuple[int, int]], spacer: int = 0):
        self.orig_start = [s for s, _ in regions]
        self.orig_end = [e for _, e in regions]
        self.cat_start: List[int] = []
        total = 0
        for s, e in regions:
            self.cat_start.append(total)
            total += e - s + spacer
        self.lengths = [e - s for s, e in regions]

    def index(self, t: float) -> int:
        """The stretch ``t`` belongs to; the spacer after stretch i counts
        as the start of stretch i + 1."""
        sample = int(round(t * SR))
        i = max(0, bisect.bisect_right(self.cat_start, sample) - 1)
        if sample >= self.cat_start[i] + self.lengths[i] and i + 1 < len(self.cat_start):
            return i + 1
        return i

    def to_original(self, t: float, index: int) -> float:
        offset = int(round(t * SR)) - self.cat_start[index]
        offset = max(0, min(self.lengths[index], offset))
        return (self.orig_start[index] + offset) / SR


def _cut_speech(audio, speech: Optional[List[List[float]]]):
    import numpy as np

    if speech is None:
        return audio, None  # no detection: recognise everything
    regions: List[Tuple[int, int]] = []
    for start, end in speech:
        s, e = max(0, int(start * SR)), min(len(audio), int(end * SR))
        if e - s < SR // 20:
            continue
        if regions and s <= regions[-1][1]:
            regions[-1] = (regions[-1][0], max(regions[-1][1], e))
        else:
            regions.append((s, e))
    if not regions:
        return np.zeros(0, dtype=np.float32), TimeMap([(0, 0)])
    spacer = int(SPACER_S * SR)
    gap = np.zeros(spacer, dtype=np.float32)
    parts = []
    for k, (s, e) in enumerate(regions):
        if k:
            parts.append(gap)
        parts.append(audio[s:e])
    return np.concatenate(parts).astype(np.float32), TimeMap(regions, spacer)


#: Chunks of speech decoded separately: at most one Whisper window long, and
#: a pause of CHUNK_GAP_S or more starts a new one.
#: Decoding the whole concatenation as one stream let Whisper's 30-second
#: windows fall wherever the arithmetic put them. On the opening of Demon
#: Slayer (Mugen Train) the first window held 21 s of score and effects and
#: then the start of a row of whispered names: Whisper wrote nothing for it,
#: then 「ご視聴ありがとうございました」 over the 10th, and 10 of 12 names
#: were lost. The same names, decoded from a window starting where they start,
#: all came out. A long pause is a scene or shot change, so that is where a
#: window should start.
CHUNK_S = 30.0
CHUNK_GAP_S = 4.0


def _chunks(audio, speech: Optional[List[List[float]]]) -> List[Tuple[Any, Optional[TimeMap]]]:
    """(concatenated audio, TimeMap) for each chunk of speech regions; the
    whole audio with no map when there is no speech detection."""
    if speech is None:
        return [(audio, None)]
    groups: List[List[List[float]]] = []
    length = 0.0
    for start, end in sorted(speech):
        span = max(0.0, end - start)
        if groups and start - groups[-1][-1][1] < CHUNK_GAP_S and length + SPACER_S + span <= CHUNK_S:
            groups[-1].append([start, end])
            length += SPACER_S + span
        else:
            groups.append([[start, end]])
            length = span
    out = []
    for group in groups:
        work, tmap = _cut_speech(audio, group)
        if len(work) >= SR // 5:  # under 0.2 s nothing can be recognised
            out.append((work, tmap))
    return out


def _restore(segments: List[Dict[str, Any]], tmap: Optional[TimeMap]) -> None:
    """Move every segment and word back onto the original timeline.

    A word keeps to the stretch its midpoint falls in, so a word can never
    be stretched across the silence that was cut out between two stretches.
    """
    if tmap is None:
        return
    for seg in segments:
        for word in seg["words"]:
            i = tmap.index((word["start"] + word["end"]) / 2)
            word["start"] = round(tmap.to_original(word["start"], i), 3)
            word["end"] = round(tmap.to_original(word["end"], i), 3)
        if seg["words"]:
            seg["start"], seg["end"] = seg["words"][0]["start"], seg["words"][-1]["end"]
        else:
            i = tmap.index(seg["start"])
            j = tmap.index(max(seg["start"], seg["end"] - 0.01))
            seg["start"] = round(tmap.to_original(seg["start"], i), 3)
            seg["end"] = round(tmap.to_original(seg["end"], j), 3)


# --------------------------------------------------------------- mlx-whisper


class _MlxBar:
    """Stands in for tqdm inside mlx_whisper.transcribe to report progress.

    mlx-whisper has no progress callback; it advances a tqdm bar by the mel
    frames of each 30-second window it finishes, which is exactly the number
    the engine needs.
    """

    #: The share of the whole job the current chunk covers, as (start,
    #: length) in 0-1; set before each chunk is decoded.
    part: Tuple[float, float] = (0.0, 1.0)

    def __init__(self, total: Optional[int] = None, **_kwargs: Any) -> None:
        self.total = max(1, int(total or 1))
        self.n = 0

    def __enter__(self) -> "_MlxBar":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def update(self, n: int) -> None:
        self.n += n
        start, length = self.part
        proto.progress(min(99.0, 100.0 * (start + length * min(1.0, self.n / self.total))), STAGE)


def _mlx_detect(model, audio, mx) -> Tuple[str, float, Dict[str, float]]:
    """Language from up to three 30-second windows of speech.

    mlx-whisper looks at the first 30 seconds only and does not report how
    sure it is. With silence already cut out those seconds are speech, and
    summing over a few windows keeps a cold open in another language (a
    song, a news clip on a TV) from deciding the whole film.
    """
    from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

    totals: Dict[str, float] = {}
    windows = max(1, min(3, -(-len(audio) // N_SAMPLES)))
    for w in range(windows):
        piece = audio[w * N_SAMPLES:(w + 1) * N_SAMPLES]
        if w > 0 and len(piece) < SR * 5:
            break
        # Pad with silent *audio* before the mel, as Whisper itself does;
        # zero-padding the mel would read as something other than silence.
        mel = log_mel_spectrogram(piece, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = model.detect_language(mel)
        for lang, p in probs.items():
            totals[lang] = totals.get(lang, 0.0) + float(p)
        if w == 0 and max(probs.values()) > 0.9:
            break
    norm = sum(totals.values()) or 1.0
    probs = {k: v / norm for k, v in totals.items()}
    best = max(probs, key=probs.get)
    return best, probs[best], probs


def _parts(pieces: List[Tuple[Any, Optional[TimeMap]]]) -> List[Tuple[float, float]]:
    """Each chunk's (start, length) share of the job, by duration."""
    total = float(sum(len(a) for a, _ in pieces)) or 1.0
    out, done = [], 0
    for audio, _ in pieces:
        out.append((done / total, len(audio) / total))
        done += len(audio)
    return out


def run_mlx(pieces: List[Tuple[Any, Optional[TimeMap]]], detect_audio, req: Dict[str, Any]) -> Dict[str, Any]:
    import mlx.core as mx
    import mlx_whisper
    from mlx_whisper.decoding import DecodingOptions
    from mlx_whisper.tokenizer import get_tokenizer

    transcribe_module = sys.modules["mlx_whisper.transcribe"]
    transcribe_module.tqdm = types.SimpleNamespace(tqdm=_MlxBar)

    proto.progress(0, "Loading the model")
    t0 = time.monotonic()
    model = transcribe_module.ModelHolder.get_model(req["modelPath"], mx.float16)
    load_s = time.monotonic() - t0

    language = req.get("language")
    probability = None
    if not language:
        proto.progress(0, "Detecting the language")
        language, probability, _ = _mlx_detect(model, detect_audio, mx)
        proto.log(f"Detected language: {language} ({probability:.0%})")

    task = req.get("task") or "transcribe"
    prompt = (req.get("initialPrompt") or "").strip()
    if prompt:
        # Whisper applies initial_prompt to the first window only once
        # condition_on_previous_text is off; names and terms matter in every
        # window, so the prompt is handed to each decode that has none.
        tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                                  language=language, task=task)
        prompt_tokens = tokenizer.encode(" " + prompt)
        original = type(model).decode

        def decode_with_terms(self, mel, options=DecodingOptions(), **kwargs):
            if not options.prompt:
                options = dataclasses.replace(options, prompt=list(prompt_tokens))
            return original(self, mel, options, **kwargs)

        model.decode = types.MethodType(decode_with_terms, model)

    beam = int(req.get("beamSize") or 1)
    options: Dict[str, Any] = dict(
        path_or_hf_repo=req["modelPath"],
        verbose=False,
        temperature=TEMPERATURES,
        compression_ratio_threshold=req.get("compressionRatioThreshold", 2.4),
        logprob_threshold=req.get("logprobThreshold", -1.0),
        no_speech_threshold=req.get("noSpeechThreshold", 0.6),
        condition_on_previous_text=False,
        word_timestamps=True,
        hallucination_silence_threshold=req.get("hallucinationSilenceThreshold"),
        language=language,
        task=task,
        fp16=True,
    )
    # mlx-whisper 0.4.3 raises NotImplementedError for beam_size. Greedy
    # decoding at temperature 0 with best-of-N sampling on the fallback
    # temperatures is the closest it offers.
    if beam > 1:
        options["best_of"] = beam
    decoding = "greedy" + (f", best of {beam} when falling back" if beam > 1 else "") + " (mlx-whisper has no beam search)"

    proto.progress(0, STAGE)
    t1 = time.monotonic()
    segments = []
    for (audio, tmap), part in zip(pieces, _parts(pieces)):
        _MlxBar.part = part
        out = mlx_whisper.transcribe(audio, **options)
        found = []
        for seg in out.get("segments", []):
            text = str(seg.get("text") or "")
            if not text.strip():
                continue
            found.append({
                "start": float(seg["start"]), "end": float(seg["end"]), "text": text,
                "avgLogprob": float(seg.get("avg_logprob", 0.0)),
                "noSpeechProb": float(seg.get("no_speech_prob", 0.0)),
                "compressionRatio": float(seg.get("compression_ratio", 0.0)),
                "temperature": float(seg.get("temperature", 0.0)),
                "words": [
                    {"start": float(w["start"]), "end": float(w["end"]), "text": str(w["word"]),
                     "probability": round(float(w.get("probability", 0.0)), 4)}
                    for w in seg.get("words", [])
                ],
            })
        _restore(found, tmap)
        segments.extend(found)
    recognise_s = time.monotonic() - t1
    return {
        "language": language,
        "languageProbability": probability,
        "segments": segments,
        "engine": "mlx-whisper",
        "decoding": decoding,
        "device": "Apple GPU (MLX)",
        "loadS": round(load_s, 2),
        "recogniseS": round(recognise_s, 2),
    }


# ------------------------------------------------------------ faster-whisper


def run_faster(pieces: List[Tuple[Any, Optional[TimeMap]]], detect_audio, req: Dict[str, Any]) -> Dict[str, Any]:
    import ctranslate2
    from faster_whisper import WhisperModel

    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    try:
        return _run_faster(WhisperModel, pieces, detect_audio, req, device)
    except (RuntimeError, OSError) as exc:
        # A CUDA device without cuBLAS/cuDNN 9 fails on first use; the CPU
        # always works, only slower.
        if device != "cuda":
            raise
        proto.log(f"GPU recognition failed ({exc}); falling back to the CPU")
        return _run_faster(WhisperModel, pieces, detect_audio, req, "cpu")


def _run_faster(WhisperModel, pieces, detect_audio, req: Dict[str, Any], device: str) -> Dict[str, Any]:
    compute_type = "float16" if device == "cuda" else "int8"
    proto.progress(0, "Loading the model")
    t0 = time.monotonic()
    model = WhisperModel(req["modelPath"], device=device, compute_type=compute_type,
                         cpu_threads=os.cpu_count() or 4, local_files_only=True)
    load_s = time.monotonic() - t0

    beam = max(1, int(req.get("beamSize") or 1))
    prompt = (req.get("initialPrompt") or "").strip() or None
    language = req.get("language") or None
    probability = None
    if not language:
        # Once, over the first speech, rather than per chunk: a chunk can be
        # a single shout, far too little to tell a language by.
        proto.progress(0, "Detecting the language")
        language, probability, _ = model.detect_language(detect_audio, language_detection_segments=3)
        proto.log(f"Detected language: {language} ({probability:.0%})")
    proto.progress(0, STAGE)
    t1 = time.monotonic()
    segments = []
    for (audio, tmap), (start, length) in zip(pieces, _parts(pieces)):
        found = _faster_chunk(model, audio, req, language, beam, prompt, start, length)
        _restore(found, tmap)
        segments.extend(found)
    recognise_s = time.monotonic() - t1
    return {
        "language": language,
        "languageProbability": None if probability is None else float(probability),
        "segments": segments,
        "engine": "faster-whisper",
        "decoding": f"beam search, {beam} beams" if beam > 1 else "greedy",
        "device": f"{device} ({compute_type})",
        "loadS": round(load_s, 2),
        "recogniseS": round(recognise_s, 2),
    }


def _faster_chunk(model, audio, req: Dict[str, Any], language: str, beam: int, prompt: Optional[str],
                  start: float, length: float) -> List[Dict[str, Any]]:
    duration = max(len(audio) / SR, 1e-6)
    segments_iter, _info = model.transcribe(
        audio,
        language=language,
        task=req.get("task") or "transcribe",
        beam_size=beam,
        best_of=beam,
        temperature=list(TEMPERATURES),
        compression_ratio_threshold=req.get("compressionRatioThreshold", 2.4),
        log_prob_threshold=req.get("logprobThreshold", -1.0),
        no_speech_threshold=req.get("noSpeechThreshold", 0.6),
        condition_on_previous_text=False,
        initial_prompt=prompt,
        # hotwords are added to every window's prompt, unlike initial_prompt.
        hotwords=prompt,
        word_timestamps=True,
        vad_filter=False,
        hallucination_silence_threshold=req.get("hallucinationSilenceThreshold"),
    )
    segments = []
    for seg in segments_iter:
        proto.progress(min(99.0, 100.0 * (start + length * min(1.0, seg.end / duration))), STAGE)
        if not seg.text.strip():
            continue
        segments.append({
            "start": float(seg.start), "end": float(seg.end), "text": seg.text,
            "avgLogprob": float(seg.avg_logprob), "noSpeechProb": float(seg.no_speech_prob),
            "compressionRatio": float(seg.compression_ratio), "temperature": float(seg.temperature or 0.0),
            "words": [
                {"start": float(w.start), "end": float(w.end), "text": w.word,
                 "probability": round(float(w.probability), 4)}
                for w in (seg.words or [])
            ],
        })
    return segments


def handle(req: Dict[str, Any]) -> Dict[str, Any]:
    audio, rate = proto.read_wav(req["audio"])
    if rate != SR or getattr(audio, "ndim", 1) != 1:
        raise ValueError(f"expected 16 kHz mono audio, got {rate} Hz")
    work, tmap = _cut_speech(audio, req.get("speech"))
    speech_s = sum(tmap.lengths) / SR if tmap is not None else len(work) / SR
    if speech_s < 0.2:
        return {"language": req.get("language"), "languageProbability": None, "segments": [],
                "duration": len(audio) / SR, "speechDuration": speech_s,
                "engine": req.get("engine"), "decoding": "", "device": "", "loadS": 0, "recogniseS": 0}
    pieces = _chunks(audio, req.get("speech"))
    if tmap is not None:
        proto.log(f"Recognising {speech_s:.0f} s of speech out of {len(audio) / SR:.0f} s in {len(pieces)} chunks")
    # Language detection reads at most three windows of the speech.
    detect_audio = work[:3 * 30 * SR]
    run = run_mlx if req.get("engine") == "mlx" else run_faster
    result = run(pieces, detect_audio, req)
    result["duration"] = len(audio) / SR
    result["speechDuration"] = speech_s
    proto.progress(100, STAGE)
    return result


if __name__ == "__main__":
    proto.main(handle)
