"""Tests for subtitle tracks inside videos: listing, extraction, muxing,
HDR detection and file probing -- all on real files made by ffmpeg (and
mkvmerge where installed), since the failure modes live in the tools'
behaviour, not in our command strings."""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.media import CancellationToken, MediaError, ffmpeg_path  # noqa: E402
from audiosync.subs import formats, tracks  # noqa: E402
from audiosync.subs.tasks import TaskContext  # noqa: E402

SRT = "1\n00:00:00,500 --> 00:00:01,500\nHello\n\n2\n00:00:02,000 --> 00:00:03,000\n<i>World</i>\n"
ASS = (
    "[Script Info]\nScriptType: v4.00+\nPlayResX: 320\nPlayResY: 240\n\n[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
    "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
    "MarginR, MarginV, Encoding\n"
    "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1\n\n"
    "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    "Dialogue: 0,0:00:00.50,0:00:01.50,Default,,0,0,0,,{\\an8\\blur2}Bonjour\n"
)

SUBTITLE_TRACK_INFO_KEYS = {"index", "codec", "kind", "language", "title", "forced", "default",
                            "hearingImpaired", "events"}
VIDEO_INFO_KEYS = {"width", "height", "fps", "codec", "hdr", "dvProfile", "dvCompatibility", "maxCll",
                   "masteringPeak", "transfer", "primaries"}
PROBED_FILE_KEYS = {"path", "name", "kind", "duration", "video", "audioTracks", "subtitleTracks", "subtitle", "error"}


class Workspace:
    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="subsync-tracks-")
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.root, name)

    def write(self, name, data):
        path = self.path(name)
        with open(path, "wb") as handle:
            handle.write(data if isinstance(data, bytes) else data.encode("utf-8"))
        return path


def _ctx(workdir):
    return TaskContext(token=CancellationToken(), progress=lambda *_: None, log=lambda *_: None, workdir=workdir)


def _ffmpeg(*args):
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", *args], capture_output=True, check=True)


def _sup(path):
    """Two display sets of HDMV PGS, built by hand: ffmpeg cannot encode PGS."""
    def segment(kind, payload, pts):
        return b"PG" + struct.pack(">IIBH", int(pts * 90000), 0, kind, len(payload)) + payload

    def display_set(pts, show, number):
        pcs = struct.pack(">HHBHBBBB", 160, 120, 0x10, number, 0x80 if show else 0, 0, 0, 1 if show else 0)
        if show:
            pcs += struct.pack(">HBBHH", 0, 0, 0, 20, 90)
        out = segment(0x16, pcs, pts) + segment(0x17, struct.pack(">BBHHHH", 1, 0, 20, 90, 40, 10), pts)
        if show:
            out += segment(0x14, bytes([0, 0, 1, 235, 128, 128, 255]), pts)
            rle = (bytes([0x00, 0xC0, 40, 1]) + b"\x00\x00") * 10
            body = struct.pack(">HBB", 0, 0, 0xC0) + (len(rle) + 4).to_bytes(3, "big") + struct.pack(">HH", 40, 10) + rle
            out += segment(0x15, body, pts)
        return out + segment(0x80, b"", pts)

    with open(path, "wb") as handle:
        handle.write(display_set(0.5, True, 0) + display_set(1.5, False, 1)
                     + display_set(2.0, True, 2) + display_set(3.0, False, 3))
    return path


def _segments(path):
    """(type, pts, payload) of every PGS segment -- DTS is left out: ffmpeg
    writes DTS = PTS where the original had 0, which players ignore."""
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        pts, _dts, kind, size = struct.unpack(">IIBH", data[i + 2:i + 13])
        out.append((kind, pts, data[i + 13:i + 13 + size]))
        i += 13 + size
    return out


def _source_mkv(ws, pgs=True):
    """Video + audio + SRT (eng) + ASS (fre, forced, "French SDH") [+ PGS
    (jpn, default)] + a font attachment."""
    srt, ass = ws.write("a.srt", SRT), ws.write("b.ass", ASS)
    font = ws.write("font.ttf", b"not really a font")
    inputs = ["-i", srt, "-i", ass]
    maps = ["-map", "0", "-map", "1", "-map", "2", "-map", "3"]
    meta = ["-metadata:s:s:0", "language=eng", "-metadata:s:s:1", "language=fre",
            "-metadata:s:s:1", "title=French SDH", "-disposition:s:0", "0", "-disposition:s:1", "forced"]
    if pgs:
        inputs += ["-i", _sup(ws.path("c.sup"))]
        maps += ["-map", "4"]
        meta += ["-metadata:s:s:2", "language=jpn", "-disposition:s:2", "default"]
    out = ws.path("src.mkv")
    # -copyts: keep the PGS at its own times (ffmpeg would start it at zero).
    _ffmpeg("-copyts", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=24000/1001:duration=4",
            "-f", "lavfi", "-i", "sine=duration=4", *inputs, *maps,
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-c:s", "copy", *meta,
            "-attach", font, "-metadata:s:t", "mimetype=font/ttf", out)
    return out


# ------------------------------------------------------------------ listing


def test_list_tracks_on_mkv():
    with Workspace() as ws:
        found = tracks.list_tracks(_source_mkv(ws))
        assert [t.codec for t in found] == ["subrip", "ass", "hdmv_pgs_subtitle"]
        assert [t.kind for t in found] == ["text", "text", "image"]
        assert [t.language for t in found] == ["en", "fr", "ja"], "ISO 639-2/B normalised to 639-1"
        srt, ass, pgs = found
        assert ass.forced and ass.hearing_impaired and ass.title == "French SDH" and not ass.default
        assert pgs.default and not srt.forced and srt.stream_index == 2
        assert set(srt.to_dict()) == SUBTITLE_TRACK_INFO_KEYS
        assert srt.to_dict()["hearingImpaired"] is False


def test_event_counts_from_mkvmerge_statistics():
    if not shutil.which("mkvmerge"):
        print("  SKIP  (mkvmerge not installed)")
        return
    with Workspace() as ws:
        remuxed = ws.path("remux.mkv")
        subprocess.run(["mkvmerge", "-q", "-o", remuxed, _source_mkv(ws)], capture_output=True)
        assert [t.events for t in tracks.list_tracks(remuxed)] == [2, 1, 4]


# --------------------------------------------------------------- extraction


def test_extract_text_and_pgs_tracks():
    with Workspace() as ws:
        source = _source_mkv(ws)
        out_dir = ws.path("out")
        srt = tracks.extract(source, 0, out_dir)
        assert srt == os.path.join(out_dir, "src.track1.en.srt")
        assert [c.text for c in formats.read(srt).cues] == ["Hello", "<i>World</i>"]
        ass = tracks.extract(source, 1, out_dir)
        assert ass.endswith(".track2.fr.ass")
        assert "{\\an8\\blur2}Bonjour" in open(ass, encoding="utf-8").read(), "ASS extracted with its tags"
        sup = tracks.extract(source, 2, out_dir)
        assert sup.endswith(".track3.ja.sup") and formats.detect_format(sup) == "pgs"
        assert _segments(sup) == _segments(ws.path("c.sup")), "PGS payload changed on the way out"
        try:
            tracks.extract(source, 7, out_dir)
        except MediaError as exc:
            assert "no subtitle track 8" in str(exc)
        else:
            raise AssertionError("extracted a track that does not exist")


def test_extract_mov_text_becomes_srt():
    with Workspace() as ws:
        mp4 = ws.path("clip.mp4")
        _ffmpeg("-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=4", "-i", ws.write("a.srt", SRT),
                "-c:v", "libx264", "-preset", "ultrafast", "-c:s", "mov_text", "-metadata:s:s:0", "language=ger", mp4)
        found = tracks.list_tracks(mp4)
        assert found[0].codec == "mov_text" and found[0].language == "de"
        out = tracks.extract(mp4, 0, ws.path("out"))
        assert out.endswith(".de.srt") and [c.text for c in formats.read(out).cues] == ["Hello", "<i>World</i>"]


def test_extract_task_all_tracks():
    with Workspace() as ws:
        source = _source_mkv(ws)
        job = {"id": "x", "task": "extract", "input": {"video": {"path": source}},
               "options": {"tracks": "all"}, "output": {"dir": ws.path("dest")}}
        result = tracks.run_extract_task(job, _ctx(ws.path("work")))
        names = sorted(os.path.basename(o.path) for o in result.outputs)
        assert names == ["src.track1.en.srt", "src.track2.fr.ass", "src.track3.ja.sup"], names
        assert [o.format for o in result.outputs] == ["srt", "ass", "pgs"]
        job["options"]["tracks"] = [1]
        job["output"]["format"] = "vtt"
        result = tracks.run_extract_task(job, _ctx(ws.path("work2")))
        assert len(result.outputs) == 1 and result.outputs[0].path.endswith("src.track2.fr.vtt")
        assert formats.read(result.outputs[0].path).cues[0].align == 8


# ------------------------------------------------------------------ video


def test_video_info_sdr_and_hdr10():
    with Workspace() as ws:
        info = tracks.video_info(_source_mkv(ws, pgs=False))
        assert set(info) == VIDEO_INFO_KEYS
        assert info["hdr"] == "sdr" and info["codec"] == "h264" and (info["width"], info["height"]) == (160, 120)
        assert abs(info["fps"] - 24000 / 1001) < 1e-9

        hdr = ws.path("hdr.mkv")
        params = ("log-level=error:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
                  "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):"
                  "max-cll=1000,400:hdr10-opt=1:repeat-headers=1")
        made = subprocess.run(
            [ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-f", "lavfi",
             "-i", "testsrc2=size=256x144:rate=24:duration=0.5", "-pix_fmt", "yuv420p10le",
             "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", params, hdr],
            capture_output=True,
        )
        if made.returncode != 0:
            print("  SKIP  (no libx265 for the HDR10 clip)")
            return
        info = tracks.video_info(hdr)
        assert info["hdr"] == "hdr10", info
        assert info["transfer"] == "smpte2084" and info["primaries"] == "bt2020"
        assert info["maxCll"] == 1000 and info["masteringPeak"] == 1000.0
        assert info["dvProfile"] is None and info["codec"] == "hevc"


# ------------------------------------------------------------------ probing


def test_probe_file_video_subtitles_and_errors():
    with Workspace() as ws:
        video = tracks.probe_file(_source_mkv(ws))
        assert set(video) == PROBED_FILE_KEYS and video["error"] is None
        assert video["kind"] == "video" and video["video"]["hdr"] == "sdr" and video["duration"] > 3.5
        assert len(video["subtitleTracks"]) == 3 and video["audioTracks"][0]["channels"] == 1
        assert set(video["audioTracks"][0]) == {"index", "codec", "language", "title", "channels"}

        text = tracks.probe_file(ws.write("Film.ja.srt", ("1\n00:00:01,000 --> 00:00:02,000\n待っていた。\n")
                                          .encode("cp932")))
        assert text["kind"] == "subtitle" and text["error"] is None
        assert text["subtitle"] == {"format": "srt", "kind": "text", "cues": 1, "language": "ja", "encoding": "cp932"}

        sup = tracks.probe_file(ws.path("c.sup"))
        assert sup["subtitle"]["kind"] == "image" and sup["subtitle"]["cues"] == 2

        idx = ws.write("dvd.idx", "# VobSub index file, v7\nid: en, index: 0\ntimestamp: 00:00:01:000, filepos: 0\n"
                                  "timestamp: 00:00:02:000, filepos: 800\n")
        ws.write("dvd.sub", b"\x00\x00\x01\xba" + b"\x00" * 60)
        vob = tracks.probe_file(idx)
        assert vob["subtitle"]["format"] == "vobsub" and vob["subtitle"]["cues"] == 2
        assert tracks.probe_file(ws.path("dvd.sub"))["subtitle"]["format"] == "vobsub"

        broken = tracks.probe_file(ws.write("broken.mkv", b"not a video at all"))
        assert broken["error"] and broken["kind"] == "unknown"
        missing = tracks.probe_file(ws.path("nope.mkv"))
        assert "not found" in missing["error"].lower()


# ---------------------------------------------------------------------- mux


def _check_mux(ws, engine):
    source = _source_mkv(ws)
    out = ws.path(f"muxed-{engine}.mkv")
    new = [
        {"path": ws.write("Movie.de.srt", SRT), "title": "Deutsch", "forced": True, "hearingImpaired": True},
        {"path": ws.write("jp.srt", ("1\n00:00:01,000 --> 00:00:02,000\n待っていた。\n").encode("cp932")),
         "language": "ja", "default": True},
    ]
    logged = []
    tracks.mux(source, new, out, engine=engine, log=logged.append)
    found = tracks.list_tracks(out)
    assert [t.codec for t in found] == ["subrip", "ass", "hdmv_pgs_subtitle", "subrip", "subrip"], found
    german, japanese = found[3], found[4]
    assert german.language == "de" and german.title == "Deutsch", "language from the file name"
    assert german.forced and german.hearing_impaired and not german.default
    assert japanese.language == "ja" and japanese.default
    assert not found[2].default, "the old default gave way to the new one"
    assert found[1].forced and found[1].title == "French SDH", "existing tracks keep their flags"
    extracted = tracks.extract(out, 4, ws.path(f"check-{engine}"))
    assert formats.read(extracted).cues[0].text == "待っていた。", "legacy encoding converted to UTF-8"
    kinds = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", out],
                           capture_output=True, text=True).stdout.split()
    assert kinds.count("attachment") == 1, "font attachment lost"
    assert not [n for n in os.listdir(ws.root) if ".part." in n], "staging file left behind"
    pgs = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "s:2", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", out], capture_output=True, text=True).stdout.split()
    assert float(pgs[0]) == 0.5, f"PGS moved from 0.5 s to {pgs[0]} s"
    late = ws.path(f"late-{engine}.mkv")
    tracks.mux(source, [{"path": ws.path("c.sup")}], late, keep_existing=False, engine=engine)
    pgs = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "s:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", late], capture_output=True, text=True).stdout.split()
    assert float(pgs[0]) == 0.5, f"added .sup moved from 0.5 s to {pgs[0]} s"


def test_mux_with_ffmpeg():
    with Workspace() as ws:
        _check_mux(ws, "ffmpeg")


def test_mux_with_mkvmerge():
    if not shutil.which("mkvmerge"):
        print("  SKIP  (mkvmerge not installed)")
        return
    with Workspace() as ws:
        _check_mux(ws, "mkvmerge")


def test_mux_mp4_and_replacing_tracks():
    with Workspace() as ws:
        source = _source_mkv(ws)
        mp4 = ws.path("out.mp4")
        logged = []
        tracks.mux(source, [{"path": ws.write("x.vtt", "WEBVTT\n\n00:01.000 --> 00:02.000\nHi\n"),
                             "language": "es"}], mp4, container="mp4", log=logged.append)
        found = tracks.list_tracks(mp4)
        assert [t.codec for t in found] == ["mov_text"] * 3 and found[2].language == "es"
        assert any("image subtitles" in m for m in logged), logged
        try:
            tracks.mux(source, [{"path": ws.path("c.sup")}], ws.path("bad.mp4"), container="mp4")
        except MediaError as exc:
            assert "image subtitles" in str(exc)
        else:
            raise AssertionError("put PGS into MP4")
        assert not os.path.exists(ws.path("bad.mp4"))

        only = ws.path("only.mkv")
        tracks.mux(source, [{"path": ws.write("t.ttml", formats.render(formats.parse(SRT, "srt"), "ttml")),
                             "language": "it"}], only, keep_existing=False, engine="ffmpeg")
        found = tracks.list_tracks(only)
        assert [(t.codec, t.language) for t in found] == [("subrip", "it")], "TTML converted, old tracks gone"


def test_mux_task():
    with Workspace() as ws:
        source = _source_mkv(ws, pgs=False)
        job = {"id": "m", "task": "mux",
               "input": {"video": {"path": source},
                         "subtitles": [{"path": ws.write("new.srt", SRT), "language": "ko", "title": "Korean"}]},
               "options": {"container": "mkv", "keepExisting": True}, "output": {}}
        result = tracks.run_mux_task(job, _ctx(ws.path("work")))
        out = result.outputs[0]
        assert out.kind == "video" and out.path == ws.path("src.muxed.mkv"), out.path
        assert [t.language for t in tracks.list_tracks(out.path)] == ["en", "fr", "ko"]


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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
