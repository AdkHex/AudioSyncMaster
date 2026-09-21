# AudioSyncMaster

Measures the timing offset between a video file and a separate audio track, and
writes a corrected file with the audio aligned.

Typical use: you have a movie or a season of episodes plus a dubbed audio track
that does not line up. AudioSyncMaster tells you by how much, how confident it
is, and whether the offset drifts over the running time — then fixes it.

## How it works

Both tracks are decoded to 16 kHz mono and reduced to onset-strength envelopes:
curves tracking where audio energy *rises*. Correlating envelopes rather than
raw waveforms survives the codec, loudness and channel-layout differences that
separate a dub from its source, because it responds to the timing of transients
rather than to sample values.

Each pair is measured at several points across the file, which gives three
things a single measurement cannot:

- **A confidence score** from the sharpness of each correlation peak. Audio that
  does not match produces a diffuse peak and is rejected rather than answered.
- **Drift detection** by fitting a line through the per-window offsets. A 25fps
  versus 23.976fps mismatch shows up as a slope, not as noise.
- **Robustness**, because the median of several windows survives one window
  landing on silence, a music cue, or a repeated phrase.

### Sign convention

**A positive delay means the audio track starts later than the video.** To
align, that much is trimmed from the audio's start. A negative delay means the
audio starts early and silence is inserted instead.

This is asserted directly by `tests/test_correlate.py` and verified end to end
by a real mux round-trip in `tests/test_mux.py`.

## Modes

- **Movies** — several video files against one audio track.
- **Series** — a folder of episodes against a folder of dubs, paired by name.
- **Compare** — every video against every audio track, to work out *which*
  release a dub was timed for. A dub synced to a WEB-DL drifts against a BluRay
  with different framing; comparing both at once shows which one it belongs to.
  Capped at five files per side, since the work is the product of both.
- **Dub sync** — a queue of videos against their dubs, each a different edit
  of its video: scenes missing, a longer logo, a different speed. Pairs each
  episode (or movie) with its own dub and writes, in parallel, a track the
  length of each video with the original audio filling every gap. See below.

## Audio tracks

A container often carries several audio streams — the original language, a dub,
a commentary. When a file has more than one, a picker appears showing each
stream's language, title, codec and channel layout, and the choice reaches
ffmpeg as `-map 0:a:N`. Without it every comparison silently used the first
stream, which on a disc rip is often not the one you want.

## Frame rate and cuts

Steady drift almost always has one cause: audio mastered at a different frame
rate. The app reads each file's rate and names the conversion — "timed against
a 25fps source, but this video is 23.976fps" — along with the resampling factor
that cancels it exactly.

Drift larger than any standard conversion can produce (beyond ~45 ms/s) means
something else: the files contain different material. Those are reported as
**Different cut** and excluded from the fixable set, because no single delay or
speed ratio aligns them. For those, there is dub sync.

## Dub sync

A dub is often a different edit of the same film: a scene the dubbing studio
never received, a recap trimmed for broadcast, a longer logo at the head. No
single delay describes it. Dub sync works out, along the whole runtime, which
stretch of the dub belongs at each moment of the video, and writes one track
exactly the video's length: the dub wherever the dub exists, the video's own
audio wherever it does not, crossfaded at every seam.

```sh
python python/dubsync.py MOVIE.mkv MOVIE.hin.eac3
python python/dubsync.py MOVIE.mkv MOVIE.hin.eac3 --codec eac3 --mux --lang hin
python python/dubsync.py MOVIE.mkv MOVIE.hin.eac3 --plan-only
python python/dubsync.py --from-plan MOVIE.hin.dubsynced.dubsync.json -o fixed.flac
```

The result is `MOVIE.hin.dubsynced.flac` beside the video (`--codec` for
wav, aac, ac3, eac3 or opus; `--mux` for a copy of the video with the track
added), a JSON plan beside it, and a report:

```
4 stretches of dub, 4 fills from the original (0:00:21.458 in all), fills at +4.4 dB
  fill  0:00:00.000 - 0:00:00.735  <- org 0:00:00.000                   0.7s  dub starts late
  dub   0:00:00.735 - 0:00:59.796  <- dub 0:00:01.541     +0.806s  match 0.08
  fill  0:00:59.796 - 0:01:15.010  <- org 0:00:59.796                  15.2s  dub is cut here
  dub   0:01:15.010 - 0:02:59.996  <- dub 0:01:00.815    -14.194s  match 0.07
  ...
--- checking the finished track against the original ---
the finished track sits 0ms from the original typically and 0ms at its worst, measured at 10 spots.
```

Each `dub` line is a stretch of the video's timeline, where in the dub it was
found, and the offset (dub time minus video time). Each `fill` is a stretch
the dub does not have, taken from the video's own audio and re-levelled to
sit among the dub. The check at the end decodes the finished track and
measures it against the video at a dozen spots and in a sweep of short
windows, so a mistake shows up as a number rather than on first viewing.
The check sums the three bands' correlations and, among peaks as tall as
the tallest, reports the one nearest zero: a beat gives the full band a
peak every period, and a track that sits where it should gives one at
zero in every band.

How it works: the frame rate is settled first, before any offset or cut is
trusted. When the video carries a frame rate, the dub's mastering rate is
checked against it directly -- a 25fps-mastered dub on a 23.976fps video is
played at 25/23.976 throughout, and the plan says so ("video 23.976 fps,
dub mastered at 25 fps"); a dub at the video's own rate is confirmed, not
assumed. Without metadata, the rate is still verified from the audio's
symptoms. Then both tracks are reduced to onset envelopes, since the music
and effects under a dub are the same stems as under the original even
though the dialogue is not -- three envelopes each: the whole spectrum,
the 30-250 Hz band (bass, footsteps, rumble) and the 4-8 kHz band
(ambience, foley). The dialogue sits between those two bands, so in a
scene with no music, where the full band hears only two languages'
consonants and agrees on nothing, the low band still hears the bed the
mixes share; on a real pair it found 21 of the 23 minutes the full band
had given up on. The video is cut into 30-second windows, each
correlated against the dub across every plausible offset, and the offsets
are chosen as one path through all the windows at once, which stays put for
free and pays to jump -- so a window that locks onto a repeated musical
phrase cannot splice the track on its own. Every stretch is then measured
from inside at 2 ms, and split wherever the offset steps, so a one-frame cut
in the middle of a scene is found too. Every cut is placed where the two
tracks stop agreeing: first on the envelopes, then, where the two mixes
demonstrably share a waveform, on the waveform itself, to the sample. Every
gap is then searched again at the envelope's full 2 ms resolution, with
windows of 10, 30 and 90 seconds and only the offsets the neighbouring
stretches allow -- quiet scenes whose shared music and effects are too
faint for the coarse pass are found this way, cuts inside them included.
Stretches where the dub has gone silent while the video has not are
filled; silence in both is a pause, not a cut. When more than a tenth of
the video is still without dub after that, every window of the video is
searched across the whole dub at 2 ms: a dub cut as TV episodes -- with
recaps, openings and endings between the film's scenes, and the episodes
in any order -- puts scenes at offsets the coarse pass cannot reach, and
its sharp onsets do not survive the coarse pooling. A cue that recurs
(a theme heard four times, an opening every episode) correlates at every
occurrence, loudest where it is mixed loudest, so each window keeps its
few tallest peaks and takes the one that continues what is already
known: the one near the neighbouring stretches' offset, or the one that
does not wind the dub back by a minute or two; a whole episode back is
allowed, since that is what episodes out of order look like. The songs
and recaps are simply never used; nothing is trimmed from the video, and
only the scenes the dub really lacks are filled. A dub at a different
speed is caught two ways: a large conversion (PAL) by trying the standard
ones on the audio, and a small one (24 against 23.976 fps, a millisecond a
second) by reading the drift off the coarse alignment itself; either way the
dub is decoded at the compensating rate throughout. A dub conformed scene
by scene, with some scenes a few frames out, is followed scene by scene:
steps as small as a few milliseconds are followed when the readings are
sharp enough to tell them apart, and a step is believed only when the
piece it cuts out, taken whole, agrees better at its own offset than at
its neighbour's. The 4-8 kHz band has the last word on where a stretch
sits: at a small picture trim -- a few frames -- the dub's dialogue and
effects follow the picture but its music is often left running, so for
a while the music sits a whole number of frames from the effects, and
the dialogue, recorded to the picture, goes with the effects (the
ambience steps exactly at the shot changes at the effects' offset). Where
the two disagree the effects decide, and a music-only level that is not a
whole number of frames from the effects' level -- a cue laid twice, a
beat's alias -- is folded into it rather than followed. Two stretches
less than a tenth of a second apart are the same scene, and nothing
between them is ever filled: the gap is bridged with each side keeping
its own offset, because a few seconds a frame out of lip-sync is a far
smaller mistake than the other language over a scene the dub has. A gap
the dub is audible across whose sides sit up to two seconds apart -- or
any such gap where a wrongly placed step could misplace no more than five
seconds of dub -- is bridged too, the step put where the agreement changes
from one offset to the other, or in the middle when the agreement says
nothing, and the note says which; the frames the dub lacks are filled at
the step. An edge the evidence cannot place closer than a quarter of a
second is pulled inward by its uncertainty, so the dub only ever plays
where the dub belongs and the original takes the doubt. A track that is
already in sync comes back as one stretch at 0 ms with nothing filled --
feed the finished track back in as the dub to check it.

What it will not do: it keeps the dub across a passage that merely
correlates weakly when the offset is the same either side and the dub is
audible there, because replacing a scene that has the right language with
one that does not is the worse mistake; the plan notes where it did so
(as a note, not a warning: nothing was changed there). At the very start
and end of the file, where there is only one side to vouch for it, the
dub is kept this way for at most half a minute; a longer uncorrelated
leader is filled and marked *Replaced*, since on a dub made of episodes it
was another episode's ending.
`--fill-unmatched` (in the app: Settings, "Replace stretches that did not
correlate") fills such passages from the original instead, and marks those
fills `dub audible but did not correlate; replaced` -- shown as *Replaced*
in the app -- so they can be told from real cuts and checked by ear. Leave
it off unless a kept passage turns out to be the wrong scene.
Cut placement is only as precise as the shared bed allows: at a cut that
falls in a silence, the edge lands where the bed stops. Offsets are reported
as decoded, so a raw AC3 or E-AC3 dub reads 5.3 ms of decoder priming into
them; the rendered track is placed by the same decode and is not affected,
and a raw AC3 or E-AC3 output is written early by the same amount so that it
plays on the sample. Manual corrections go in the JSON plan and come back
in with `--from-plan`.

In the app, the **Dub sync** tab does the same thing, as a queue rather
than a single pair. It opens on one of two scopes, chosen at the top of the
sidebar: **Movies** pairs each video with its dub by filename; **Series**
pairs a season by season and episode number. Drop a folder of movies (or
episodes) on one side and the folder of dubs on the other -- any format
ffmpeg reads, bare or inside an MP4/MKV -- and the pairing preview shows
what will run, with hand repairs where a match is wrong. Press Sync and the
whole queue runs in parallel, up to the configured worker count. Each pair
reports its plan as soon as its analysis is done -- before its track is
written -- and each finished row shows the written track and the check
against the video; a row's plan is collapsed behind its summary, so a
season reads as a list of results rather than a wall of tables. Choose
what to write the synced tracks as (each dub's own codec by default, or
FLAC, E-AC3, AC3, AAC, Opus, WAV) and whether to also write a copy of each
video with the track added. Stop interrupts every job. The engine is
reached through the bridge's `dubsyncBatch` command, which reports
`dubsyncJobStart`, `dubsyncJobProgress`, `dubsyncJobPlan`, `dubsyncJobDone`
and `dubsyncBatchDone` (the single-pair `dubsync` command, with
`dubsyncProgress`, `dubsyncPlan` and `dubsyncDone`, remains for one-offs).

## Reviewing results

Every measurement expands to show what it was built from: the offset at the
start and end of the file, how many sample windows were usable, the frame rate
and codec of each source, and any codec delay that was removed. Delays are also
given in video frames, which is how a mismatch is usually judged.

**Preview** renders a short excerpt with the measured delay applied and opens it
in your player. Hearing the dub land on the picture settles a borderline result
in a way a confidence score cannot.

Pairings can be corrected before a run. If series matching gets one wrong, pick
the right audio from the dropdown, or skip that video entirely — the engine uses
the corrected pairs verbatim rather than re-matching.

## Requirements

- Node 20+
- Python 3.9+
- Rust (stable) for desktop builds
- FFmpeg on `PATH`, or `ffmpeg`/`ffprobe` placed in `src-tauri/resources/ffmpeg/`
  to be bundled into the installer

## Development

```sh
./dev.sh              # macOS / Linux
dev.bat               # Windows
./dev.sh --sidecar    # also build the frozen Python engine
```

## Tests

```sh
npm run test:all      # typecheck + frontend + Python
npm run test          # frontend only
npm run test:py       # Python only (generates fixtures on first run)
```

Fixtures are synthetic audio pairs with exactly known offsets, generated by
`tests/make_fixtures.py`. They are the ground truth for every algorithm change:
if a change breaks the sign convention or lets unrelated audio through, these
fail immediately.

## Layout

```
audiosync/          Analysis engine (Python)
  correlate.py      Offset estimation, confidence scoring
  analyze.py        Multi-window analysis, drift detection
  media.py          FFmpeg decoding, probing, process lifecycle
  matching.py       Pairing video and audio files
  mux.py            Applying corrections
  batch.py          Bounded-concurrency batch runner
  dubsync.py        Dub sync: which stretch of a cut dub belongs where
  dubrender.py      Writing the synced track, and muxing it
python/bridge.py    Line-delimited JSON bridge to the desktop host
python/dubsync.py   Dub sync command line
src-tauri/          Tauri host (Rust)
src/                UI (React + TypeScript)
tests/              Python tests and fixture generation
```

The UI talks to Rust over Tauri commands; Rust talks to the Python engine over
newline-delimited JSON on stdin/stdout. All three layers use camelCase field
names so payloads cross the boundaries unchanged.
