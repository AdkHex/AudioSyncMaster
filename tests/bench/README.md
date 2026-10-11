# Dub-sync bench

Ground truth for the dub-sync engine, beyond the unit tests: real film audio,
edited in known ways, and every plan scored against what the edit really was.

## Cases

`make_cases.py` needs one source whose file carries two audio tracks that are in
step with each other: an original and a dub from the same release (a dual-audio
Blu-ray, say). It keeps the video and the original track, and builds dubs from
the other track with known edits:

- delay only: late, early, and at another frame rate
- missing head or tail
- few cuts, many short cuts, cuts in the middle of a shot, trims of 1–3 frames
- inserted material, a recap, reordered scenes, a replaced scene, dropouts
- PAL speed-up, resampled or pitch-kept; 23.976 against 24 fps; clock drift
- a different mix; low-bitrate mono; EAC3 5.1; MP3

Two measurements keep the truth honest:

- **The source's own offset.** The two tracks of one release are not always
  laid to the sample; a Blu-ray measured here carries its English 62.6 ms
  behind its Japanese.
- **Where each written file really starts.** Raw AC3 decodes 256 samples late,
  and a pitch-kept speed change shifts by its latency.

```sh
python tests/bench/make_cases.py SOURCE.mkv OUT/A --foreign OTHER.mkv
```

`OTHER.mkv` is anything else with a dub-language track 1. It supplies material
that does not belong to the video.

The `SOURCE.mkv` used here was built with ffmpeg as a 15-minute excerpt of a
dual-audio film. The video was scaled down, and both audio tracks were
re-encoded to FLAC so they share one exact clock:

```sh
ffmpeg -ss 600 -i FILM.mkv -t 900 -map 0:v:0 -map 0:a:ORIG -map 0:a:DUB \
  -c:v libx264 -preset veryfast -crf 26 -vf scale=-2:360 -c:a flac -sample_fmt s16 exA.mkv
```

### A source from an open film

Without a dual-audio release, `make_source.py` builds one from any film whose
music-and-effects stem is published. Blender's open movies publish theirs:
*Sintel* (`download.blender.org/durian/movies/`: `Sintel.2010.720p.mkv.zip`
and `sintel-m+e-st.flac`).

```sh
# The film's own English 5.1 against the stem with its 26 lines laid in Hindi:
python tests/bench/make_source.py Sintel.2010.720p.mkv sintel-m+e-st.flac sintel.mkv \
  --lines tests/bench/sources/sintel.lines.json
# A talkative film: the stem with ~190 lines spoken over it in English and in Hindi:
python tests/bench/make_source.py Sintel.2010.720p.mkv sintel-m+e-st.flac talky.mkv --dense 7
```

The stem's lag inside the film's mix is measured (AC3's 256 samples on
*Sintel*), each Hindi line is laid where the English one really starts, and the
voices are macOS's `say` (Lekha for Hindi). Such a source also carries the dub's
background alone (track 2) and its voices alone (track 3), so `make_cases.py`
builds three more cases from it, each with the voices' own ground truth:

- `voices_apart`: picture cuts the voices follow while the music is cut 10–60 s
  away, as a TV studio does; the voices sit seconds off their own music there
- `voices_apart_ntsc`: the same at 23.976 against a 24 fps video
- `bedless`: scenes where the dub kept its voices but lost its background, each
  just after a cut, so only their lines can place them

`OTHER.mkv` can be any other film with the same audio on tracks 0 and 1
(*Elephants Dream* was used).

## Running engines against the cases

```sh
python tests/bench/run_bench.py OUT/results OUT/A --engine head=. --engine old=/path/to/other/checkout --jobs 3
```

`--voices` runs the voice check after planning on the cases that judge the
voices (it needs the voice tools; `AUDIOSYNC_VOICE_TOOLS` points at an install).
For those, the table adds where the dub's lines play while it speaks: within
45 ms of their own picture, off it, or gone.

## What each rule is worth

```sh
python tests/bench/ablate.py OUT/ablation OUT/A OUT/B --baseline OUT/results/head
```

switches one of the planner's rules off at a time, plans every case again and
prints what the bench lost or gained without it. A rule whose removal changes
nothing anywhere earns its place only by a test that shows what it is for.

Each moment of the video (every 50 ms, leaving out 0.1 s around each true cut)
falls into one of these classes:

| Class | Meaning |
|---|---|
| correct | within 20 ms of where it belongs |
| close | within 45 ms |
| wrong | further off than 45 ms |
| missed | the dub has this moment but the original plays |
| false | dub material plays where none belongs |

`score.py CASE/case.json PLAN.json` scores one plan. `score.problems()` lists
where it goes wrong.

## Real pairs

A real pair has no ground truth. `real_check.py PLAN.json` holds a plan against
the shared music and effects, window by window, and flags fills where the dub
did have the scene.

A dub whose background holds two copies of itself, a tenth of a second apart,
reads "off" there wherever its dialogue follows the other copy. There the lips
decide: compare the dub's separated voices with the original's dialogue.
