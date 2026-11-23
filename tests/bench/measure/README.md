# Measurement bench

Ground truth for the delay measurement (`analyze_pair`), the way MkvBatchMux
uses it: real film audio, dubs with planted delays, and every result scored on
the number the app would fill in.

## Source

Blender's *Sintel* publishes its music-and-effects stem, which is what a dub
shares with the original. `build_source.py` turns the film and the stem into:

| File | What it is |
|---|---|
| `orig51.flac` | the film's own English AC-3 5.1 mix, decoded from the top |
| `me.flac` | the stem, moved onto the film's clock (it sits 256 samples late in the AC-3 mix) |
| `es_sparse.flac` | Spanish voices at the film's 26 subtitle times |
| `en_dense.flac`, `es_dense.flac` | English and Spanish lines every few seconds for the whole runtime |

The voices are macOS's (`say`); any language works, since only the fact that
the dialogue differs matters.

```sh
curl -O https://download.blender.org/durian/movies/Sintel.2010.720p.mkv.zip   # and unzip
curl -o sintel-me.flac "https://download.blender.org/durian/movies/sintel-m%2Be-st.flac"
ffmpeg -i Sintel.2010.720p.mkv -map 0:3 SRC/sub3.srt   # English subtitles
ffmpeg -i Sintel.2010.720p.mkv -map 0:4 SRC/sub4.srt   # Spanish subtitles
FFMPEG=ffmpeg python tests/bench/measure/build_source.py Sintel.2010.720p.mkv sintel-me.flac SRC SRC
```

## Cases

```sh
FFMPEG=ffmpeg MKVMERGE=mkvmerge python tests/bench/measure/make_cases.py SRC Sintel.2010.720p.mkv CASES
```

43 cases (about 3 GB), each with its truth in `CASES/truth.json`:

- **constant:** AAC/AC-3/E-AC-3/Opus/MP3, raw elementary streams and containers,
  5.1 and mono, remixed, quiet, dialogue-heavy, L/R swapped, inverted, a logo
  intro, a missing recap (-75 s), dense different-language dialogue, music that
  repeats sample for sample over 40% of the runtime, a one-hour file, a bad TV
  rip, a different music bed, a buried bed, phase-smeared, 44.1 kHz, a
  limited/compressed OTT mix
- **rate:** PAL 25/24 and 25/23.976, resampled and pitch-kept; 23.976 against 24
  both ways
- **drift:** speeds that match no conversion of the video's own rate
- **cut:** a censor cut; three TV edits
- **none:** unrelated audio, which must not be filled in

Truth is calibrated, not assumed. Each written file is decoded from the top and
correlated with what was encoded (raw AC-3 decodes 256 samples late, ADTS AAC
1024). A processing step that moves the content is measured too: a limiter's
lookahead, and a pitch-kept time-stretch, which moves real film content by
about +11 ms at its transients -- isolated clicks through the same stretch move
-20 ms, so clicks are not what it is calibrated on.

## Running

```sh
FFMPEG_DIR=/path/to/ffmpeg/dir python tests/bench/measure/run_bench.py . CASES out.json --jobs 4
# another revision, to compare:
mkdir -p /tmp/old && git archive v2.22.0 audiosync | tar -x -C /tmp/old
FFMPEG_DIR=... python tests/bench/measure/run_bench.py /tmp/old CASES old.json --jobs 4
```

Like the app: a survey of 6 x 45 s windows over +-60 s with the timeline off,
then for a pair it would search again (error, no delay, confidence under 0.5)
the fast route over +-300 s. Scored on `delayAtStartMs`:

| Verdict | Meaning |
|---|---|
| exact / good / close | filled in, within 2 / 10 / 45 ms |
| WRONG | filled in, further out -- the worst outcome |
| WITHHELD | not filled in (error, weak, cut, implausible) |
| rate-ok, drift-ok | the speed named (or the drift measured to 3%) and the start within 20 ms |
| cut-ok | a cut reported, with the delay before it within 10 ms |
| refused | unrelated audio left unfilled |

## Results

| | v2.22.0 | v2.23.0 |
|---|---|---|
| constant (30) | 30 within 0.8 ms | 30 within 0.8 ms |
| rate and drift (9) | 3 right (one 12.8 ms out); 0.1% speeds 31-45 ms out, pitch-kept PAL 0.7-2 s out or withheld | 9 right, all within 1.6 ms |
| three TV edits | filled in 34 ms out, called a drift | a cut, exact before it |
| censor cut | a cut, exact before it | a cut, exact before it |
| unrelated (2) | refused | refused |

MkvBatchMux's own test season (`make_media.py`), whose music repeats every
20 s, adds the case that moved the waveform search: an episode the old engine
placed 667 ms out at 99%, now exact.
