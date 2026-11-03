# sya: locate a recording from the aircraft it overheard

`sya` estimates **where and when** an audio recording was made in the continental US.
It works from the faint Doppler-shifted tones of passing aircraft, matched against
historical ADS-B flight tracks. It reimplements the technique in colsto's video
["I discovered a sound that leaks your exact location"](https://www.youtube.com/watch?v=UytYrhYLOKI),
built from first principles because no code was published.

```
$ sya locate rally.mp4 --date 2023-08-08 --tz America/Phoenix --state AZ
                       Detected flyovers
 #  window (s)  f0 Hz  speed m/s (kt)  CPA slant m  t_ca s  rms Hz
 0     161–260  108.3        83 (162)         1088   200.0    0.11
 1     619–671   86.5       160 (311)         1499   640.0    0.19
search area: Arizona (AZ)  start window: 2023-08-08T07:00Z → 2023-08-09T07:00Z
time chunks ━━━━━━━━━━━━━━━━━━━━ 24/24 through 2023-08-09 07:00Z · 6 solutions
                       Pruning passes
 pass             in         out     kept   time
 load                   1,212,553           0.9s
 altitude   1,212,553     402,118    33.2%  0.3s
 resample       2,710   1,804,113           1.1s
 candidates[0] …
 join          17,233          34     0.2%  0.01s
 …
LOCATED 33.43097, -112.08560 (AZ)  start 2023-08-08T16:05:17Z  flights a2c1f3, ad41b0
```

## Install

```bash
uv venv && . .venv/bin/activate && uv pip install -e '.[test]'
sya --help
```

The base install needs Python ≥ 3.10, ffmpeg (for video and arbitrary audio containers),
and network access to Hugging Face and AWS for the first fetch of each day.

## Usage

```
sya locate AUDIO  [time]  [area]  [options]
```

**Time.** These flags bound when the recording *started*; pick exactly one:

| flag | meaning |
|---|---|
| `--date 2023-08-08` | that whole day (in `--tz`, default UTC) |
| `--dates 2023-08-07 2023-08-09` | inclusive range of days |
| `--between 2023-08-08T17:00 2023-08-08T20:30` | timestamp range (ISO-8601 or unix) |
| `--at 2023-08-08T20:10Z --tolerance 15m` | a timestamp ± tolerance |
| `--when 2023-08-07..2023-08-09` or `--when TS..TS` | free form |

**Area.** The default is the continental US. You can narrow it:

| flag | meaning |
|---|---|
| `--state AZ` or `--state "North Dakota"` | one CONUS state (Census boundaries) |
| `--center 46.8,-100.78 --radius 25km` | a disc. The two flags must be given together. Units: `km`, `mi`, `nm`, `m` |

**Options:**

| flag | meaning |
|---|---|
| `--timeout 60` | search time budget in seconds. If one solution isn't isolated in time, `sya` prints the ranked list of remaining candidates. |
| `--fmin/--fmax/--harmonics` | frequency band of the tone's fundamental (default 60–300 Hz, 3 harmonics) |
| `--window START:END` | manual flyover windows in seconds, when automatic segmentation struggles |
| `--clip START:END` | analyse only part of a long file |
| `--temp` | air temperature °C, which sets the speed of sound |
| `--json out.json` | full result, including candidates and per-pass timings |
| `--world` | `real` (default: historical ADS-B), `synth:<preset>[:seed]`, `csv:<file>`, `readsb:<dir>` |
| `--offline`, `--cache-dir` | use only cached data, or choose the cache location |

**Other commands:**

* `sya events AUDIO --plot ev.png`: detect and fit flyovers only, with a spectrogram plot. Use this to choose the band and windows.
* `sya cache ls | fetch --date D | prune [--limit 5G] | clear | freeze …`: manage the ADS-B range cache.
* `sya simulate synth:bismarck:3 --out sim/`: write a synthetic recording plus its ground truth.
* `sya validate tests/validation/cases.toml`: run the end-to-end validation cases.
* `sya bench`: per-pass runtime on a synthetic world.

## How it works

### 1. Flyovers from audio (`sya/audio.py`)
1. Compute a blocked float32 STFT (1 s Hann window, 0.1 s hop).
2. Take log power and whiten it robustly: subtract the per-bin median (removes hum) and the per-frame median (removes gusts), then scale by MAD.
3. Harmonic sum: score each fundamental `f` by `Σₖ z(k·f)`.
4. Track the ridge with Viterbi, using an O(F) distance-transform transition.
5. Segment flyovers with hysteresis on ridge SNR.
6. Fit each flyover with the closed-form retarded-time straight-pass model, `f = f0·c/(c + dr/dtₑ)` with `t_r = tₑ + r/c`. The fit gives tone `f0`, speed `v`, closest slant distance `d` and closest-approach time `t_ca`.

### 2. Search (`sya/search.py`)
The time window is processed in 1-hour chunks. Each chunk runs these passes, and the
Rich UI shows each pass's input and output counts and timings live:

| pass | what it does |
|---|---|
| load | Reads ADS-B rows for the chunk from the range cache, clipped to the area's bounding box. |
| altitude | Drops rows too high to be any detected flyover. |
| resample | Puts every track on a 1 s grid (ECEF position and velocity). |
| terrain | Looks up ground height under every sample (AWS Terrarium tiles). |
| candidates[k] | For flyover *k*, keeps samples with speed ≈ `v` and height above ground ≤ `d`. Each yields the ground points at slant range `d` perpendicular to the track, plus the recording start that would imply. |
| region | Drops candidate points outside the state, disc or CONUS. |
| join | Uses a KD-tree in (position, start time) to keep points where ≥ 2 flyovers agree. The fixed gap between flyovers in the recording is what pins the location. |
| dedupe | Keeps one representative per (flight set, start-time bucket, ~2 km cell). |
| silence | Penalises points where another low aircraft passed closer than any detected one during the recording without being heard. |
| prescore | Computes the full Doppler-curve misfit along the real track, with a coarse scan of start time. |
| refine | Runs Nelder–Mead over (east, north, start, per-flight ADS-B time shifts with a Gaussian prior). |

The decision rule:
* **solved:** the best fit has an RMS under 1.5 Hz, and no rival elsewhere (> 1.5 km or > 30 s away) is within 2.5× of its cost.
* **ambiguous:** otherwise, or when the timeout hits. The ranked candidate list is printed and written to `--json`.
* With a single detected flyover the problem is under-determined: the solution slides along the flight track. `sya` reports this as ambiguous with candidates.

### 3. World data (`sya/world.py`, `sya/repair.py`, `sya/terrain.py`)
A `World` is any object that provides `tracks(t_lo, t_hi, bbox)` and `elevation(lat, lon)`.
The search cannot tell real worlds from synthetic ones:

* **`real`.** Historical ADS-B from adsb.lol's `globe_history`, via the Hugging Face parquet
  mirror `alexisplacet/adsblol_globe_history` (ODbL). It covers 2023-03-10 to 2024-01-01
  and 2025-12-19 to 2026-02-03. The mirror lost the integer seconds of every trace
  timestamp. `sya.repair` reconstructs them using two sources:
  * The mirror's heatmaps: 10 s snapshots, which are *lower bounds*, because a fix
    is observed 0.5–25 s after its slot stamp.
  * Ground-speed-based chaining between consecutive positions.

  Validated against ADSBExchange full traces for the same aircraft on 2023-08-01, the
  median error is about −0.5 s (IQR −2.5 to +0.6 s). Refinement absorbs the rest
  through per-flight time-shift nuisance parameters.
* **Terrain.** AWS Open Data Terrarium tiles at zoom 10, plus a coarse CONUS geoid offset
  (ADS-B geometric altitude is ellipsoidal).
* **`synth:<preset>[:seed[:single]]`.** Synthetic worlds near Bismarck, Fargo, Phoenix,
  Flagstaff or Seattle. Each has background airliners, arrivals and departures, GA
  traffic, two guaranteed flyovers (or one, with `single`), noisy ADS-B, and rendered audio.
* **Hybrid** (`sya.hybrid`). Real ADS-B traffic with synthetic audio rendered at a
  chosen observer. Each flight's sound is time-jittered to mimic ADS-B timing error.

### Range cache
* **Location.** `~/.cache/sya` by default; override with `SYA_CACHE_DIR`.
* **Layout.** One zstd parquet file per UTC hour, CONUS-filtered, under
  `adsb/<source>/<YYYY-MM-DD>/`. Days are fetched once (about 2 GB download, about 250 MB
  cached) and then served from disk for any time range or region.
* **Size limit.** Set with `SYA_CACHE_LIMIT`; the default is min(20 GB, 40% of free disk).
  The cache is pruned least-recently-used, by day.
* **Freezing.** `sya cache freeze --between … --state ND --out tests/data` copies a small
  time×area subset, plus terrain tiles, into the repo for offline tests.

## Tests, validation, benchmarks

```bash
pytest -q                                   # unit + per-pass isolation + validation cases (offline)
pytest -m holdout                           # held-out real-world cases
sya validate tests/validation/cases.toml    # same validation cases, as a table
sya bench --repeat 3                        # per-pass timings, full pipeline
python benchmarks/bench_scaling.py          # scaling with traffic volume and window length
```

* `tests/test_passes.py` exercises each search pass in isolation: candidates contain the
  truth, region filtering, join consistency and rejection when times disagree, dedupe,
  silence detection of an injected unheard aircraft, refine convergence, the decision rule,
  timeout behaviour, and single-flyover ambiguity.
* `tests/validation/cases.toml` holds end-to-end cases. Each one is an audio file plus
  flags, mapped to an expected location ± tolerance (and optionally start time ± tolerance).
  The cases cover every time-input form, `--state`, `--center/--radius`, negative
  area/state cases, the ambiguous candidate list, and real-ADS-B hybrid cases that read
  the frozen data in `tests/data`.

RESULTS_PLACEHOLDER

## Limitations
* **Propagation.** Straight-line, still-air propagation is assumed. Wind and temperature
  gradients bias speed and distance; budget a few hundred metres.
* **Tones.** You need ≥ 2 flyovers with trackable tones and ADS-B-equipped aircraft.
  Military aircraft and many GA aircraft without ADS-B can't be matched.
* **Overlapping tones.** The ridge tracker follows one tone at a time. With overlapping
  aircraft, use `--window`.
* **Timing.** The ADS-B timing reconstruction (see above) adds roughly ±3 s per flight.
* **Search cost.** Cost scales with ADS-B rows in the window: about 25 M rows per CONUS
  day. A full-CONUS multi-day search may exceed the default 60 s budget; narrow with
  `--state` or `--center/--radius`, or raise `--timeout`.
