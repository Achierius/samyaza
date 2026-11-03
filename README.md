# sya: locate a recording from the aircraft it overheard

`sya` estimates **where and when** an audio recording was made in the continental US.
It works from the Doppler-shifted tones of passing aircraft, matched against historical
ADS-B flight tracks. It reimplements the idea in colsto's video
["I discovered a sound that leaks your exact location"](https://www.youtube.com/watch?v=UytYrhYLOKI).
No code was published, so everything here is built from the physics.

> **Status.**
> * Synthetic recordings are located to roughly 10–100 m.
> * Synthetic audio rendered over *real* ADS-B traffic in North Dakota and Arizona is located
>   to 15–100 m, with start time within ±2 s.
> * On **real field recordings**, `sya` recovers *which aircraft* flew over and *when*: the
>   Doppler curve fits that aircraft's ADS-B track at the true location, and no other
>   aircraft fits.
> * A full blind position fix from real audio has **not** been demonstrated yet. Every real
>   recording found so far had at most one usable ADS-B-equipped flyover; see "Real
>   recordings" below.
> * An unconstrained CONUS × multi-hour search in dense traffic is not reliable; constrain
>   it with `--state` or `--center/--radius`.

## Quick start

```bash
uv venv && . .venv/bin/activate && uv pip install -e '.[test]'

sya locate clip.mp4 --date 2023-08-08 --tz America/Phoenix --state AZ
sya locate clip.wav --between 2023-08-08T07:00 2023-08-08T09:00 --tz America/Chicago
sya locate clip.wav --dates 2023-08-07 2023-08-09 --center 46.88,-96.79 --radius 40km
```

A real run: real ADS-B traffic near Fargo with rendered audio, from validation case
`hybrid-nd-fargo-state-between-local`. The table is abridged.

```
search area: North Dakota (ND)   start window: 2023-08-08T12:00:00Z → 2023-08-08T14:00:00Z
time chunks ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2/2 through 2023-08-08 14:00Z · 600 solutions 0:00:08
                            Pruning passes
 pass                in        out    kept     time
 decode                  3,600,000           0.26s
 events               1         11          16.04s
 load                       18,576           0.54s
 altitude        18,576     13,803   74.3%   0.01s
 resample           138     62,018           0.04s
 candidates[0]   62,018     10,069   16.2%   0.03s
 …  (one candidates/region pair per detected flyover)
 join            79,534      7,992     10%   0.13s
 dedupe           7,992      1,693   21.2%   0.02s
 silence          1,693      1,690   99.8%   0.01s   radius 123 m
 prescore         1,693        600   35.4%  10.18s
 polish             600        600    100%  19.22s
 refine             600         40   6.67%   9.31s
LOCATED 46.90789, -96.81996 (ND)  start 2023-08-08T13:02:01Z  cost 2.600  flights ab0e05, acfc75, …
```

The truth was 46.90821, −96.82014, start 13:02:00Z, so the result is 38 m and 1 s off.

## CLI

```
sya locate AUDIO [time] [area] [options]      locate a recording
sya events AUDIO [--plot ev.png]              detect + fit flyovers only (choose band/windows)
sya cache  ls | fetch | prune | clear | freeze
sya simulate synth:<preset>[:seed[:single]]   write a synthetic recording + truth
sya validate [cases.toml]                     end-to-end validation table
sya bench                                     per-pass runtime on a synthetic world
```

**Time.** These flags bound when the recording *started*. Use exactly one. Naive times are
read in `--tz` (default UTC).

| flag | meaning |
|---|---|
| `--date 2023-08-08` | that whole day |
| `--dates 2023-08-07 2023-08-09` | inclusive range of days |
| `--between 2023-08-08T07:00 2023-08-08T09:00` | timestamp range (ISO-8601 or unix) |
| `--at 2023-08-08T13:00Z --tolerance 20m` | a timestamp ± tolerance |
| `--when 2023-08-07..2023-08-09` / `--when TS..TS` | free form |

**Area.** The default is the continental US.

| flag | meaning |
|---|---|
| `--state ND` / `--state "North Dakota"` | one CONUS state (Census 2018 boundaries) |
| `--center 46.88,-96.79 --radius 40km` | a disc. Both flags are required together. Units: `km`, `mi`, `nm`, `m` |

**Options:**

| flag | meaning |
|---|---|
| `--timeout 60` | search budget in seconds. If a single solution isn't isolated in time, the ranked list of remaining candidates is printed (and written to `--json`). |
| `--fmin/--fmax/--harmonics` | band of the tone's fundamental. Defaults: 60–300 Hz, 3 harmonics, where the top harmonic is skipped, see `audio.harmonic_sum`. |
| `--max-slope`, `--min-len`, `--window S:E`, `--clip S:E` | flyover detection controls |
| `--temp` | air temperature (°C), which sets the speed of sound |
| `--world` | `real` (default), `synth:<preset>[:seed]`, `csv:<file>`, `readsb:<dir>` |
| `--cache-dir`, `--offline`, `--terrain tiles[:zN]\|flat:<m>` | data sources |
| `--min-window` | shortest flyover curve to fit, in seconds (default 30); lower it for close, fast passes |
| `--anywhere` | experimental: allow a `--center/--radius` disc outside CONUS (regional ADS-B fetched on demand) |

## How it works

### 1. Flyovers from audio (`sya/audio.py`)
1. **Spectrogram.** A blocked float32 STFT (1 s Hann window, 0.1 s hop). Log power is
   robustly whitened: the per-bin median (hum) and the per-frame median (gusts) are
   subtracted, then values are scaled by MAD.
2. **Harmonic sum.** Each candidate fundamental is scored across its harmonics.
3. **Iterative ridge extraction.**
   * Viterbi tracks the strongest ridge, using an O(F) distance-transform transition.
   * The ridge is cut at SNR hysteresis and at jumps.
   * Inside each ridge, a window is fitted around every steep descent (an "S-curve"),
     trying several widths.
   * The ridge and its harmonics/sub-harmonics are masked, and the process repeats so
     that overlapping aircraft separate.
4. **Fit.** Each window is fitted with the closed-form retarded-time straight-pass model
   `f = f0·c/(c + dr/dtₑ)`, `t_r = tₑ + r/c`. The fit gives tone `f0`, speed `v`,
   closest slant distance `d` and closest-approach time `t_ca`. Implausible fits are
   rejected.

### 2. Search (`sya/search.py`)
The start-time window is cut into 1-hour chunks. Up to 4 worker processes run these passes
on each chunk, and Rich shows their in/out counts and times live:

| pass | purpose |
|---|---|
| load | Read ADS-B rows for the chunk from the range cache, clipped to the area's bounding box. |
| altitude | Drop rows too high, or moving too fast (raw speed), to be any usable flyover. |
| resample | Put every track on a 2 s grid (ECEF position and velocity). |
| candidates[k] | For flyover *k*, keep samples with speed ≈ `v` and height above ground ≤ `d`. Terrain is looked up only for survivors. Each sample yields the ground points at slant `d` perpendicular to the track, plus the recording start it implies. |
| region[k] | Keep points inside CONUS, the state or the disc. Uses a 0.05° raster, with exact polygon tests only in border cells. |
| join | Use a KD-tree in (position, scaled start time) to keep points where ≥ 2 flyovers agree. Anchors are the most selective flyovers; flyovers heard > 5 km away are left out of the join. |
| dedupe | Keep one representative per (flight set, 20 s, ~2 km). |
| silence | Penalise points where an unheard low aircraft passed closer than the detected flyovers. |
| prescore | Full Doppler-curve misfit along the real tracks. It uses 1 s decimated curves, a vectorised scan of start time (±15 s) and a 3×3 grid of positions (±500 m). |
| polish | A short Nelder–Mead over (east, north, start) for the globally best few hundred candidates. |
| refine | A full Nelder–Mead over (east, north, start, per-flight ADS-B time shift), with the shifts under a Gaussian prior. It then tries to attribute still-unexplained flyovers to any nearby flight ("explain-more") and re-refines. |

* **Scoring.** Each flyover contributes a Huber misfit capped at 0.5, which is also the
  cost of leaving it unexplained. Residuals are measured relative to the tone frequency
  (expressed as Hz at a 100 Hz tone), so 1 kHz jet fan tones and 100 Hz propeller tones are
  judged alike.
* **solved:** the best solution explains ≥ 2 *independent* flyovers at < 1 Hz RMS
  (several spectral lines of one aircraft's single pass count once), and no other
  solution elsewhere (> 1.5 km or > 30 s away) that also explains ≥ 2 flyovers is within
  0.5 of its cost.
* **ambiguous:** otherwise, including after a timeout, the candidates are listed.
* **none:** nothing explains any flyover.
* Every stage has a minimum amount of work it completes even after the deadline, so a
  timeout still produces candidates.

### 3. World data (`sya/world.py`, `sya/repair.py`, `sya/terrain.py`)
A *world* is anything with `tracks(t_lo, t_hi, bbox)` and `elevation(lat, lon)`. The search
cannot tell the following apart:

* **Real ADS-B.** adsb.lol `globe_history`, via the Hugging Face parquet mirror
  [`alexisplacet/adsblol_globe_history`](https://huggingface.co/datasets/alexisplacet/adsblol_globe_history)
  (ODbL). Coverage: 2023-03-10 → 2024-01-01 and 2025-12-19 → 2026-02-03.
  * **Timestamp repair.** The mirror's trace files lost the integer seconds of every
    timestamp; only the fraction survives. `sya.repair` rebuilds them using:
    * relative timing from distance ÷ reported ground speed, snapped to the surviving
      fractional seconds;
    * the mirror's 10 s heatmaps, as anchors. Their stamps are slot *starts*: the true fix
      time is 0.5–25 s later (measured), so they are used as lower bounds in a windowed
      90th-percentile estimator.
  * **Accuracy.** Against ADSBExchange's full traces for the same aircraft on 2023-08-01,
    the median error is −0.5 s (IQR −1.7 to +1.0 s; 44% within 1 s, 68% within 2 s), with
    a tail of 10+ s. Refinement absorbs this through per-flight time shifts.
* **ADSBExchange samples** (1st of each month since 2016, exact timestamps). Fetch a region with
  `sya cache fetch --source adsbx-region --center … --radius … --between …`. Aircraft are
  found from snapshots around the time window, then their full traces are downloaded.
* **Terrain.** AWS Open Data *Terrarium* tiles (zoom 10 by default), with a −28 m CONUS
  geoid offset, because ADS-B geometric altitude is ellipsoidal.
* **Synthetic worlds** (`synth:<preset>[:seed]`; presets `bismarck`, `fargo`, `phoenix`,
  `flagstaff`, `seattle`). Each has background airliners, arrivals and departures, GA
  traffic, two guaranteed flyovers, noisy ADS-B (sampled every 1–8 s, 25 ft altitude
  quantisation) and rendered audio. Add `:single` to keep only one flyover.
* **Hybrid worlds** (`sya.hybrid`). Real (frozen) ADS-B, with audio rendered from every real
  aircraft heard at a chosen observer. Each flight is jittered ±1.5 s to mimic ADS-B timing
  error.

### Range cache
* **Location.** `~/.cache/sya`, or set `SYA_CACHE_DIR`.
* **Layout.** One zstd parquet file per UTC hour, CONUS-filtered, under
  `adsb/<source>/<YYYY-MM-DD>/`. A day is fetched once (about 1.8 GB download, 3–5 min with
  4 processes; about 300 MB cached) and then serves any range or region.
* **Size limit.** Set with `SYA_CACHE_LIMIT`; the default is min(20 GB, 40% of free disk).
  Days are pruned least-recently-used. Offline readers never touch the LRU markers.
* **Freezing.** `sya cache freeze --state ND --between … --out tests/data` copies a
  time×area subset plus z8 terrain tiles. `tests/data` holds about 19 MB of frozen ADS-B
  and terrain for the validation cases.

## Validation results

```bash
pytest -q                     # 38 unit/isolation tests + 16 end-to-end validation cases (offline, ~7 min)
pytest -m holdout             # held-out real recordings (network: downloads audio + ADS-B days)
sya validate tests/validation/cases.toml
sya validate tests/validation/holdout.toml
```

**Unit and isolation tests** (`tests/test_*.py`) cover:
* regions (state lookup, CONUS, disc, raster vs exact);
* time specs and CLI flag rules;
* the track table, resampling, cache fetch-once, offline mode, LRU pruning and freezing;
* every search pass on its own: candidates contain the truth; the region pass drops
  outside points; the join keeps the consistent pair and rejects a 5-minute timing shift;
  dedupe; silence flags an injected unheard aircraft; prescore and refine converge; the
  decision rule; timeout; single-flyover ambiguity;
* audio: ridge tracking, pass fitting, harmonic sum.

**End-to-end cases** (`tests/validation/cases.toml`): each is audio + flags → location ±
tolerance (and start ± 5 s). All 15 pass:

| case | flags exercised | result |
|---|---|---|
| synth Bismarck ND | `--date`, CONUS | 113 m, −1.0 s |
| synth Fargo ND | `--dates` + `--tz` | 18 m, +0.1 s |
| synth Phoenix AZ | `--between` + `--tz` | 15 m, +0.1 s |
| synth Flagstaff AZ | `--at --tolerance` | 22 m |
| synth Seattle WA | `--when TS..TS` | 14 m |
| synth Bismarck ND | `--state ND` | 113 m |
| synth Bismarck ND | `--state Arizona` | none (correct) |
| synth Phoenix AZ | `--center/--radius` | 15 m |
| synth Phoenix AZ | disc excluding the truth | none (correct) |
| synth Phoenix AZ, one flyover | ambiguity | truth among candidates |
| **hybrid** Fargo ND (real ADS-B) | `--state ND --between` local | 38 m, +1.1 s |
| **hybrid** Phoenix AZ (real ADS-B) | `--state AZ --between` local | 99 m, +1.5 s |
| **hybrid** Phoenix AZ (real ADS-B) | `--date --center --radius` | 15 m, −0.6 s |
| **hybrid** Grand Forks ND (real ADS-B) | `--date --center --radius` | known failure (see below) |
| **real** Medora ND speech, 2026-07-01 (public domain) | `--state ND --date` | no flyovers detected (correct: podium audio, no aircraft) |
| **real** Iowa City IA field recording, 2023-12-04 (public domain) | `--at --tolerance --center --radius` | blind: correct aircraft `ade930` at #2, start within 14 s; position 4.9 km along-track (one flyover) |

The Grand Forks hybrid case is marked as a known failure (`xfail`). It is the most fragile
busy-pattern scene in the suite, and it broke when misfit residuals became frequency-relative,
a change needed for real jet fan tones.

**Held-out real recordings** (`tests/validation/holdout.toml`, public domain or CC sound-map
field recordings with GPS and time): Rockaway Park NY under JFK arrivals, a Brooklyn street
with helicopter passes, and the Verde River AZ. All three end as "no flyovers detected",
with no false location.

Investigation of those recordings against the ADS-B truth:
* **Rockaway.** Arrivals pass every 60–100 s at 1–3 km, but they are heard as broadband
  noise with ground-reflection interference fringes, not trackable tones.
* **Brooklyn.** The helicopter Doppler sweeps are explained by no ADS-B flight.
* **Verde.** No low traffic within 8 km.

### Real recordings: what was tried and what it shows

**Search effort.** Two sources were screened systematically against historical ADS-B.

* **radio aporee sound map.** All of its ~80k geotagged, time-stamped field recordings were
  filtered to the continental US and ADS-B coverage: 287 recordings on adsb.lol mirror days
  and 85 on the 1st of a month. All 372 were downloaded and run through flyover detection,
  in three frequency bands. About 20 contain at least one tonal flyover.
* **AeroSonicDB YPAD-0523** (Zenodo, CC BY-NC). These are ADS-B-labelled clips recorded under
  Adelaide Airport's approach. On 2023-03-01, nine labelled aircraft passed in 45 min, and
  that day is covered by ADSBExchange's 1st-of-month samples.

**What works on real audio.** At the recordist's GPS pin, a real Doppler curve matches the
ADS-B track of one specific aircraft, while every other aircraft nearby does not fit:

| recording | aircraft | misfit at the pin | other aircraft | implied start vs metadata |
|---|---|---|---|---|
| Iowa City IA, 2023-12-04 | `ade930` | 0.02–0.03 | ≥ 0.5 (cap) | +146 s (metadata time is rounded) |
| Log House Landing MN, 2023-04-26 | `aae678` | 0.016–0.026, in three separate frequency bands | ≥ 0.5 or inconsistent timing | +482–483 s, consistent across bands |

For Iowa City, the blind search (30 km disc, ±30 min) also puts the right aircraft and start
time in its top candidates. This is now a committed validation case.

**What does not work yet: a blind fix from real audio.** No real recording found so far has
two independent flyovers by ADS-B-equipped aircraft that `sya` can both detect and fit:

* **Field recordings** usually catch at most one aircraft. A single aircraft fixes the time
  and the cross-track distance, but leaves the along-track position free; Iowa City comes
  out 4.9 km off for this reason.
* **Dense traffic** (Venice Beach, LA, 2024-04-01, ADSBExchange): many flights fit any one
  curve, but no single location and start time is consistent with all three flyovers.
* **Close-in airport traffic** (AeroSonicDB, about 50–300 m away): sweeps are very steep,
  pilots change engine power, and propeller combs confuse the fundamental. Two jets matched
  their labelled aircraft (E190 cost 0.02, 737 cost 0.01), but with inconsistent timing, and
  the Cessna 172 did not fit at all, likely because of engine power changes.
* **Earlier attempts:** jets on approach heard as broadband noise (Rockaway NY), helicopters
  not in ADS-B (Brooklyn NY), and no audible aircraft (Verde AZ, Medora ND).

**Why it's hard to find better data here.** Free historical ADS-B covers only
2023-03-10 → 2024-01-01, 2025-12-19 → 2026-02-03 (adsb.lol mirror), and the 1st of each month
(ADSBExchange samples). The best-matched public audio falls outside these windows. For
example, the Eclipse Soundscapes project has thousands of AudioMoth recordings across the US,
with exact GPS and time, all on 2024-04-08.

**What would settle it:**
1. **Record a test clip yourself.** Record 15–20 minutes on a phone, under an approach path
   or near a busy general-aviation airport, on the 1st of a month (ADSBExchange publishes
   that day's samples) or on any day you can get ADS-B for. Note the GPS position and the
   start time.
2. **OpenSky research access.** With credentials (`--opensky`, via pyopensky), any date
   becomes searchable. That opens the eclipse dataset and the rest of the sound map.
3. **White House tarmac press gaggles** (public domain, on Wikimedia Commons). Wikimedia was
   rate-limiting downloads throughout this session:
   * Palm Beach International, 2026-01-19, about 18:26 EST (inside mirror coverage);
   * Joint Base Andrews, 2026-07-01, 08:30 EDT;
   * DFW, 2026-10-01, 15:14 CDT.

   The last two need the ADSBExchange source.

## Benchmarks

**Full pipeline** (`sya bench --repeat 2`): synthetic Bismarck, 1-day CONUS window, about
9.7 s in total. Event fitting (2.3 s), polish (1.4 s) and refine (1.0 s) dominate; the
search passes over ADS-B are milliseconds.

**Scaling** (`python benchmarks/bench_scaling.py`): search cost grows linearly with the
ADS-B rows in the window, and refinement is roughly constant.

| | load | resample | join | prescore | total |
|---|---|---|---|---|---|
| 129k rows | 0.04 | 0.07 | 0.00 | 0.03 | 1.1 s |
| 1.9M rows | 0.96 | 1.12 | 0.02 | 0.03 | 4.0 s |

**Real CONUS traffic** (2-hour window, about 4.6M ADS-B rows, 4 workers, time in s summed
over workers):

| load | altitude | resample | join | prescore | polish | refine | wall |
|---|---|---|---|---|---|---|---|
| 6.7 | 1.3 | 8.4 | 16.4 | 56 | 31 | 13 | ~120 s |

A real CONUS day is about 25M rows. With the default 60 s budget, CONUS covers about an hour
of start window; longer windows time out and return the candidate list. In this 2-hour
real-traffic test (Fargo hybrid, CONUS), the search with the current tolerances lost the
truth among about 750k coincidental joins. An earlier, tighter configuration found it at
19 m in 110 s. **Constrain the area** (`--state`, `--center/--radius`) for dense traffic.

## Limitations / next steps
* **Tonal flyovers only.** Jets on approach mostly produce broadband noise plus
  ground-reflection fringes. A fringe (Lloyd's mirror) model would make recordings like
  Rockaway usable.
* **Busy pattern traffic.** Crossing tones bias the straight-pass fits. The Grand Forks
  hybrid case passes offline but failed against live z10 terrain.
* **Propagation.** Straight-line, still-air propagation is assumed. Wind biases `v` and `d`.
* **ADS-B coverage.** It is limited to the mirror's dates. A source for the ADSBExchange
  1st-of-month samples (exact timestamps) is a natural addition.
