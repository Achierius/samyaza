"""End-to-end check on several random synthetic worlds.

python examples/eval_synthetic.py --seeds 0 1 2 3 4 5
"""
import argparse
import time

import numpy as np

from planeloc import audio, locate, synth

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
ap.add_argument("--window", type=float, default=3600, help="T0 search window half-width (s)")
a = ap.parse_args()

rows = []
for seed in a.seeds:
    t = time.time()
    w = synth.make_world(seed=seed)
    tracks = synth.adsb_tracks(w, seed=seed + 1)
    x, fs = synth.render_audio(w, seed=seed + 2)
    evs, _ = audio.extract_events(x, fs, 60, 300, harmonics=3)
    if len(evs) < 2:
        print(f"seed {seed}: only {len(evs)} events detected")
        rows.append((seed, np.nan, np.nan, len(evs), 0))
        continue
    res, _, _ = locate.locate(evs, tracks, w.frame.lat0, w.frame.lon0, 10000, 100,
                              w.T0 - a.window, w.T0 + a.window, w.dur, ground_h=w.ground_h,
                              silence_range_m=1000, verbose=False)
    if not res:
        print(f"seed {seed}: no solution")
        rows.append((seed, np.nan, np.nan, len(evs), 0))
        continue
    r = res[0]
    err = np.hypot(r.e - w.obs_en[0], r.n - w.obs_en[1])
    margin = res[1].cost_b / r.cost_b if len(res) > 1 else np.inf
    print(f"seed {seed}: {len(evs)} events, error {err:6.0f} m, dT0 {r.T0 - w.T0:+6.1f} s, "
          f"flights {r.flights}, runner-up/best {margin:.1f}  ({time.time() - t:.0f}s)")
    rows.append((seed, err, r.T0 - w.T0, len(evs), margin))

errs = np.array([r[1] for r in rows])
print(f"\nsolved {np.sum(errs < 500)}/{len(rows)} within 500 m; median error {np.nanmedian(errs):.0f} m")
