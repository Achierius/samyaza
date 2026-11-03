"""How the search passes scale with traffic volume and time-window length.

    python benchmarks/bench_scaling.py [--json out.json]

Uses synthetic worlds (no network). Traffic is scaled by the number of background
flights over a 200 km box; the window is scaled by searching 1, 3 and 7 days.
"""
import argparse
import json
import time

import numpy as np
from rich.console import Console
from rich.table import Table

from sya import regions
from sya.search import SearchConfig, search
from sya.synth import event_from_truth, make_world, synth_world_data
from sya.timespec import TimeWindow
from sya.timing import PassLog

ap = argparse.ArgumentParser()
ap.add_argument("--json")
a = ap.parse_args()
con = Console(width=220)
results = []
PASSES = ["load", "altitude", "resample", "candidates[0]", "candidates[1]", "region[0]", "join", "silence",
          "prescore", "refine"]

for nbg, days in [(150, 1), (600, 1), (2400, 1), (600, 3), (600, 7)]:
    w = make_world("synth:bismarck:1", n_background=nbg, area_m=200000, pad_s=days * 86400 // 2)
    world = synth_world_data(w)
    evs = [event_from_truth(w, w.flights[0], 165, 240), event_from_truth(w, w.flights[1], 615, 670, seed=1)]
    t0 = np.floor(w.T0 / 86400) * 86400 - 86400 * (days // 2)
    log = PassLog()
    t = time.perf_counter()
    res = search(evs, world, regions.conus(), TimeWindow(t0, t0 + days * 86400, f"{days}d"), w.dur,
                 SearchConfig(timeout_s=3600), log)
    tot = time.perf_counter() - t
    err = regions.haversine_m(res.best.lat, res.best.lon, *w.obs_latlon) if res.best else float("nan")
    p = log.as_dict()
    results.append(dict(background=nbg, days=days, rows=p["load"]["n_out"], total_s=tot, status=res.status,
                        err_m=float(err), passes=p))

tab = Table(title="search scaling (synthetic)")
for c in ["flights", "days", "ADS-B rows", *PASSES, "total s", "status", "err m"]:
    tab.add_column(c, justify="right")
for r in results:
    tab.add_row(str(r["background"]), str(r["days"]), f"{r['rows']:,}",
                *[f"{r['passes'].get(k, {}).get('seconds', 0):.2f}" for k in PASSES],
                f"{r['total_s']:.2f}", r["status"], f"{r['err_m']:.0f}")
con.print(tab)
if a.json:
    json.dump(results, open(a.json, "w"), indent=2)
