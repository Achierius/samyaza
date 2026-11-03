"""Runtime benchmarks: full pipeline per pass, plus scaling of the search passes."""
from __future__ import annotations

import json
import time

import numpy as np
from rich.console import Console
from rich.table import Table

from .timing import PassLog

console = Console()


def _pipeline_once(spec: str, window_days: float = 1.0):
    from . import regions
    from .audio import extract_events
    from .search import SearchConfig, search
    from .synth import make_world, render_audio, synth_world_data
    from .timespec import TimeWindow

    log = PassLog()
    w = make_world(spec)
    with log.timed("render-audio (synthetic input, not part of locate)"):
        x, fs = render_audio(w)
    evs, _ = extract_events(x, fs, 60, 300, log=log)
    world = synth_world_data(w)
    t0 = np.floor(w.T0 / 86400) * 86400
    res = search(evs, world, regions.conus(), TimeWindow(t0, t0 + window_days * 86400, "bench"), w.dur,
                 SearchConfig(timeout_s=600), log)
    err = float(regions.haversine_m(res.best.lat, res.best.lon, *w.obs_latlon)) if res.best else float("nan")
    return log, res.status, err


def run_bench(spec="synth:bismarck:1", repeat=1, out=None):
    runs = []
    for i in range(repeat):
        t = time.perf_counter()
        log, status, err = _pipeline_once(spec)
        runs.append(dict(total=time.perf_counter() - t, status=status, err_m=err, passes=log.as_dict()))
    names = list(runs[0]["passes"])
    tab = Table(title=f"sya bench: {spec}, 1-day CONUS window, {repeat} run(s)")
    tab.add_column("pass")
    tab.add_column("mean s", justify="right")
    tab.add_column("min s", justify="right")
    tab.add_column("share", justify="right")
    tot = np.mean([r["total"] for r in runs])
    for n in names:
        v = [r["passes"][n]["seconds"] for r in runs if n in r["passes"]]
        tab.add_row(n, f"{np.mean(v):.3f}", f"{np.min(v):.3f}", f"{100 * np.mean(v) / tot:.1f}%")
    tab.add_row("[bold]total", f"[bold]{tot:.3f}", f"{min(r['total'] for r in runs):.3f}", "100%")
    console.print(tab)
    console.print(f"status {runs[-1]['status']}, error {runs[-1]['err_m']:.0f} m")
    if out:
        json.dump(dict(spec=spec, runs=runs), open(out, "w"), indent=2)
    return runs
