"""`sya`: locate an audio recording in the continental US from aircraft Doppler curves."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import regions
from .timespec import TimeWindow, at_window, between_window, dates_window, parse_when, day_window
from .timing import PassLog

console = Console(stderr=True)
out_console = Console()

PASS_ORDER = ["load", "altitude", "resample", "terrain", "candidates", "region", "join", "dedupe", "silence",
              "prescore", "refine"]


# ------------------------------------------------------------------ argument helpers
def add_time_args(p):
    g = p.add_argument_group("time (when the recording STARTED; pick one)")
    g.add_argument("--date", help="a single day, e.g. 2023-08-08 (interpreted in --tz)")
    g.add_argument("--dates", nargs=2, metavar=("FIRST", "LAST"), help="inclusive range of days")
    g.add_argument("--between", nargs=2, metavar=("START", "END"), help="timestamp range (ISO8601 or unix)")
    g.add_argument("--at", help="a timestamp; combine with --tolerance (default 15m)")
    g.add_argument("--tolerance", default="15m", help="± for --at (e.g. 90s, 15m, 2h)")
    g.add_argument("--when", help="free form: DATE | DATE..DATE | TS..TS")
    g.add_argument("--tz", default="UTC", help="time zone for dates/naive timestamps, e.g. America/Phoenix")


def time_window(a) -> TimeWindow:
    given = [x for x in ("date", "dates", "between", "at", "when") if getattr(a, x)]
    if len(given) != 1:
        raise SystemExit("error: give exactly one of --date, --dates, --between, --at, --when")
    if a.date:
        lo, hi = day_window(a.date, a.tz)
        return TimeWindow(lo, hi, f"date {a.date} ({a.tz})")
    if a.dates:
        return dates_window(*a.dates, a.tz)
    if a.between:
        return between_window(*a.between, a.tz)
    if a.at:
        return at_window(a.at, a.tolerance, a.tz)
    return parse_when(a.when, a.tz)


def add_area_args(p):
    g = p.add_argument_group("area (default: continental US)")
    g.add_argument("--center", help="LAT,LON; requires --radius")
    g.add_argument("--radius", help="radius around --center, e.g. 25km, 10mi, 5000 (m)")
    g.add_argument("--state", help="restrict to one CONUS state (code or name)")


def parse_radius(s: str) -> float:
    s = s.strip().lower()
    for suf, mul in (("km", 1000.0), ("mi", 1609.344), ("nm", 1852.0), ("m", 1.0)):
        if s.endswith(suf):
            return float(s[: -len(suf)]) * mul
    return float(s)


def area(a) -> regions.Region:
    if (a.center is None) != (a.radius is None):
        raise SystemExit("error: --center and --radius must be given together")
    if a.center and a.state:
        raise SystemExit("error: use either --center/--radius or --state, not both")
    if a.center:
        lat, lon = map(float, a.center.split(","))
        try:
            return regions.radius(lat, lon, parse_radius(a.radius))
        except ValueError as e:
            raise SystemExit(f"error: {e}")
    if a.state:
        try:
            return regions.state(a.state)
        except ValueError as e:
            raise SystemExit(f"error: {e}")
    return regions.conus()


def add_audio_args(p):
    g = p.add_argument_group("audio analysis")
    g.add_argument("--fmin", type=float, default=60)
    g.add_argument("--fmax", type=float, default=300)
    g.add_argument("--harmonics", type=int, default=3)
    g.add_argument("--snr", type=float, default=3.0, help="ridge SNR threshold (z units)")
    g.add_argument("--min-len", type=float, default=20.0, help="minimum flyover length (s)")
    g.add_argument("--max-slope", type=float, default=12.0, help="max tone slope (Hz/s)")
    g.add_argument("--window", action="append", help="manual flyover window START:END seconds (repeatable)")
    g.add_argument("--clip", help="analyse only START:END seconds of the file")
    g.add_argument("--temp", type=float, default=15.0, help="air temperature °C")


def add_world_args(p):
    g = p.add_argument_group("world data")
    g.add_argument("--world", default="real",
                   help="real (historical ADS-B, cached) | synth:<preset>[:seed] | csv:<file> | readsb:<dir>")
    g.add_argument("--cache-dir", help="cache root (default $SYA_CACHE_DIR or ~/.cache/sya)")
    g.add_argument("--offline", action="store_true", help="never download; use only cached data")
    g.add_argument("--terrain", default="tiles", help="tiles | flat:<metres>")


def make_world(a):
    from .terrain import FlatTerrain
    from .world import TableWorld, make_real_world

    w = a.world
    if w == "real":
        return make_real_world(a.cache_dir, offline=a.offline, terrain=a.terrain)
    if w.startswith("synth:"):
        from .synth import make_world as mk, synth_world_data

        return synth_world_data(mk(w))
    ter = FlatTerrain(float(a.terrain[5:])) if a.terrain.startswith("flat:") else make_real_world(
        a.cache_dir, offline=a.offline).terrain
    if w.startswith("csv:"):
        from .adsb import load_csv
        from .tracks import TrackTable

        return TableWorld(w, TrackTable.from_tracks(load_csv(w[4:])), ter)
    if w.startswith("readsb:"):
        from .adsb import load_readsb_dir
        from .tracks import TrackTable

        return TableWorld(w, TrackTable.from_tracks(load_readsb_dir(w[7:])), ter)
    raise SystemExit(f"error: unknown --world {w!r}")


def _window_arg(s):
    a, b = s.split(":")
    return float(a), float(b)


def analyse_audio(a, log: PassLog):
    from .audio import extract_events, load_audio
    from .physics import speed_of_sound

    with log.timed("decode") as h:
        x, fs = load_audio(a.audio)
        h["out"] = len(x)
    if a.clip:
        c0, c1 = _window_arg(a.clip)
        x = x[int(c0 * fs): int(c1 * fs)]
    with log.timed("events", n_in=1) as h:
        evs, H = extract_events(x, fs, a.fmin, a.fmax, harmonics=a.harmonics, snr_thresh=a.snr,
                                min_len_s=a.min_len, windows=[_window_arg(w) for w in a.window] if a.window else None,
                                c=speed_of_sound(a.temp), max_slope_hz_s=a.max_slope)
        h["out"] = len(evs)
    return evs, H, len(x) / fs


def events_table(evs):
    t = Table(title="Detected flyovers", show_lines=False)
    for col in ("#", "window (s)", "f0 Hz", "speed m/s (kt)", "CPA slant m", "t_ca s", "rms Hz"):
        t.add_column(col, justify="right")
    for i, e in enumerate(evs):
        f = e.fit
        t.add_row(str(i), f"{f.t_start:.0f}–{f.t_end:.0f}", f"{f.f0:.1f}", f"{f.v:.0f} ({f.v * 1.944:.0f})",
                  f"{f.d:.0f}", f"{f.t_ca:.1f}", f"{f.rms_hz:.2f}")
    return t


# ------------------------------------------------------------------------- Rich UI
class RichUI:
    def __init__(self, log: PassLog, enabled=True):
        self.log = log
        self.enabled = enabled
        self.progress = Progress(TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(),
                                 TextColumn("{task.fields[info]}"), TimeElapsedColumn(), console=console)
        self.dl = None
        self.chunks = None
        self.live = None

    def table(self):
        t = Table(title="Pruning passes", expand=False)
        for col, j in (("pass", "left"), ("in", "right"), ("out", "right"), ("kept", "right"), ("time", "right"),
                       ("", "left")):
            t.add_column(col, justify=j)
        names = sorted(self.log.passes, key=lambda n: (PASS_ORDER.index(n.split("[")[0]) if n.split("[")[0] in PASS_ORDER
                                                       else -1, n))
        for n in names:
            s = self.log.passes[n]
            kept = (f"{100 * s.n_out / s.n_in:.3g}%" if s.n_in and s.n_out is not None and s.n_out <= s.n_in
                    else "")
            t.add_row(n, f"{s.n_in:,}" if s.n_in is not None else "", f"{s.n_out:,}" if s.n_out is not None else "",
                      kept, f"{s.seconds:.2f}s", s.note)
        return t

    def render(self):
        return Group(self.progress, self.table())

    def __enter__(self):
        if self.enabled:
            self.live = Live(self.render(), console=console, refresh_per_second=4, transient=False)
            self.live.__enter__()
            self.log.listeners.append(lambda st: self.live.update(self.render()))
        return self

    def __exit__(self, *exc):
        if self.live:
            self.live.update(self.render())
            self.live.__exit__(*exc)

    def download(self, n, total):
        if not self.enabled:
            return
        if self.dl is None:
            self.dl = self.progress.add_task("ADS-B shards", total=total, info="")
        self.progress.advance(self.dl, n)
        task = next(t for t in self.progress.tasks if t.id == self.dl)
        if task.completed >= total:
            self.progress.remove_task(self.dl)
            self.dl = None

    def start_chunks(self, n):
        if self.enabled:
            self.chunks = self.progress.add_task("time chunks", total=n, info="")

    def chunk_done(self, i, a, b, nsol, log):
        if self.enabled and self.chunks is not None:
            from datetime import datetime, timezone

            ts = datetime.fromtimestamp(b, timezone.utc).strftime("%Y-%m-%d %H:%MZ")
            self.progress.update(self.chunks, advance=1, info=f"through {ts} · {nsol} solutions")
            if self.live:
                self.live.update(self.render())


def solutions_table(sols, title):
    t = Table(title=title)
    for col in ("#", "lat", "lon", "state", "start (UTC)", "cost", "rms Hz", "flights", "unheard"):
        t.add_column(col)
    for i, s in enumerate(sols):
        d = s.to_dict()
        t.add_row(str(i + 1), f"{s.lat:.5f}", f"{s.lon:.5f}", s.state or "", d["start_iso"], f"{s.cost:.3f}",
                  ", ".join("–" if not np.isfinite(r) else f"{r:.2f}" for r in s.rms_hz),
                  ", ".join(f or "–" for f in s.flights), str(s.silence_hits))
    return t


# ------------------------------------------------------------------------ commands
def cmd_locate(a):
    from .search import SearchConfig, search

    log = PassLog()
    win = time_window(a)
    reg = area(a)
    world = make_world(a)
    cfg = SearchConfig(temp_c=a.temp, timeout_s=a.timeout, chunk_s=a.chunk)
    if a.min_events:
        cfg.min_events = a.min_events
    t0 = time.perf_counter()
    if a.events_json:
        from .audio import Event

        d = json.load(open(a.events_json))
        evs = [Event.from_dict(e) for e in d["events"]]
        dur = d["duration"]
    else:
        evs, _, dur = analyse_audio(a, log)
    if not a.quiet:
        console.print(events_table(evs))
        lo, hi = win.iso()
        console.print(f"[bold]search area[/]: {reg.label}   [bold]start window[/]: {lo} → {hi} ({win.label})   "
                      f"[bold]world[/]: {world.name}   [bold]timeout[/]: {cfg.timeout_s:g}s")
    if a.use:
        evs = [evs[int(i)] for i in a.use.split(",")]
    if not evs:
        res = None
        console.print("[red]no aircraft flyovers detected in the audio[/]")
        out = dict(status="none", reason="no flyovers detected", passes=log.as_dict())
    else:
        with RichUI(log, enabled=not a.quiet) as ui:
            res = search(evs, world, reg, win, dur, cfg, log, ui)
        out = res.to_dict()
        if res.status == "solved":
            b = res.best
            out_console.print(f"[bold green]LOCATED[/] {b.lat:.5f}, {b.lon:.5f}"
                              f"{' (' + b.state + ')' if b.state else ''}  start {b.to_dict()['start_iso']}  "
                              f"cost {b.cost:.3f}  flights {', '.join(f or '–' for f in b.flights)}")
        elif res.candidates:
            out_console.print(f"[bold yellow]AMBIGUOUS[/]: could not isolate one solution"
                              f"{' before the timeout' if res.timed_out else ''}; "
                              f"{len(res.candidates)} candidates remain")
            out_console.print(solutions_table(res.candidates[: a.top], "Remaining candidates (best first)"))
        else:
            out_console.print("[bold red]NO SOLUTION[/] in the given area and time window")
        for n in res.notes:
            console.print(f"[dim]note: {n}[/]")
    out["events"] = [e.fit.to_dict() for e in evs]
    out["elapsed_s"] = time.perf_counter() - t0
    out["passes"] = log.as_dict()
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2, default=float)
    if a.print_json:
        print(json.dumps(out, indent=2, default=float))
    return out


def cmd_events(a):
    log = PassLog()
    evs, H, dur = analyse_audio(a, log)
    out_console.print(events_table(evs))
    if a.out:
        json.dump({"audio": a.audio, "duration": dur, "temp_c": a.temp, "events": [e.to_dict() for e in evs]},
                  open(a.out, "w"))
        console.print(f"wrote {a.out}")
    if a.plot:
        _plot_events(H, evs, a.plot, a.temp)
        console.print(f"wrote {a.plot}")


def _plot_events(H, evs, path, temp):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .physics import speed_of_sound, straight_pass

    c = speed_of_sound(temp)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.pcolormesh(H.t, H.f, H.z.T, vmin=0, vmax=np.percentile(H.z, 99.5), shading="auto", cmap="magma")
    for i, e in enumerate(evs):
        ax.plot(e.t, e.f, ".", ms=1.5, color="cyan")
        tt = np.linspace(e.t[0], e.t[-1], 400)
        f = e.fit
        ax.plot(tt, straight_pass(tt, f.f0, f.v, f.d, f.t_ca, c), "w--", lw=1)
        ax.text(f.t_ca, f.f0, f" {i}", color="w")
    ax.set_xlabel("time in recording (s)")
    ax.set_ylabel("fundamental (Hz)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)


def cmd_simulate(a):
    import soundfile as sf

    from .synth import make_world, render_audio

    w = make_world(a.spec)
    x, fs = render_audio(w)
    os.makedirs(a.out, exist_ok=True)
    sf.write(os.path.join(a.out, "recording.flac"), x, fs)
    lat, lon = w.obs_latlon
    from datetime import datetime, timezone

    truth = dict(spec=a.spec, lat=lat, lon=lon, ground_h=w.ground_h, T0=w.T0,
                 T0_iso=datetime.fromtimestamp(w.T0, timezone.utc).isoformat(), dur=w.dur,
                 state=regions.which_state(lat, lon))
    json.dump(truth, open(os.path.join(a.out, "truth.json"), "w"), indent=2)
    out_console.print_json(data=truth)


def cmd_cache(a):
    from datetime import date

    from .world import AdsbCache, dir_size, parse_size

    cache = AdsbCache(a.cache_dir, offline=False)
    if a.action == "ls":
        days = cache.cached_days()
        t = Table(title=f"ADS-B cache at {cache.dir} (limit {cache.limit / 1e9:.1f} GB)")
        for c in ("source", "day", "hours", "size", "complete", "last used"):
            t.add_column(c)
        for d in days:
            t.add_row(d["source"], d["day"], str(len(d["hours"])), f"{d['bytes'] / 1e6:.0f} MB", str(d["complete"]),
                      time.strftime("%Y-%m-%d %H:%M", time.localtime(d["last_used"])))
        out_console.print(t)
        out_console.print(f"total {sum(d['bytes'] for d in days) / 1e9:.2f} GB")
    elif a.action == "fetch":
        win = time_window(a)
        log = PassLog()
        with RichUI(log) as ui:
            from .world import _days

            for d in _days(win.t_lo, win.t_hi):
                p = cache.ensure_day(d, progress=ui.download)
                console.print(f"{d}: {'cached at ' + p if p else 'not available from any source'}")
    elif a.action == "prune":
        removed = cache.prune(limit=parse_size(a.limit) if a.limit else None)
        out_console.print(f"removed {len(removed)} day(s)")
    elif a.action == "clear":
        removed = cache.prune(limit=0)
        out_console.print(f"removed {len(removed)} day(s)")
    elif a.action == "freeze":
        win = time_window(a)
        reg = area(a)
        n = cache.freeze(a.out, win.t_lo, win.t_hi, reg.padded_bbox(10000))
        out_console.print(f"froze {n:,} rows into {a.out}")


def cmd_validate(a):
    from .validate import run_cases

    ok = run_cases(a.cases, only=a.only, tags=a.tag, verbose=not a.quiet)
    sys.exit(0 if ok else 1)


def cmd_bench(a):
    from .bench import run_bench

    run_bench(a.spec, repeat=a.repeat, out=a.json)


def build_parser():
    p = argparse.ArgumentParser(prog="sya", description="Locate audio recordings in the continental US from the "
                                "Doppler curves of passing aircraft, matched against historical ADS-B.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("locate", help="find where/when a recording was made")
    s.add_argument("audio", nargs="?", help="audio or video file")
    s.add_argument("--events-json", help="use flyovers from `sya events --out` instead of analysing audio")
    add_time_args(s)
    add_area_args(s)
    add_audio_args(s)
    add_world_args(s)
    s.add_argument("--timeout", type=float, default=60.0, help="search time budget in seconds (default 60)")
    s.add_argument("--chunk", type=float, default=3600.0, help="time-chunk size for the search (s)")
    s.add_argument("--min-events", type=int, help="flyovers that must match ADS-B (default 2)")
    s.add_argument("--use", help="comma-separated flyover indices to use")
    s.add_argument("--top", type=int, default=15, help="candidates to list when ambiguous")
    s.add_argument("--json", help="write full result JSON here")
    s.add_argument("--print-json", action="store_true")
    s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(fn=cmd_locate)

    s = sub.add_parser("events", help="detect and fit flyovers in audio")
    s.add_argument("audio")
    add_audio_args(s)
    s.add_argument("--out")
    s.add_argument("--plot")
    s.set_defaults(fn=cmd_events)

    s = sub.add_parser("simulate", help="write a synthetic recording + ground truth")
    s.add_argument("spec", help="synth:<preset>[:seed]  (presets: bismarck, fargo, phoenix, flagstaff, seattle)")
    s.add_argument("--out", default="sim")
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("cache", help="manage the ADS-B range cache")
    s.add_argument("action", choices=["ls", "fetch", "prune", "clear", "freeze"])
    s.add_argument("--cache-dir")
    s.add_argument("--limit", help="size limit for prune, e.g. 5G")
    s.add_argument("--out", default="tests/data", help="freeze destination")
    add_time_args(s)
    add_area_args(s)
    s.set_defaults(fn=cmd_cache)

    s = sub.add_parser("validate", help="run end-to-end validation cases")
    s.add_argument("cases", nargs="?", default="tests/validation/cases.toml")
    s.add_argument("--only", help="run only this case name")
    s.add_argument("--tag", action="append", help="run cases with this tag (repeatable)")
    s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("bench", help="per-pass runtime benchmark on a synthetic world")
    s.add_argument("--spec", default="synth:bismarck:1")
    s.add_argument("--repeat", type=int, default=1)
    s.add_argument("--json")
    s.set_defaults(fn=cmd_bench)
    return p


def run(argv=None):
    """Programmatic entry point; returns the command's result (e.g. locate's JSON dict)."""
    a = build_parser().parse_args(argv)
    if a.cmd == "locate" and not a.audio and not a.events_json:
        raise SystemExit("error: give an audio file or --events-json")
    return a.fn(a)


def main(argv=None):
    run(argv)


if __name__ == "__main__":
    main()
