"""End-to-end validation cases: audio file + CLI flags  ->  expected location ± tolerance.

Cases live in a TOML file:

    [[case]]
    name   = "synth-bismarck-single-date"
    audio  = "synth:bismarck:1"            # a file path (relative to the TOML) or a synth spec
    args   = ["--date", "2023-08-15", "--world", "synth:bismarck:1"]
    tags   = ["synthetic", "date"]
    [case.expect]
    status = "solved"                      # solved | ambiguous | none
    lat = 46.83241                          # expected location (solved / ambiguous)
    lon = -100.75876
    tol_m = 300                            # solved: best within tol;
                                           # ambiguous: truth within tol of one of the top `within_top`
    within_top = 15
    start = "2023-08-15T16:12:32Z"         # optional: expected recording start
    tol_s = 5

Paths inside `args` that start with "@/" are resolved relative to the TOML file.
"""
from __future__ import annotations

import os
import time
import tomllib
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from .regions import haversine_m
from .timespec import parse_timestamp

console = Console(stderr=True)


@dataclass
class CaseResult:
    name: str
    ok: bool
    status: str
    err_m: float | None
    dt_s: float | None
    seconds: float
    detail: str


def synth_audio(spec: str, cache_dir: str) -> str:
    import soundfile as sf

    from .synth import make_world, render_audio

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, spec.replace(":", "_") + ".flac")
    if not os.path.exists(path):
        x, fs = render_audio(make_world(spec))
        sf.write(path + ".tmp.flac", x, fs)
        os.replace(path + ".tmp.flac", path)
    return path


def load_cases(path: str):
    with open(path, "rb") as fh:
        return tomllib.load(fh).get("case", [])


def run_case(case: dict, base: str) -> CaseResult:
    from .cli import run

    audio = case["audio"]
    if audio.startswith("synth:"):
        audio = synth_audio(audio, os.path.join(base, ".synth-cache"))
    elif not os.path.isabs(audio):
        audio = os.path.join(base, audio)
    args = [a.replace("@/", base + os.sep) if isinstance(a, str) else str(a) for a in case.get("args", [])]
    t = time.perf_counter()
    out = run(["locate", audio, "--quiet", *args])
    secs = time.perf_counter() - t
    exp = case["expect"]
    st = out.get("status")
    want = exp.get("status", "solved")
    err = dt = None
    ok = st == want
    detail = ""
    if want == "solved" and st == "solved":
        b = out["best"]
        err = float(haversine_m(b["lat"], b["lon"], exp["lat"], exp["lon"]))
        ok = err <= exp["tol_m"]
        if "start" in exp:
            dt = b["T0"] - parse_timestamp(exp["start"])
            ok &= abs(dt) <= exp.get("tol_s", 10)
        detail = f"best {b['lat']:.5f},{b['lon']:.5f}"
    elif want == "ambiguous" and st in ("ambiguous", "solved"):
        cands = out["candidates"][: exp.get("within_top", 15)]
        errs = [float(haversine_m(c["lat"], c["lon"], exp["lat"], exp["lon"])) for c in cands]
        err = min(errs) if errs else None
        ok = err is not None and err <= exp["tol_m"]
        detail = f"{len(out['candidates'])} candidates; truth is #{errs.index(err) + 1 if errs else '-'}"
    elif want == "none":
        ok = st == "none"
        detail = f"{len(out.get('candidates', []))} candidates"
    else:
        detail = f"got status {st}"
    return CaseResult(case["name"], bool(ok), st, err, dt, secs, detail)


def run_cases(path: str, only=None, tags=None, verbose=True) -> bool:
    base = os.path.dirname(os.path.abspath(path))
    cases = load_cases(path)
    if only:
        cases = [c for c in cases if c["name"] == only]
    if tags:
        cases = [c for c in cases if set(tags) & set(c.get("tags", []))]
    results = []
    for c in cases:
        if verbose:
            console.print(f"[bold]case[/] {c['name']} …")
        results.append(run_case(c, base))
    t = Table(title=f"Validation: {os.path.basename(path)}")
    for col in ("case", "pass", "status", "error m", "Δstart s", "time s", "detail"):
        t.add_column(col)
    for r in results:
        t.add_row(r.name, "[green]PASS[/]" if r.ok else "[red]FAIL[/]", r.status,
                  "" if r.err_m is None else f"{r.err_m:.0f}", "" if r.dt_s is None else f"{r.dt_s:+.1f}",
                  f"{r.seconds:.1f}", r.detail)
    Console().print(t)
    return all(r.ok for r in results)
