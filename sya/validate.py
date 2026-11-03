"""End-to-end validation cases: audio file + CLI flags  ->  expected location ± tolerance.

Cases live in a TOML file:

    [[case]]
    name   = "synth-bismarck-single-date"
    audio  = "synth:bismarck:1"            # a file path (relative to the TOML) or a synth spec
    args   = ["--date", "2023-08-15", "--world", "synth:bismarck:1"]
    tags   = ["synthetic", "date"]
    [case.expect]
    status = "solved"                      # solved | ambiguous | none | no-false-positive
    lat = 46.83241                          # expected location (solved / ambiguous)
    lon = -100.75876
    tol_m = 300                            # solved: best within tol;
                                           # ambiguous: truth within tol of one of the top `within_top`
    within_top = 15
    start = "2023-08-15T16:12:32Z"         # optional: expected recording start
    tol_s = 5

Paths inside `args` that start with "@/" are resolved relative to the TOML file.

`audio = "hybrid"` renders audio from REAL (frozen) ADS-B at a given observer:
    [case.hybrid]
    lat = 46.9  ;  lon = -96.8  ;  start = "2023-08-08T18:20:00Z"  ;  seed = 0
    cache_dir = "../data"     # frozen ADS-B + terrain, relative to the TOML
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


def hybrid_audio(h: dict, base: str, cache_dir: str) -> str:
    """Render audio for a [case.hybrid] table: real (frozen) ADS-B, synthetic sound."""
    import soundfile as sf

    from .hybrid import hybrid_world
    from .synth import render_audio
    from .timespec import parse_timestamp
    from .world import make_real_world

    os.makedirs(cache_dir, exist_ok=True)
    key = f"hybrid_{h['lat']:.5f}_{h['lon']:.5f}_{h['start'].replace(':', '')}_{h.get('seed', 0)}"
    path = os.path.join(cache_dir, key + ".flac")
    if not os.path.exists(path):
        world = make_real_world(os.path.join(base, h.get("cache_dir", "../data")), offline=True,
                                terrain=h.get("terrain", "tiles:z8"))
        w = hybrid_world(world, h["lat"], h["lon"], parse_timestamp(h["start"]), h.get("dur", 900.0),
                         seed=h.get("seed", 0))
        x, fs = render_audio(w, seed=h.get("seed", 0) + 2)
        sf.write(path + ".tmp.flac", x, fs)
        os.replace(path + ".tmp.flac", path)
    return path


def url_audio(url: str, cache_dir: str) -> str:
    """Download a remote recording once (holdout cases reference audio we do not redistribute)."""
    import hashlib
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    ext = os.path.splitext(url.split("?")[0])[1] or ".bin"
    path = os.path.join(cache_dir, hashlib.sha1(url.encode()).hexdigest()[:16] + ext)
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "sya-validate/0.2"})
        with urllib.request.urlopen(req, timeout=600) as r, open(path + ".part", "wb") as fh:
            fh.write(r.read())
        os.replace(path + ".part", path)
    return path


def load_cases(path: str):
    with open(path, "rb") as fh:
        return tomllib.load(fh).get("case", [])


def run_case(case: dict, base: str) -> CaseResult:
    from .cli import run

    audio = case["audio"]
    if audio.startswith("url:"):
        audio = url_audio(audio[4:], os.path.join(base, ".audio-cache"))
    elif audio == "hybrid":
        audio = hybrid_audio(case["hybrid"], base, os.path.join(base, ".synth-cache"))
    elif audio.startswith("synth:"):
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
        if "flight" in exp:
            # identify the right aircraft at the right time (a single flyover cannot fix the
            # along-track position, so the location tolerance is loose)
            t_exp = parse_timestamp(exp["start"])
            hits = [i for i, c in enumerate(cands) if exp["flight"] in (c.get("flights") or [])
                    and abs(c["T0"] - t_exp) <= exp.get("tol_s", 30)
                    and haversine_m(c["lat"], c["lon"], exp["lat"], exp["lon"]) <= exp["tol_m"]]
            ok = bool(hits)
            if hits:
                c = cands[hits[0]]
                err = float(haversine_m(c["lat"], c["lon"], exp["lat"], exp["lon"]))
                dt = c["T0"] - t_exp
            detail = (f"{exp['flight']} at #{hits[0] + 1} of {len(out['candidates'])}" if hits
                      else f"{exp['flight']} not among top {len(cands)}")
        else:
            errs = [float(haversine_m(c["lat"], c["lon"], exp["lat"], exp["lon"])) for c in cands]
            err = min(errs) if errs else None
            ok = err is not None and err <= exp["tol_m"]
            detail = f"{len(out['candidates'])} candidates; truth is #{errs.index(err) + 1 if errs else '-'}"
    elif want == "none":
        ok = st == "none"
        detail = f"{len(out.get('candidates', []))} candidates" if "candidates" in out else out.get("reason", "")
    elif want == "no-false-positive":
        # the recording may not be localisable, but sya must not claim a wrong location
        if st == "solved":
            b = out["best"]
            err = float(haversine_m(b["lat"], b["lon"], exp["lat"], exp["lon"]))
            ok = err <= exp.get("tol_m", 1000)
            detail = f"solved at {err:.0f} m from truth"
        else:
            ok = True
            detail = out.get("reason") or f"{st}, {len(out.get('candidates', []))} candidates"
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
