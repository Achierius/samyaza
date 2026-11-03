"""Command line interface: python -m planeloc <command> ..."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np


def _time(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def _window(s: str):
    a, b = s.split(":")
    return float(a), float(b)


# ------------------------------------------------------------------ commands
def cmd_simulate(a):
    import soundfile as sf

    from . import adsb, synth

    os.makedirs(a.out, exist_ok=True)
    w = synth.make_world(seed=a.seed)
    x, fs = synth.render_audio(w)
    sf.write(os.path.join(a.out, "recording.wav"), x.astype(np.float32), fs)
    adsb.save_csv(synth.adsb_tracks(w, seed=a.seed + 1), os.path.join(a.out, "adsb.csv"))
    lat, lon, _ = w.frame.from_enu(w.obs)
    truth = dict(lat=float(lat), lon=float(lon), ground_h=w.ground_h, T0=w.T0, T0_iso=_iso(w.T0), dur=w.dur,
                 center=[w.frame.lat0, w.frame.lon0])
    json.dump(truth, open(os.path.join(a.out, "truth.json"), "w"), indent=2)
    print(json.dumps(truth, indent=2))


def cmd_events(a):
    from .audio import extract_events, load_audio
    from .physics import speed_of_sound

    x, fs = load_audio(a.audio)
    windows = [_window(s) for s in a.window] if a.window else None
    evs, H = extract_events(x, fs, a.fmin, a.fmax, harmonics=a.harmonics, win_s=a.win, hop_s=a.hop,
                            snr_thresh=a.snr, min_len_s=a.min_len, windows=windows,
                            c=speed_of_sound(a.temp), max_slope_hz_s=a.max_slope)
    for i, e in enumerate(evs):
        f = e.fit
        print(f"event {i}: {f.t_start:7.1f}-{f.t_end:7.1f}s  f0={f.f0:7.2f}Hz  v={f.v:6.1f}m/s "
              f"({f.v * 1.944:5.0f}kt)  d={f.d:6.0f}m  t_ca={f.t_ca:7.2f}s  rms={f.rms_hz:.2f}Hz")
    json.dump({"audio": a.audio, "duration": len(x) / fs, "temp_c": a.temp,
               "events": [e.to_dict() for e in evs]}, open(a.out, "w"))
    print(f"wrote {a.out}")
    if a.plot:
        _plot_events(H, evs, a.plot, a.temp)


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
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


def _load_tracks(a, t_lo, t_hi, bbox):
    from . import adsb

    tracks = []
    for p in a.adsb or []:
        tracks += adsb.load_readsb_dir(p, t_lo, t_hi, bbox) if os.path.isdir(p) else adsb.load_csv(p)
    if a.opensky:
        tracks += adsb.fetch_opensky(t_lo, t_hi, bbox)
    return tracks


def _bbox(lat, lon, radius_m):
    dlat = radius_m / 111_000
    dlon = radius_m / (111_000 * np.cos(np.radians(lat)))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


def cmd_locate(a):
    from .audio import Event
    from .locate import locate

    ev = json.load(open(a.events))
    events = [Event.from_dict(d) for d in ev["events"]]
    if a.use:
        events = [events[int(i)] for i in a.use.split(",")]
    if not events:
        raise SystemExit("no events")
    lat, lon = map(float, a.center.split(","))
    t_lo, t_hi = _time(a.t_lo), _time(a.t_hi)
    dur = ev["duration"]
    tracks = _load_tracks(a, t_lo - 600, t_hi + dur + 600, _bbox(lat, lon, a.radius + 30000))
    print(f"[locate] loaded {len(tracks)} ADS-B track segments")
    res, grid, cands = locate(events, tracks, lat, lon, a.radius, a.spacing, t_lo, t_hi, dur,
                              ground_h=a.ground_h, temp_c=a.temp, silence_range_m=a.silence_range,
                              refine_top=a.refine_top)
    out = []
    for i, r in enumerate(res[: a.top]):
        print(f"#{i + 1}: {r.lat:.6f},{r.lon:.6f}  start={_iso(r.T0)}  cost={r.cost_b:.4f}  "
              f"rms={[round(x, 2) for x in r.rms_hz]}Hz  f0={[round(x, 1) for x in r.f0]}  "
              f"flights={r.flights}  silence_hits={r.silence_hits}")
        out.append(dict(lat=r.lat, lon=r.lon, start_unix=r.T0, start_iso=_iso(r.T0), cost=r.cost_b,
                        rms_hz=r.rms_hz, f0=r.f0, flights=r.flights, silence_hits=r.silence_hits))
    if len(res) > 1:
        print(f"margin: runner-up cost / best cost = {res[1].cost_b / max(res[0].cost_b, 1e-12):.2f}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    if a.plot:
        _plot_locate(grid, cands, res, a.plot, lat, lon)


def _plot_locate(grid, cands, res, path, lat0, lon0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(grid.e / 1000, grid.n / 1000, s=0.3, c="0.85")
    if cands:
        ce = np.array([[c.e, c.n, c.cost_a] for c in cands])
        sc = ax.scatter(ce[:, 0] / 1000, ce[:, 1] / 1000, c=ce[:, 2], s=8, cmap="viridis_r", label="coarse")
        fig.colorbar(sc, ax=ax, label="coarse cost")
    for i, r in enumerate(res[:5]):
        ax.plot(r.e / 1000, r.n / 1000, "r*" if i == 0 else "rx", ms=14 if i == 0 else 8)
    ax.set_aspect("equal")
    ax.set_xlabel(f"east of {lat0:.4f},{lon0:.4f} (km)")
    ax.set_ylabel("north (km)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


def cmd_adsb(a):
    from . import adsb

    lat, lon = map(float, a.center.split(","))
    t_lo, t_hi = _time(a.start), _time(a.end)
    tracks = _load_tracks(a, t_lo, t_hi, _bbox(lat, lon, a.radius))
    adsb.save_csv(tracks, a.out)
    print(f"wrote {len(tracks)} track segments to {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="planeloc", description="Geolocate audio from aircraft Doppler curves")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="generate a synthetic recording + ADS-B + ground truth")
    s.add_argument("--out", default="sim")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("events", help="extract and fit flyover Doppler curves from audio/video")
    s.add_argument("audio")
    s.add_argument("--fmin", type=float, default=60)
    s.add_argument("--fmax", type=float, default=300)
    s.add_argument("--harmonics", type=int, default=3)
    s.add_argument("--win", type=float, default=1.0, help="STFT window (s)")
    s.add_argument("--hop", type=float, default=0.1, help="STFT hop (s)")
    s.add_argument("--snr", type=float, default=3.0)
    s.add_argument("--min-len", type=float, default=20.0)
    s.add_argument("--max-slope", type=float, default=12.0, help="max tone slope Hz/s")
    s.add_argument("--window", action="append", help="manual event window start:end (s), repeatable")
    s.add_argument("--temp", type=float, default=15.0, help="air temperature °C")
    s.add_argument("--out", default="events.json")
    s.add_argument("--plot")
    s.set_defaults(fn=cmd_events)

    def add_src(s):
        s.add_argument("--adsb", action="append", help="CSV file or extracted readsb globe_history dir")
        s.add_argument("--opensky", action="store_true", help="query OpenSky Trino (pyopensky)")

    s = sub.add_parser("locate", help="search for recording location + start time")
    s.add_argument("events")
    add_src(s)
    s.add_argument("--center", required=True, help="lat,lon of search area centre")
    s.add_argument("--radius", type=float, default=10000)
    s.add_argument("--spacing", type=float, default=100)
    s.add_argument("--t-lo", required=True, help="earliest recording start (unix or ISO8601)")
    s.add_argument("--t-hi", required=True, help="latest recording start")
    s.add_argument("--ground-h", type=float, default=0.0, help="ground ellipsoid height (m)")
    s.add_argument("--temp", type=float, default=15.0)
    s.add_argument("--silence-range", type=float, help="penalise unheard flights closer than this (m)")
    s.add_argument("--use", help="comma-separated event indices to use")
    s.add_argument("--refine-top", type=int, default=60)
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--out")
    s.add_argument("--plot")
    s.set_defaults(fn=cmd_locate)

    s = sub.add_parser("adsb", help="extract ADS-B for an area/time into CSV")
    add_src(s)
    s.add_argument("--center", required=True)
    s.add_argument("--radius", type=float, default=40000)
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--out", default="adsb.csv")
    s.set_defaults(fn=cmd_adsb)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
