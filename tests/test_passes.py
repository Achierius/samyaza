"""Each search pass in isolation, on a synthetic ND world with flyovers built from truth."""
import time

import numpy as np
import pytest

from sya import regions, search
from sya.search import SearchConfig, Solution
from sya.timespec import TimeWindow
from sya.timing import PassLog
from sya.tracks import TrackTable, resample


@pytest.fixture(scope="module")
def setup(bismarck):
    w, world, evs = bismarck
    cfg = SearchConfig()
    t_lo, t_hi = w.T0 - 1800, w.T0 + w.dur + 1800
    rs = resample(world.tracks(t_lo, t_hi), t_lo, t_hi)
    elev = world.elevation(rs.lat, rs.lon)
    return w, world, evs, cfg, rs, elev


def _err(w, lat, lon):
    return regions.haversine_m(lat, lon, *w.obs_latlon)


def test_candidates_contain_truth(setup):
    w, world, evs, cfg, rs, elev = setup
    for k, ev in enumerate(evs):
        C = search.event_candidates(ev, rs, elev, regions.conus(), w.T0 - 1800, w.T0 + 1800, w.c, cfg, k=k)
        true_seg = [i for i, ic in enumerate(rs.icao) if ic == w.flights[k].icao]
        on = np.isin(C.seg, true_seg)
        assert on.any()
        e = _err(w, C.lat[on], C.lon[on])
        i = np.argmin(e + 50 * np.abs(C.T0[on] - w.T0))
        assert e[i] < 400 and abs(C.T0[on][i] - w.T0) < 4
        assert len(C.T0) < 0.2 * rs.t.size  # speed/height filters prune most samples


def test_region_pass_drops_outside(setup):
    w, world, evs, cfg, rs, elev = setup
    far = regions.radius(46.0, -100.0, 5000)
    C = search.event_candidates(evs[0], rs, elev, far, w.T0 - 1800, w.T0 + 1800, w.c, cfg)
    assert len(C.T0) == 0
    az = regions.state("AZ")
    assert len(search.event_candidates(evs[0], rs, elev, az, w.T0 - 1800, w.T0 + 1800, w.c, cfg).T0) == 0


def test_join_keeps_consistent_pair_only(setup):
    w, world, evs, cfg, rs, elev = setup
    log = PassLog()
    cands = [search.event_candidates(ev, rs, elev, regions.conus(), w.T0 - 1800, w.T0 + 1800, w.c, cfg, log, k)
             for k, ev in enumerate(evs)]
    J = search.join_events(cands, evs, cfg, log)
    n_in = sum(len(c.T0) for c in cands)
    assert 0 < len(J.T0) < 0.05 * n_in
    best = np.argmin(J.cost)
    assert _err(w, J.lat[best], J.lon[best]) < 500 and abs(J.T0[best] - w.T0) < 4
    assert [rs.icao[s] for s in J.segs[best]] == [f.icao for f in w.flights[:2]]
    # shift the second event's T0 by 5 minutes: nothing should join any more
    c2 = cands[1]
    shifted = search.EventCands(c2.seg, c2.te, c2.xyz, c2.lat, c2.lon, c2.T0 + 300, c2.cost, c2.sigma)
    assert len(search.join_events([cands[0], shifted], evs, cfg, PassLog()).T0) == 0


def test_dedupe_keeps_best(setup):
    w, world, evs, cfg, rs, elev = setup
    log = PassLog()
    cands = [search.event_candidates(ev, rs, elev, regions.conus(), w.T0 - 1800, w.T0 + 1800, w.c, cfg, log, k)
             for k, ev in enumerate(evs)]
    J = search.join_events(cands, evs, cfg, log)
    D = search.dedupe(J, cfg, log)
    assert len(D.T0) <= len(J.T0) and D.cost[0] == J.cost.min()


def test_silence_flags_unheard_close_flight(setup):
    w, world, evs, cfg, rs, elev = setup
    J = search.Joint(np.array([w.frame.origin + w.obs @ w.frame.R]), np.array([w.obs_latlon[0]]),
                     np.array([w.obs_latlon[1]]), np.array([w.T0]), np.array([[-1, -1]]), np.array([0.0]))
    h0 = search.silence(J, rs, evs, w.dur, w.c, cfg, PassLog(), world.elevation)
    # inject an unheard aircraft passing 300 m overhead mid-recording
    lat0, lon0 = w.obs_latlon
    t = w.T0 + 300 + np.arange(-60, 60, 2.0)
    lat = lat0 + (t - t[0] - 60) * 60 / 111_000
    extra = TrackTable.from_columns(["zz9999"] * t.size, t, lat, np.full(t.size, lon0), np.full(t.size, w.ground_h + 300))
    rs2 = resample(extra, t[0], t[-1])
    h1 = search.silence(J, rs2, evs, w.dur, w.c, cfg, PassLog(), world.elevation)
    assert h1[0] == h0[0] + 1


def test_prescore_and_refine_converge(setup):
    w, world, evs, cfg, rs, elev = setup
    log = PassLog()
    cands = [search.event_candidates(ev, rs, elev, regions.conus(), w.T0 - 1800, w.T0 + 1800, w.c, cfg, log, k)
             for k, ev in enumerate(evs)]
    J = search.dedupe(search.join_events(cands, evs, cfg, log), cfg, log)
    hits = np.zeros(len(J.T0), int)
    dl = time.monotonic() + 60
    pre = search.prescore(J, hits, evs, rs, w.c, world.elevation, cfg, log, dl)
    sols = search.refine(pre, evs, w.c, cfg, log, dl)
    best = min(sols, key=lambda s: s.cost)
    assert _err(w, best.lat, best.lon) < 60 and abs(best.T0 - w.T0) < 1.0
    assert max(best.rms_hz) < 0.6


def test_decide_logic():
    cfg = SearchConfig()
    mk = lambda lat, cost, T0=0.0: Solution(lat, -100.0, T0, cost, [0.3, 0.3], [100, 100], ["a", "b"])
    assert search.decide([], cfg)[0] == "none"
    assert search.decide([mk(46.8, 0.1)], cfg)[0] == "solved"
    assert search.decide([mk(46.8, 0.1), mk(46.8005, 0.12)], cfg)[0] == "solved"   # same place
    assert search.decide([mk(46.8, 0.1), mk(47.2, 0.12)], cfg)[0] == "ambiguous"   # rival elsewhere
    assert search.decide([mk(46.8, 0.1), mk(47.2, 2.0)], cfg)[0] == "solved"       # rival much worse
    one = Solution(46.8, -100, 0, 0.1, [0.3, float("nan")], [100, 100], ["a", None])
    assert search.decide([one], cfg)[0] == "ambiguous"  # a single explained flyover is not enough
    comb = Solution(46.8, -100, 0, 0.1, [0.3, 0.3], [100, 150], ["a", "a"])
    from sya.audio import Event, PassFit
    evs = [Event(np.zeros(1), np.zeros(1), np.ones(1), PassFit(100, 50, 500, 140.0, 0.2, 10)),
           Event(np.zeros(1), np.zeros(1), np.ones(1), PassFit(150, 50, 500, 141.0, 0.2, 10))]
    assert search.decide([comb], cfg, evs)[0] == "ambiguous"  # two lines of one pass = one constraint
    bad = Solution(46.8, -100, 0, 5.0, [4.0, 5.0], [100, 100], ["a", "b"])
    assert search.decide([bad], cfg)[0] == "ambiguous"  # poor fit is never "solved"


def test_full_search_timeout_returns_candidates(bismarck):
    w, world, evs = bismarck
    cfg = SearchConfig(timeout_s=0.0)
    res = search.search(evs, world, regions.conus(), TimeWindow(w.T0 - 86400, w.T0 + 86400, "2d"), w.dur, cfg)
    assert res.timed_out and res.searched_s < res.window_s


def test_full_search_single_event_is_ambiguous(bismarck):
    w, world, evs = bismarck
    res = search.search(evs[:1], world, regions.conus(), TimeWindow(w.T0 - 3600, w.T0 + 3600, "2h"), w.dur,
                        SearchConfig(timeout_s=30))
    assert res.status == "ambiguous" and len(res.candidates) > 1
    errs = [_err(w, c.lat, c.lon) for c in res.candidates]
    assert min(errs) < 1500


def test_quiet_map_marks_tones_bursts_and_events():
    from sya.audio import quiet_map
    from sya.physics import speed_of_sound, straight_pass

    fs = 2000
    t = np.arange(0, 120, 1 / fs)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, t.size)
    tone = (t > 20) & (t < 40)
    x[tone] += 0.5 * np.sin(2 * np.pi * 150 * t[tone])  # tonal stretch, not a detected event
    burst = (t > 70) & (t < 80)
    x[burst] *= 6.0  # broadband masking (~15 dB)
    q = quiet_map(x, fs, 60, 300, 1, [], snr_thresh=3.0)
    assert not q.is_quiet(25, 35)       # a tone was present: not "quiet"
    assert not q.is_quiet(72, 78)       # masked
    assert q.is_quiet(50, 60) and q.is_quiet(95, 110)
    assert 0.4 < q.fraction < 0.9


def test_unheard_aircraft_penalises_close_silent_pass(setup):
    w, world, evs, cfg, rs, elev = setup
    from sya.audio import QuietMap
    from sya.geo import LocalFrame

    lat, lon = w.obs_latlon
    frame = LocalFrame(lat, lon, w.ground_h)
    obs = frame.ground_enu(0.0, 0.0, w.ground_h)
    # a slow aircraft passing 300 m overhead, 400 s into the recording (the heard ones are ~1.1-1.5 km)
    t = w.T0 + 400 + np.arange(-120, 120, 1.0)
    enu = np.column_stack([(t - t[120]) * 50.0, np.zeros_like(t), np.full_like(t, 300.0)]) + obs * [0, 0, 1]
    lc = search.LocalCand(lat, lon, w.T0, w.ground_h, frame, [None, None], ["a", "b"], 0, 0.0,
                          nearby=[("zz9999", t, enu)])
    rms = [0.2, 0.2]
    all_quiet = QuietMap(np.arange(900.0), np.ones(900, bool), 900.0)
    busy = QuietMap(np.arange(900.0), np.zeros(900, bool), 900.0)
    x = (0.0, 0.0, w.T0)
    assert search.unheard_aircraft(lc, x, evs, rms, all_quiet, w.c, cfg) == ["zz9999"]
    assert search.unheard_aircraft(lc, x, evs, rms, busy, w.c, cfg) == []  # masked: no evidence
    far = [("zz9999", t, enu + [0, 3000.0, 0])]  # 3 km away: quieter than what we heard, no penalty
    lc.nearby = far
    assert search.unheard_aircraft(lc, x, evs, rms, all_quiet, w.c, cfg) == []
