import numpy as np

from planeloc import audio, locate, synth
from planeloc.geo import LocalFrame
from planeloc.physics import received_ratio, speed_of_sound, straight_pass

C = speed_of_sound(15)


def test_geo_roundtrip():
    fr = LocalFrame(47.4, -122.1, 50)
    enu = np.array([[1234.0, -5678.0, 900.0], [0, 0, 0]])
    lat, lon, h = fr.from_enu(enu)
    np.testing.assert_allclose(fr.to_enu(lat, lon, h), enu, atol=1e-6)


def test_straight_pass_matches_track_model():
    v, d, f0 = 80.0, 900.0, 110.0
    te = np.arange(-120, 120, 0.05)
    pos = np.stack([v * te, np.full_like(te, d), np.zeros_like(te)], 1)
    t_r, _, ratio = received_ratio(te, pos, np.zeros(3), C)
    t = np.linspace(-60, 60, 200) + d / C
    expect = straight_pass(t, f0, v, d, d / C, C)
    got = f0 * np.interp(t, t_r, ratio)
    np.testing.assert_allclose(got, expect, atol=0.02)


def test_fit_pass_recovers_parameters():
    rng = np.random.default_rng(0)
    t = np.arange(0, 60, 0.1)
    f = straight_pass(t, 120.0, 75.0, 1200.0, 31.0, C) + rng.normal(0, 0.3, t.size)
    fit = audio.fit_pass(t, f, c=C)
    assert abs(fit.f0 - 120) < 0.2
    assert abs(fit.v - 75) < 1.5
    assert abs(fit.d - 1200) < 40
    assert abs(fit.t_ca - 31) < 0.2


def _event_from_truth(world, fl, t0_rel, t1_rel, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    t_r, _, ratio = received_ratio(fl.t, fl.p, world.obs, world.c)
    t = np.arange(t0_rel, t1_rel, 0.1)
    f = fl.f0 * np.interp(world.T0 + t, t_r, ratio) + rng.normal(0, noise, t.size)
    return audio.Event(t, f, np.ones_like(t), audio.fit_pass(t, f, c=world.c))


def test_locate_two_events_without_audio():
    w = synth.make_world(seed=4, n_background=25)
    tracks = synth.adsb_tracks(w, seed=5)
    evs = [_event_from_truth(w, w.flights[0], 170, 235), _event_from_truth(w, w.flights[1], 615, 670, seed=1)]
    res, _, _ = locate.locate(evs, tracks, w.frame.lat0, w.frame.lon0, 8000, 150,
                              w.T0 - 1800, w.T0 + 1800, w.dur, ground_h=w.ground_h, verbose=False)
    assert res, "no solution"
    best = res[0]
    assert np.hypot(best.e - w.obs_en[0], best.n - w.obs_en[1]) < 150
    assert abs(best.T0 - w.T0) < 1.0
    assert best.flights == ["tgt000", "tgt001"]


def test_ridge_tracker_follows_chirp():
    fs = 2000
    t = np.arange(0, 60, 1 / fs)
    f_inst = straight_pass(t, 100.0, 70.0, 800.0, 30.0, C)
    phase = 2 * np.pi * np.cumsum(f_inst) / fs
    x = 0.5 * np.sin(phase) + np.random.default_rng(1).normal(0, 1, t.size)
    S = audio.spectrogram(x, fs, 200)
    H = audio.harmonic_sum(S, 60, 160, 1)
    r = audio.track_ridge(H)
    truth = np.interp(r.t, t, f_inst)
    assert np.median(np.abs(r.f - truth)) < 0.5
