import os

import numpy as np
import pytest

from sya import audio
from sya.physics import received_ratio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def event_from_truth(world, fl, t0_rel, t1_rel, noise=0.2, seed=0):
    """A flyover Event built from the true track (skips audio): what the audio stage would emit."""
    rng = np.random.default_rng(seed)
    t_r, _, ratio = received_ratio(fl.t, fl.p, world.obs, world.c)
    t = np.arange(t0_rel, t1_rel, 0.1)
    f = fl.f0 * np.interp(world.T0 + t, t_r, ratio) + rng.normal(0, noise, t.size)
    return audio.Event(t, f, np.ones_like(t), audio.fit_pass(t, f, c=world.c))


@pytest.fixture(scope="session")
def bismarck():
    """Synthetic ND world + its two guaranteed flyovers as Events."""
    from sya.synth import make_world, synth_world_data

    w = make_world("synth:bismarck:1")
    evs = [event_from_truth(w, w.flights[0], 165, 240), event_from_truth(w, w.flights[1], 615, 670, seed=1)]
    return w, synth_world_data(w), evs
