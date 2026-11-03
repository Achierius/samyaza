import os

import numpy as np
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


from sya.synth import event_from_truth  # noqa: E402


@pytest.fixture(scope="session")
def bismarck():
    """Synthetic ND world + its two guaranteed flyovers as Events."""
    from sya.synth import make_world, synth_world_data

    w = make_world("synth:bismarck:1")
    evs = [event_from_truth(w, w.flights[0], 165, 240), event_from_truth(w, w.flights[1], 615, 670, seed=1)]
    return w, synth_world_data(w), evs
