"""End-to-end validation cases (audio + flags -> location ± tolerance).

Each case in tests/validation/*.toml becomes one pytest test.
Holdout cases (tests/validation/holdout.toml) are excluded from the default run;
run them with `pytest -m holdout` or `sya validate tests/validation/holdout.toml`.
"""
import os

import pytest

from sya.validate import load_cases, run_case

HERE = os.path.join(os.path.dirname(__file__), "validation")


def _cases(fname):
    p = os.path.join(HERE, fname)
    return [pytest.param(c, id=c["name"]) for c in load_cases(p)] if os.path.exists(p) else []


@pytest.mark.validation
@pytest.mark.parametrize("case", _cases("cases.toml"))
def test_case(case):
    r = run_case(case, HERE)
    assert r.ok, f"{r.name}: status={r.status} err={r.err_m} dt={r.dt_s} {r.detail}"


@pytest.mark.holdout
@pytest.mark.parametrize("case", _cases("holdout.toml"))
def test_holdout(case):
    r = run_case(case, HERE)
    assert r.ok, f"{r.name}: status={r.status} err={r.err_m} dt={r.dt_s} {r.detail}"
