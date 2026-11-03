"""Per-pass bookkeeping: wall time and how many items each pass kept."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class PassStat:
    name: str
    seconds: float = 0.0
    n_in: int | None = None
    n_out: int | None = None
    calls: int = 0
    note: str = ""


@dataclass
class PassLog:
    passes: dict = field(default_factory=dict)
    listeners: list = field(default_factory=list)

    @contextmanager
    def timed(self, name: str, n_in: int | None = None):
        st = self.passes.setdefault(name, PassStat(name))
        t = time.perf_counter()
        holder = {"out": None, "note": ""}
        try:
            yield holder
        finally:
            st.seconds += time.perf_counter() - t
            st.calls += 1
            if n_in is not None:
                st.n_in = (st.n_in or 0) + n_in
            if holder["out"] is not None:
                st.n_out = (st.n_out or 0) + holder["out"]
            if holder["note"]:
                st.note = holder["note"]
            for fn in self.listeners:
                fn(st)

    def count(self, name: str, n_in=None, n_out=None, note=""):
        st = self.passes.setdefault(name, PassStat(name))
        if n_in is not None:
            st.n_in = (st.n_in or 0) + n_in
        if n_out is not None:
            st.n_out = (st.n_out or 0) + n_out
        if note:
            st.note = note
        for fn in self.listeners:
            fn(st)

    def as_dict(self):
        return {k: dict(seconds=round(v.seconds, 4), n_in=v.n_in, n_out=v.n_out, calls=v.calls, note=v.note)
                for k, v in self.passes.items()}
