# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU clock sanity guard (X-149; l3_farm/analysis/ncu_clock_guard.md).

Why: an ``ncu`` run that is killed or times out (``--clock-control base``, ncu's default, locks the SM clock to the
base clock for the profiled kernels) can leave the GPU locked LOW. On the L40S of this farm the state looks like this
(reproduced, analysis/ncu_guard_data/): an idle GPU with ``clocks.sm`` fixed at 1,065 MHz (max 2,520), ``pstate`` P0,
``clocks.mem`` at 9,001 (not 405), ``utilization.gpu`` 0; an unlocked idle GPU sits at 210 MHz / P8 / 405 MHz memory
(a few seconds after the last kernel; right after a run it still reads max for 1-5 s). Every later timing on that GPU
is ~2.2x too slow, silently. ``clocks.applications.graphics`` is "[Requested functionality has been deprecated]" on
current drivers, so it cannot be used.

Flag ``KERNELBLASTER_CLOCK_GUARD=1`` (ModelBlaster ``--clock-guard on``), default OFF. With it on, the evaluators query the
leased GPU once before every timed step (and always right after an ncu timeout / failure); a suspect GPU is quarantined
in the GPUPool and the step is re-timed on another GPU when there is one; otherwise the result is marked
``gpu.clock.suspect`` and the agent treats the step as inconclusive (never a regression). The flow never resets clocks:
see the manual procedure in the doc (``sudo rusu reset-gpu -i <id>`` on the node, or release the allocation).

Env: ``KERNELBLASTER_CLOCK_QUARANTINE_S`` (default 900), ``KERNELBLASTER_CLOCK_CONFIRM_S`` (default 1.5: a locked-looking
idle sample is confirmed by a second one that long after, so a clock ramping down is not taken for a lock).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

QUERY_FIELDS = ("index,uuid,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,clocks.applications.graphics,"
                "clocks_throttle_reasons.active,utilization.gpu,pstate")
#: the first eight are the documented query; utilization and pstate tell "idle" from "busy"
OK, LOCKED, THROTTLED, UNKNOWN = "ok", "locked", "throttled", "unknown"

# clocks_throttle_reasons bits (nvml clocksEventReason*)
R_IDLE, R_APP, R_SWPOWER, R_HWSLOW, R_SYNC, R_SWTHERM, R_HWTHERM, R_HWBRAKE, R_DISPLAY = (
    0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x100)
_HARD = R_HWSLOW | R_HWTHERM | R_HWBRAKE | R_SWTHERM
_NAMES = {R_IDLE: "gpu_idle", R_APP: "applications_clocks_setting", R_SWPOWER: "sw_power_cap", R_HWSLOW: "hw_slowdown",
          R_SYNC: "sync_boost", R_SWTHERM: "sw_thermal_slowdown", R_HWTHERM: "hw_thermal_slowdown",
          R_HWBRAKE: "hw_power_brake", R_DISPLAY: "display_clock_setting"}


def enabled() -> bool:
    return os.getenv("KERNELBLASTER_CLOCK_GUARD", "").strip().lower() in ("1", "true", "yes", "on")


def quarantine_s() -> float:
    try:
        return float(os.getenv("KERNELBLASTER_CLOCK_QUARANTINE_S", "900"))
    except ValueError:
        return 900.0


def confirm_s() -> float:
    try:
        return float(os.getenv("KERNELBLASTER_CLOCK_CONFIRM_S", "1.5"))
    except ValueError:
        return 1.5


def query_command(ident=None, index=None) -> str:
    """The nvidia-smi query for one GPU (UUID or index; neither: every GPU)."""
    sel = f" -i {ident}" if ident and str(ident).startswith("GPU-") else (f" -i {index}" if index is not None else "")
    return f"nvidia-smi --query-gpu={QUERY_FIELDS}{sel} --format=csv,noheader"


def _num(v: str) -> Optional[float]:
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", v or "")
    return float(m.group(1)) if m else None             # "[N/A]", "[Requested functionality has been deprecated]"


@dataclass
class ClockSample:
    index: Optional[int] = None
    uuid: Optional[str] = None
    sm: Optional[float] = None
    max_sm: Optional[float] = None
    mem: Optional[float] = None
    max_mem: Optional[float] = None
    app_gfx: Optional[float] = None
    reasons: Optional[int] = None
    util: Optional[float] = None
    pstate: Optional[str] = None

    def reason_names(self) -> list:
        r = self.reasons or 0
        return [n for b, n in _NAMES.items() if r & b]

    def to_dict(self) -> dict:
        return {k: v for k, v in {"index": self.index, "uuid": self.uuid, "sm_mhz": self.sm, "max_sm_mhz": self.max_sm,
                                  "mem_mhz": self.mem, "max_mem_mhz": self.max_mem, "app_gfx_mhz": self.app_gfx,
                                  "throttle": None if self.reasons is None else hex(self.reasons),
                                  "throttle_names": self.reason_names() or None, "util_pct": self.util,
                                  "pstate": self.pstate}.items() if v is not None}


def parse_query(text: str) -> list[ClockSample]:
    """Lines of ``nvidia-smi --query-gpu=<QUERY_FIELDS> --format=csv,noheader[,nounits]``; the shorter documented
    query (without utilization / pstate) parses too."""
    out = []
    for line in (text or "").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 7 or not p[0].isdigit():
            continue
        r = None
        if len(p) > 7:
            try:
                r = int(p[7], 16) if p[7].lower().startswith("0x") else int(p[7])
            except ValueError:
                r = None
        out.append(ClockSample(int(p[0]), p[1] or None, _num(p[2]), _num(p[3]), _num(p[4]), _num(p[5]), _num(p[6]),
                               r, _num(p[8]) if len(p) > 8 else None, p[9] if len(p) > 9 else None))
    return out


@dataclass
class ClockVerdict:
    state: str = UNKNOWN                    # ok | locked | throttled | unknown (query failed)
    reasons: list = field(default_factory=list)
    sample: Optional[ClockSample] = None
    phase: str = "pre"                      # pre (before a timed step) | post_ncu (after an ncu timeout / failure)
    confirmed: bool = False                 # a locked-looking idle sample was confirmed by a second one
    t: float = 0.0

    @property
    def suspect(self) -> bool:
        return self.state in (LOCKED, THROTTLED)

    def to_dict(self) -> dict:
        return {"state": self.state, "suspect": self.suspect, "phase": self.phase, "reasons": list(self.reasons),
                **({"confirmed": True} if self.confirmed else {}), **(self.sample.to_dict() if self.sample else {})}


def classify(s: ClockSample, *, idle_util: float = 10.0) -> tuple[str, list]:
    """(state, reasons) of one sample. Rules, from the reproduced locked state (see the module doc):

    * hard throttle (HW slowdown / thermal / power brake): ``throttled``;
    * application clocks (where the driver reports them) below 0.9 x max: ``locked``;
    * GPU idle (util <= ``idle_util``) with the SM clock in the band [0.25, 0.85] x max: ``locked``. A healthy idle GPU
      reads the idle floor (~8% of max) or, for a few seconds after a run, the boost clock; neither is in the band;
    * GPU not idle and the SM clock below 0.6 x max: ``throttled`` (a running GPU boosts; ncu's base-clock lock gives
      0.42 x max on the L40S).
    The one-sample verdicts of the last two rules are confirmed by a second sample (``ClockGuard.check``): clocks
    ramp up and down in the first moments of a load and after it.
    """
    if s.sm is None or not s.max_sm:
        return UNKNOWN, ["no SM clock in the query"]
    frac = s.sm / s.max_sm
    r = s.reasons or 0
    if r & _HARD:
        return THROTTLED, [f"hardware/thermal throttle active ({'+'.join(n for b, n in _NAMES.items() if r & b and b & _HARD)})"]
    if s.app_gfx is not None and s.app_gfx < 0.9 * s.max_sm:
        return LOCKED, [f"application clocks {s.app_gfx:.0f} MHz < 0.9 x max {s.max_sm:.0f}"]
    idle = s.util is None or s.util <= idle_util
    if idle and 0.25 <= frac <= 0.85:
        return LOCKED, [f"idle GPU (util {s.util if s.util is not None else '?'}%) with the SM clock held at "
                        f"{s.sm:.0f} MHz = {100 * frac:.0f}% of max {s.max_sm:.0f} (an unlocked idle GPU reads "
                        f"~{0.08 * s.max_sm:.0f} MHz or the max)"]
    if not idle and frac < 0.6:
        return THROTTLED, [f"busy GPU (util {s.util:.0f}%) at {s.sm:.0f} MHz = {100 * frac:.0f}% of max {s.max_sm:.0f}"
                           + (f" ({'+'.join(s.reason_names())})" if s.reasons else "")]
    return OK, []


Runner = Callable[[str], Awaitable[Optional[str]]]


class ClockGuard:
    """Counts, remembers and (through ``check``) classifies. One per evaluator; the evaluator supplies the way to run
    a command on the GPU's host (local shell / the existing ssh path)."""

    def __init__(self, *, quarantine: Optional[float] = None, confirm: Optional[float] = None, clock=time.time,
                 sleep=None):
        self.quarantine_s = quarantine_s() if quarantine is None else quarantine
        self.confirm_s = confirm_s() if confirm is None else confirm
        self.clock = clock
        self.sleep = sleep
        self.checks = 0
        self.failed_queries = 0
        self.suspects = 0
        self.events: list = []              # the suspect verdicts (bounded), newest last
        self.last: dict = {}                # device key -> the last verdict

    async def _sample(self, run: Runner, ident, index) -> Optional[ClockSample]:
        text = await run(query_command(ident, index))
        rows = parse_query(text or "")
        if not rows:
            return None
        if ident and str(ident).startswith("GPU-"):
            rows = [r for r in rows if r.uuid == ident] or rows
        return rows[0]

    async def check(self, run: Runner, *, ident=None, index=None, phase: str = "pre", key=None,
                    label: str = "") -> ClockVerdict:
        import asyncio
        self.checks += 1
        s = await self._sample(run, ident, index)
        v = ClockVerdict(phase=phase, t=self.clock())
        if s is None:
            self.failed_queries += 1
            v.state, v.reasons = UNKNOWN, ["nvidia-smi clock query failed"]
        else:
            v.sample = s
            v.state, v.reasons = classify(s)
            if v.suspect and not v.reasons[0].startswith(("application clocks", "hardware/thermal")):
                # idle-band rule: confirm (a clock that is ramping down moves between the two samples)
                await (self.sleep or asyncio.sleep)(self.confirm_s)
                s2 = await self._sample(run, ident, index)
                if s2 is not None and s2.sm is not None and s.sm is not None and abs(s2.sm - s.sm) <= 0.03 * (s.max_sm or 1) \
                        and classify(s2)[0] in (LOCKED, THROTTLED):
                    v.state, v.reasons = classify(s2)
                    v.sample, v.confirmed = s2, True
                else:
                    v.state, v.reasons = OK, []
                    v.sample = s2 or s
        if key is not None:
            self.last[key] = v
        if v.suspect:
            self.suspects += 1
            self.events.append({"t": round(v.t, 1), "device": label or str(key), **v.to_dict()})
            del self.events[:-50]
        return v

    def stats(self) -> dict:
        return {"checks": self.checks, "suspect": self.suspects, "failed_queries": self.failed_queries,
                "quarantine_s": self.quarantine_s, "events": list(self.events[-10:])}


def clock_suspect(res) -> bool:
    """Did the GPU this result was timed on read clock-suspect (None-safe; works on EvalResult or its dict)?"""
    g = getattr(res, "gpu", None) if not isinstance(res, dict) else res.get("gpu")
    c = (g or {}).get("clock") if isinstance(g, dict) else None
    return bool(c and c.get("suspect"))
