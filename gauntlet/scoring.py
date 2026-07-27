"""Scoring rules for the Gauntlet.

NOTE: the official competition rules are still being finalized — treat these
values as placeholders that show *how* runs will be judged, not the final
numbers. Everything is in one place so rules can be tuned without touching
the simulation.

Current rules:
    +50   per checkpoint (traffic light passed, crossing passed, tunnel exited)
    +500  crossing the finish line
    -100  running a red light
    -150  hitting the bicycle
    -75   hitting the tunnel obstacle
    -10   each wall/curb strike
    -200  flipping the car (ends the run)
Ties are broken by elapsed time (faster wins).
"""

from __future__ import annotations

from dataclasses import dataclass, field


POINTS = {
    "checkpoint": 50,
    "finish": 500,
    "red_light": -100,
    "hit_bicycle": -150,
    "hit_obstacle": -75,
    "hit_wall": -10,
    "flipped": -200,
}


@dataclass
class ScoreKeeper:
    """Tallies scoring events during a run."""

    events: list = field(default_factory=list)
    _pending_reward: float = 0.0

    def _add(self, kind: str, t: float, points: int, detail: str = ""):
        self.events.append({"t": round(t, 2), "event": kind, "points": points, "detail": detail})
        # mirror score changes into the RL reward channel (scaled down)
        self._pending_reward += points / 100

    def checkpoint(self, name: str, t: float):
        self._add("checkpoint", t, POINTS["checkpoint"], name)

    def finish(self, t: float):
        self._add("finish", t, POINTS["finish"])

    def red_light(self, t: float):
        self._add("red_light", t, POINTS["red_light"])

    def collision(self, hazard: str, t: float):
        if hazard == "bicycle":
            self._add("hit_bicycle", t, POINTS["hit_bicycle"])
        elif hazard == "obstacle":
            self._add("hit_obstacle", t, POINTS["hit_obstacle"])
        else:
            self._add("hit_wall", t, POINTS["hit_wall"])

    def flipped(self, t: float):
        self._add("flipped", t, POINTS["flipped"])

    def timeout(self, t: float):
        self._add("timeout", t, 0)

    def total(self) -> int:
        return sum(e["points"] for e in self.events)

    def consume_reward(self) -> float:
        """Reward accumulated since last call (used by GauntletEnv.step)."""
        r, self._pending_reward = self._pending_reward, 0.0
        return r

    def summary(self) -> str:
        lines = ["--- Gauntlet run summary ---"]
        for e in self.events:
            detail = f" ({e['detail']})" if e["detail"] else ""
            lines.append(f"  t={e['t']:7.2f}s  {e['event']:<12}{detail:<18} {e['points']:+d}")
        lines.append(f"  TOTAL: {self.total()}")
        return "\n".join(lines)
