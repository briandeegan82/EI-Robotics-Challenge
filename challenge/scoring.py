"""Scoring for the EI Robotics Challenge, following the official guidelines.

Primary metric: fastest valid lap time (teams get 3 attempts; run 3 episodes
and keep the best). At lap completion, each penalty point adds 1 second to
the recorded lap time (e.g. −5 points → +5 s).

Penalties:
    -2   leaving the track, once per incident until the car returns
    -5   hesitation/stalling more than 2 s
    -10  crossing the traffic-light stop line on red
    -10  taking the lane the fork sign didn't select
    -15  contact with an obstacle

Attempt-forfeiting fouls (no lap time recorded):
    collision with track structure, flip, lap timeout
"""

from __future__ import annotations

from dataclasses import dataclass, field

POINTS = {
    "lap_complete": 0,
    "off_track_minor": -2,
    "stall": -5,
    "traffic_light_violation": -10,
    "wrong_lane": -10,
    "obstacle_contact": -15,
}

SECONDS_PER_PENALTY_POINT = 1.0


@dataclass
class ScoreKeeper:
    """Tallies scoring events during one attempt."""

    events: list = field(default_factory=list)
    forfeited: bool = False
    lap_time: float | None = None
    raw_lap_time: float | None = None
    _pending_reward: float = 0.0

    def _add(self, kind: str, t: float, points: int, detail: str = ""):
        self.events.append({"t": round(t, 2), "event": kind, "points": points, "detail": detail})
        self._pending_reward += points / 10

    def penalty_points(self) -> int:
        """Magnitude of penalty points accrued (non-negative)."""
        return max(0, -self.total())

    # ---- lap ----
    def lap_complete(self, t: float):
        """Record finish: add 1 s to lap time for each penalty point."""
        penalty_s = self.penalty_points() * SECONDS_PER_PENALTY_POINT
        self.raw_lap_time = t
        self.lap_time = t + penalty_s
        detail = f"raw {t:.2f}s + {penalty_s:.0f}s penalties = {self.lap_time:.2f}s"
        self._add("lap_complete", t, 0, detail)
        self._pending_reward += 5

    # ---- penalties ----
    def off_track_minor(self, t: float):
        self._add("off_track_minor", t, POINTS["off_track_minor"])

    def stall(self, t: float):
        self._add("stall", t, POINTS["stall"])

    def traffic_light_violation(self, t: float):
        self._add("traffic_light_violation", t, POINTS["traffic_light_violation"],
                  "crossed stop line on red")

    def wrong_lane(self, t: float):
        self._add("wrong_lane", t, POINTS["wrong_lane"], "took the lane the sign didn't select")

    def obstacle_contact(self, t: float):
        self._add("obstacle_contact", t, POINTS["obstacle_contact"])

    # ---- forfeits ----
    def forfeit(self, reason: str, t: float):
        """Attempt-forfeiting foul: no lap time is recorded."""
        self.forfeited = True
        self.lap_time = None
        self.raw_lap_time = None
        self._add("forfeit", t, 0, reason)
        self._pending_reward -= 5

    # ---- results ----
    def total(self) -> int:
        return sum(e["points"] for e in self.events)

    def result(self) -> dict:
        """Attempt outcome: lap_time is None if the attempt was forfeited."""
        return {"lap_time": self.lap_time, "raw_lap_time": self.raw_lap_time,
                "points": self.total(), "valid": self.lap_time is not None,
                "events": list(self.events)}

    def consume_reward(self) -> float:
        """Reward accumulated since last call (used by ChallengeEnv.step)."""
        r, self._pending_reward = self._pending_reward, 0.0
        return r

    def summary(self) -> str:
        lines = ["--- EI Robotics Challenge attempt summary ---"]
        for e in self.events:
            detail = f" ({e['detail']})" if e["detail"] else ""
            lines.append(f"  t={e['t']:7.2f}s  {e['event']:<16}{detail:<50} {e['points']:+d}")
        if self.lap_time is not None:
            lines.append(
                f"  LAP TIME: {self.lap_time:.2f}s"
                f" (raw {self.raw_lap_time:.2f}s + {self.penalty_points()}s penalties)"
                f"   points: {self.total()}"
            )
        else:
            lines.append(f"  NO VALID LAP (forfeited)   points: {self.total()}")
        return "\n".join(lines)
