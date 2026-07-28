"""Scoring for the Gauntlet, following the official hackathon guidelines.

Primary metric: fastest valid lap time (teams get 3 attempts; run 3 episodes
and keep the best). Points are secondary and used for tie-breaking.

Implemented from the official tables:

Bonuses (auto-judged subset — smoothness/precision are judged by humans):
    +5   obstacle avoidance (dynamic bicycle passed without contact)
    +5   low-light navigation (tunnel #2 cleared without contact)
    +5   high-glare handling (no lane violations through the glare curve)
    +5   speed section mastery (>1.5 m/s there without losing the line)
    +10  stopping accuracy (stop within 5 cm of the cube, no contact)

Penalties:
    -2   leaving the track, once per incident until the car returns
    -5   hesitation/stalling more than 2 s
    -10  crossing the traffic-light stop line on red
    -15  contact with an obstacle or the stop cube

Attempt-forfeiting fouls (no lap time recorded):
    collision with track structure, flip, lap timeout
"""

from __future__ import annotations

from dataclasses import dataclass, field

POINTS = {
    "lap_complete": 0,
    "bonus": 5,
    "stop_precise": 10,
    "off_track_minor": -2,
    "stall": -5,
    "traffic_light_violation": -10,
    "obstacle_contact": -15,
    "cube_contact": -15,
}


@dataclass
class ScoreKeeper:
    """Tallies scoring events during one attempt."""

    events: list = field(default_factory=list)
    forfeited: bool = False
    lap_time: float | None = None
    stop_gap: float | None = None
    _pending_reward: float = 0.0

    def _add(self, kind: str, t: float, points: int, detail: str = ""):
        self.events.append({"t": round(t, 2), "event": kind, "points": points, "detail": detail})
        self._pending_reward += points / 10

    # ---- lap ----
    def lap_complete(self, t: float):
        self.lap_time = t
        self._add("lap_complete", t, 0, f"lap time {t:.2f}s")
        self._pending_reward += 5

    def bonus(self, name: str, t: float):
        self._add("bonus", t, POINTS["bonus"], name)

    # ---- penalties ----
    def off_track_minor(self, t: float):
        self._add("off_track_minor", t, POINTS["off_track_minor"])

    def stall(self, t: float):
        self._add("stall", t, POINTS["stall"])

    def traffic_light_violation(self, t: float):
        self._add("traffic_light_violation", t, POINTS["traffic_light_violation"],
                  "crossed stop line on red")

    def obstacle_contact(self, t: float):
        self._add("obstacle_contact", t, POINTS["obstacle_contact"])

    def cube_contact(self, t: float):
        self._add("cube_contact", t, POINTS["cube_contact"])

    # ---- end-zone stop ----
    def stop_result(self, gap: float, t: float):
        """Car has come to rest near the cube; judge the stop."""
        self.stop_gap = gap
        if 0 < gap <= 0.05:
            self._add("stop_precise", t, POINTS["stop_precise"], f"gap {gap * 100:.1f} cm")
        elif 0 < gap <= 0.10:
            self._add("stop_ok", t, 0, f"gap {gap * 100:.1f} cm")
        else:
            self._add("stop_missed", t, 0, f"gap {gap * 100:.1f} cm (need <= 10 cm)")

    def stop_timeout(self, t: float):
        self._add("stop_timeout", t, 0, "did not stop within 30 s of finishing")

    # ---- forfeits ----
    def forfeit(self, reason: str, t: float):
        """Attempt-forfeiting foul: no lap time is recorded."""
        self.forfeited = True
        self.lap_time = None
        self._add("forfeit", t, 0, reason)
        self._pending_reward -= 5

    # ---- results ----
    def total(self) -> int:
        return sum(e["points"] for e in self.events)

    def result(self) -> dict:
        """Attempt outcome: lap_time is None if the attempt was forfeited."""
        return {"lap_time": self.lap_time, "points": self.total(),
                "valid": self.lap_time is not None, "events": list(self.events)}

    def consume_reward(self) -> float:
        """Reward accumulated since last call (used by GauntletEnv.step)."""
        r, self._pending_reward = self._pending_reward, 0.0
        return r

    def summary(self) -> str:
        lines = ["--- Gauntlet attempt summary ---"]
        for e in self.events:
            detail = f" ({e['detail']})" if e["detail"] else ""
            lines.append(f"  t={e['t']:7.2f}s  {e['event']:<16}{detail:<26} {e['points']:+d}")
        if self.lap_time is not None:
            lines.append(f"  LAP TIME: {self.lap_time:.2f}s   points: {self.total()}")
        else:
            lines.append(f"  NO VALID LAP (forfeited)   points: {self.total()}")
        return "\n".join(lines)
