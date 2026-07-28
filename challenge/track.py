"""Arc-length geometry for the diagram-shaped EI Robotics Challenge course.

The course is a closed sequence of straight lines and circular corner fillets.
All consumers use the same ``s`` coordinate, measured in metres in the driving
direction.  This keeps rendering, rules, controllers, and obstacle placement
aligned even when the layout changes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

# Diagram-proportioned centerline.  It starts just after the top-left corner,
# then follows the sketch clockwise through two stacked return straights.
WAYPOINTS = (
    (-3.45, 2.35), (3.15, 2.35), (3.85, 1.65), (3.85, 0.75),
    (3.30, 0.20), (-2.70, 0.20), (-3.30, -0.40), (-3.30, -0.80),
    (-2.75, -1.35), (2.35, -1.35), (3.05, -1.60), (3.45, -2.15),
    (3.10, -2.70), (2.40, -2.95), (-3.35, -2.95), (-4.10, -2.20),
    (-4.10, 1.70),
)
CORNER_RADIUS = 0.38

# Road and vehicle envelope.
ROAD_HALF_WIDTH = 0.18
CAR_HALF_WIDTH = 0.082
OFFTRACK_MINOR = ROAD_HALF_WIDTH - CAR_HALF_WIDTH
CHOKE_GAP_HALF = 0.15


@dataclass(frozen=True)
class Segment:
    kind: str
    s0: float
    length: float
    p0: tuple[float, float]
    p1: tuple[float, float]
    heading: float = 0.0
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0
    angle0: float = 0.0
    delta: float = 0.0


def _unit(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    return dx / length, dy / length


def _build_segments() -> tuple[Segment, ...]:
    count = len(WAYPOINTS)
    entries: list[tuple[float, float]] = []
    exits: list[tuple[float, float]] = []
    arcs: list[tuple[tuple[float, float], float, float, float]] = []

    for i, point in enumerate(WAYPOINTS):
        prev = WAYPOINTS[(i - 1) % count]
        nxt = WAYPOINTS[(i + 1) % count]
        incoming = _unit(point[0] - prev[0], point[1] - prev[1])
        outgoing = _unit(nxt[0] - point[0], nxt[1] - point[1])
        turn = math.atan2(
            incoming[0] * outgoing[1] - incoming[1] * outgoing[0],
            incoming[0] * outgoing[0] + incoming[1] * outgoing[1],
        )
        tangent = CORNER_RADIUS * math.tan(abs(turn) / 2)
        tangent = min(
            tangent,
            0.42 * math.dist(prev, point),
            0.42 * math.dist(point, nxt),
        )
        radius = tangent / max(math.tan(abs(turn) / 2), 1e-9)
        entry = (point[0] - incoming[0] * tangent, point[1] - incoming[1] * tangent)
        exit_ = (point[0] + outgoing[0] * tangent, point[1] + outgoing[1] * tangent)
        side = 1.0 if turn > 0 else -1.0
        center = (
            entry[0] - incoming[1] * side * radius,
            entry[1] + incoming[0] * side * radius,
        )
        entries.append(entry)
        exits.append(exit_)
        arcs.append((center, radius, math.atan2(entry[1] - center[1], entry[0] - center[0]), turn))

    segments: list[Segment] = []
    s0 = 0.0
    for i in range(count):
        j = (i + 1) % count
        p0, p1 = exits[i], entries[j]
        length = math.dist(p0, p1)
        heading = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        segments.append(Segment("line", s0, length, p0, p1, heading=heading))
        s0 += length

        center, radius, angle0, delta = arcs[j]
        length = abs(delta) * radius
        segments.append(
            Segment("arc", s0, length, entries[j], exits[j],
                    center=center, radius=radius, angle0=angle0, delta=delta)
        )
        s0 += length
    return tuple(segments)


SEGMENTS = _build_segments()
TOTAL = sum(segment.length for segment in SEGMENTS)


def path_point(s: float) -> tuple[float, float, float]:
    """Return centerline ``(x, y, heading)`` at arc length ``s``."""
    s = s % TOTAL
    segment = SEGMENTS[-1]
    for candidate in SEGMENTS:
        if s < candidate.s0 + candidate.length:
            segment = candidate
            break
    local = min(max(s - segment.s0, 0.0), segment.length)
    if segment.kind == "line":
        t = local / segment.length
        x = segment.p0[0] + (segment.p1[0] - segment.p0[0]) * t
        y = segment.p0[1] + (segment.p1[1] - segment.p0[1]) * t
        return x, y, segment.heading
    direction = 1.0 if segment.delta > 0 else -1.0
    angle = segment.angle0 + direction * local / segment.radius
    x = segment.center[0] + segment.radius * math.cos(angle)
    y = segment.center[1] + segment.radius * math.sin(angle)
    return x, y, angle + direction * math.pi / 2


def _arc_progress(segment: Segment, angle: float) -> float:
    direction = 1.0 if segment.delta > 0 else -1.0
    if direction > 0:
        swept = (angle - segment.angle0) % (2 * math.pi)
    else:
        swept = (segment.angle0 - angle) % (2 * math.pi)
    return min(swept, abs(segment.delta))


def frenet(x: float, y: float) -> tuple[float, float]:
    """Project a world point onto the closest centerline segment."""
    best: tuple[float, float, float] | None = None
    for segment in SEGMENTS:
        if segment.kind == "line":
            vx, vy = segment.p1[0] - segment.p0[0], segment.p1[1] - segment.p0[1]
            t = ((x - segment.p0[0]) * vx + (y - segment.p0[1]) * vy) / (segment.length**2)
            t = min(max(t, 0.0), 1.0)
            px, py = segment.p0[0] + t * vx, segment.p0[1] + t * vy
            heading = segment.heading
            s = segment.s0 + t * segment.length
        else:
            angle = math.atan2(y - segment.center[1], x - segment.center[0])
            swept = _arc_progress(segment, angle)
            direction = 1.0 if segment.delta > 0 else -1.0
            projected_angle = segment.angle0 + direction * swept
            px = segment.center[0] + segment.radius * math.cos(projected_angle)
            py = segment.center[1] + segment.radius * math.sin(projected_angle)
            heading = projected_angle + direction * math.pi / 2
            s = segment.s0 + swept * segment.radius
        dx, dy = x - px, y - py
        distance2 = dx * dx + dy * dy
        lateral = dx * -math.sin(heading) + dy * math.cos(heading)
        if best is None or distance2 < best[0]:
            best = distance2, s % TOTAL, lateral
    assert best is not None
    return best[1], best[2]


def _s_at(x: float, y: float) -> float:
    return frenet(x, y)[0]


# Named features in driving order. Coordinates are deliberately on long,
# straight portions so tunnels, stop lines, and obstacle offsets are stable.
START_S = _s_at(-2.85, 2.35)
CHOKE = (_s_at(1.25, 2.35), _s_at(1.70, 2.35))
DYN_OBSTACLE_S = _s_at(0.80, 0.20)
BICYCLE_YAW_OFFSET = math.pi / 2
SPEED = (_s_at(1.55, 0.20), _s_at(0.25, 0.20))
TUNNEL_1 = (_s_at(-0.65, 0.20), _s_at(-1.95, 0.20))
SHINE = (_s_at(-2.70, -1.35), _s_at(1.60, -1.35))
TUNNEL_2 = (_s_at(0.65, -2.95), _s_at(-0.80, -2.95))
T2_OBSTACLE_S = _s_at(-0.10, -2.95)
TRAFFIC_LIGHT_S = _s_at(-4.10, 0.80)
TRAFFIC_STOP_S = _s_at(-4.10, 0.48)

# Expanded footprint used by the generated floor and overview cameras.
MIN_X = min(point[0] for point in WAYPOINTS) - 0.7
MAX_X = max(point[0] for point in WAYPOINTS) + 0.7
MIN_Y = min(point[1] for point in WAYPOINTS) - 0.7
MAX_Y = max(point[1] for point in WAYPOINTS) + 0.7


def s_delta(s_new: float, s_old: float) -> float:
    """Shortest signed arc-length difference s_new - s_old on the loop."""
    d = (s_new - s_old) % TOTAL
    if d > TOTAL / 2:
        d -= TOTAL
    return d


def in_range(s: float, rng: tuple[float, float]) -> bool:
    """True if s lies inside the (possibly wrapping) arc-length range."""
    lo, hi = rng[0] % TOTAL, rng[1] % TOTAL
    s = s % TOTAL
    if lo <= hi:
        return lo <= s <= hi
    return s >= lo or s <= hi
