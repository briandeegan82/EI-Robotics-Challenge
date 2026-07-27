"""Track geometry for the Gauntlet line-following course.

The track is a stadium-shaped loop (two straights joined by semicircles)
driven counter-clockwise, sized to fit the official 7m x 4m footprint.
A white line on a dark surface marks the driving path.

Everything here is shared by the XML generator (generate_track.py) and the
simulation (env.py), so changing the geometry in one place updates both.
Distances are meters; ``s`` is arc length along the loop measured from the
bottom-left corner of the stadium, increasing in the driving direction.

Course sections, in driving order (matching the official guidelines):

    start line -> lane keeping -> choke point -> curve -> tunnel #1 ->
    speed section (dynamic obstacle) -> high-glare curve -> tunnel #2 ->
    end zone (stop cube, placed after the lap is complete)
"""

from __future__ import annotations

import math

# stadium shape
STRAIGHT_HALF = 2.0            # straights span x in [-2, 2]
RADIUS = 1.1                   # semicircle radius; straights at y = +/-RADIUS
LEN_STRAIGHT = 2 * STRAIGHT_HALF
LEN_CURVE = math.pi * RADIUS
TOTAL = 2 * LEN_STRAIGHT + 2 * LEN_CURVE   # ~14.9 m lap

# s where each region begins
S_BOTTOM = 0.0                 # bottom straight, heading +x, y = -RADIUS
S_RIGHT = LEN_STRAIGHT         # right semicircle
S_TOP = S_RIGHT + LEN_CURVE    # top straight, heading -x, y = +RADIUS
S_LEFT = S_TOP + LEN_STRAIGHT  # left semicircle (high-glare area)

# course sections (arc-length ranges / positions along the loop)
START_S = 1.6                  # start/finish line (x = -0.4, bottom straight)
CHOKE = (3.4, 3.8)             # narrow corridor at the end of the bottom straight
TUNNEL_1 = (S_TOP + 0.1, S_TOP + 1.3)      # top straight, x 1.9 -> 0.7
SPEED = (TUNNEL_1[1], S_LEFT - 0.1)        # open speed section
DYN_OBSTACLE_S = S_TOP + 2.8               # dynamic obstacle (x = -0.8)
SHINE = (S_LEFT, TOTAL)                    # left curve: high-glare floor
TUNNEL_2 = (0.1, 1.3)          # bottom straight, x -1.9 -> -0.7 (low light)
T2_OBSTACLE_S = 0.7            # obstacle inside tunnel #2 (x = -1.3)
CUBE_S = 2.9                   # stop cube center, placed after lap completion

# lane geometry
LANE_HALF_WIDTH = 0.2          # official sections are ~0.4 m wide
OFFTRACK_MINOR = 0.13          # |lateral| beyond this = wheel on the boundary
OFFTRACK_LOST = 0.40           # |lateral| beyond this = complete loss of track


def path_point(s: float) -> tuple[float, float, float]:
    """Return (x, y, heading) of the line at arc length s (heading in rad)."""
    s = s % TOTAL
    if s < S_RIGHT:                                   # bottom straight
        return -STRAIGHT_HALF + s, -RADIUS, 0.0
    if s < S_TOP:                                     # right semicircle
        phi = -math.pi / 2 + (s - S_RIGHT) / RADIUS
        return (STRAIGHT_HALF + RADIUS * math.cos(phi),
                RADIUS * math.sin(phi),
                phi + math.pi / 2)
    if s < S_LEFT:                                    # top straight
        return STRAIGHT_HALF - (s - S_TOP), RADIUS, math.pi
    phi = math.pi / 2 + (s - S_LEFT) / RADIUS         # left semicircle
    return (-STRAIGHT_HALF + RADIUS * math.cos(phi),
            RADIUS * math.sin(phi),
            phi + math.pi / 2)


def frenet(x: float, y: float) -> tuple[float, float]:
    """Return (s, lateral) for a world position near the track.

    ``lateral`` is the signed offset from the line: positive to the left of
    the driving direction, negative to the right.
    """
    if x > STRAIGHT_HALF:                             # right semicircle
        phi = math.atan2(y, x - STRAIGHT_HALF)        # [-pi/2, pi/2] on-track
        s = S_RIGHT + (phi + math.pi / 2) * RADIUS
        lat = RADIUS - math.hypot(x - STRAIGHT_HALF, y)
    elif x < -STRAIGHT_HALF:                          # left semicircle
        phi = math.atan2(y, x + STRAIGHT_HALF)
        if phi < 0:
            phi += 2 * math.pi                        # [pi/2, 3pi/2] on-track
        s = S_LEFT + (phi - math.pi / 2) * RADIUS
        lat = RADIUS - math.hypot(x + STRAIGHT_HALF, y)
    elif y < 0:                                       # bottom straight
        s = x + STRAIGHT_HALF
        lat = y + RADIUS
    else:                                             # top straight
        s = S_TOP + (STRAIGHT_HALF - x)
        lat = RADIUS - y
    return s % TOTAL, lat


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
