"""Generate challenge/assets/challenge.xml from the track geometry in track.py.

    python -m challenge.generate_track

Rerun after changing track.py (or the wall/gate/tunnel parameters below). The
generated XML is committed so teams don't need to run this unless they
customize the course.
"""

from __future__ import annotations

import math
from pathlib import Path

from . import track

W = track.ROAD_HALF_WIDTH    # paved half-width (car stays between the edges)
LINE_W = 0.0125              # white edge-line half-width (~25 mm line)
ROAD_Z = 0.0206              # paved surface, just above the floor
EDGE_Z = 0.0212              # edge lines, just above the road
ROAD_STEP = 0.02             # arc sweep resolution (finer = smoother curves)

# dark (below the road brightness) so it reads as "not road" to the camera
DARK_WALL = 'rgba="0.13 0.13 0.15 1"'
CHECKER_BLACK = 'rgba="0.08 0.08 0.08 1"'
CHECKER_WHITE = 'rgba="0.92 0.92 0.92 1"'

# Full span the fork touches, including its widening/narrowing tapers on
# either side of the constant-width core (track.FORK_S).
FORK_FULL = (track.FORK_S[0] - track.FORK_TAPER, track.FORK_S[1] + track.FORK_TAPER)


def _quat_from_axes(x_axis, y_axis, z_axis) -> tuple[float, float, float, float]:
    """Return MuJoCo (w, x, y, z) quat for a rotation with the given columns."""
    xx, xy, xz = x_axis
    yx, yy, yz = y_axis
    zx, zy, zz = z_axis
    trace = xx + yy + zz
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (zy - yz) * s
        y = (xz - zx) * s
        z = (yx - xy) * s
    elif xx > yy and xx > zz:
        s = 2.0 * math.sqrt(1.0 + xx - yy - zz)
        w = (zy - yz) / s
        x = 0.25 * s
        y = (xy + yx) / s
        z = (xz + zx) / s
    elif yy > zz:
        s = 2.0 * math.sqrt(1.0 + yy - xx - zz)
        w = (xz - zx) / s
        x = (xy + yx) / s
        y = 0.25 * s
        z = (yz + zy) / s
    else:
        s = 2.0 * math.sqrt(1.0 + zz - xx - yy)
        w = (yx - xy) / s
        x = (xz + zx) / s
        y = (yz + zy) / s
        z = 0.25 * s
    return w, x, y, z


def road_and_edges() -> list[str]:
    """Sweep the paved strip and edge lines along the shared centerline.

    On a curve a straight box is a chord of the arc, so two things can tear:
    the paved strip leaves wedges of ground at the outer radius, and the thin
    edge lines either dash (outer) or fan past the boundary (inner). Both are
    avoided here without overshooting boxes: a fine ROAD_STEP keeps the road
    chords hugging the arc, and each edge line is emitted as a proper polyline
    -- one box per gap between successive boundary points at that line's own
    radius -- so it stays continuous and on-curve on both sides.
    """
    g = []
    index = 0
    for segment in track.SEGMENTS:
        divisions = math.ceil(segment.length / ROAD_STEP) if segment.kind == "arc" else 1
        cuts = [segment.s0 + segment.length * i / divisions for i in range(divisions + 1)]
        if segment.kind == "line":
            for boundary in (*track.FORK_S, *FORK_FULL):
                if segment.s0 < boundary < segment.s0 + segment.length:
                    cuts.append(boundary)
            cuts.sort()

        # paved strip: chord boxes centered on the centerline. On a curve these
        # wide (2*W) boxes overlap heavily near the tight inner radius; if they
        # were all coplanar the overlaps would z-fight into bright radial moire,
        # so on arcs each box gets a tiny cycling z-offset (well under
        # EDGE_Z - ROAD_Z) to make neighbours resolve by draw order instead.
        # Straight boxes abut without overlapping, so they stay flat at ROAD_Z.
        for road_i, (lo, hi) in enumerate(zip(cuts, cuts[1:])):
            s = (lo + hi) / 2
            if track.in_range(s, FORK_FULL):
                continue  # the fork (incl. tapers) has its own pavement, drawn separately
            x, y, heading = track.path_point(s)
            road_half = (hi - lo) / 2 + 0.004
            z = ROAD_Z + (road_i % 6) * 6e-5 if segment.kind == "arc" else ROAD_Z
            g.append(
                f'<geom name="road_{index}" type="box" size="{road_half:.4f} {W} 0.0006" '
                f'pos="{x:.4f} {y:.4f} {z:.5f}" euler="0 0 {heading:.4f}" '
                f'material="road" contype="0" conaffinity="0"/>'
            )
            index += 1

        # edge lines: a continuous polyline at each boundary radius, built from
        # chords between the boundary points sampled at every cut
        for side, sign in (("l", 1), ("r", -1)):
            points = []
            for c in cuts:
                px, py, ph = track.path_point(c)
                points.append((px - math.sin(ph) * sign * W, py + math.cos(ph) * sign * W))
            for (lo, hi), (ax, ay), (bx, by) in zip(zip(cuts, cuts[1:]), points, points[1:]):
                if track.in_range((lo + hi) / 2, FORK_FULL):
                    continue  # the fork (incl. tapers) paints its own edges, drawn separately
                mx, my = (ax + bx) / 2, (ay + by) / 2
                ehead = math.atan2(by - ay, bx - ax)
                ehalf = math.hypot(bx - ax, by - ay) / 2 + 0.002
                g.append(
                    f'<geom name="edge_{index}_{side}" type="box" '
                    f'size="{ehalf:.4f} {LINE_W} 0.0004" '
                    f'pos="{mx:.4f} {my:.4f} {EDGE_Z}" euler="0 0 {ehead:.4f}" '
                    f'material="line" contype="0" conaffinity="0"/>'
                )
                index += 1
    return g


def shine_light() -> list[str]:
    """Floodlight tower beyond the SHINE straight, aimed back down it.

    The straight's paving stays the plain "road" material (no special-cased
    surface); the reflective streak the car drives toward instead comes from
    a real light interacting with that material's own (modest) specular
    response, rather than a patch of baked-in shininess. The <light> itself
    has no visible geometry in MuJoCo, so this also builds the pole, housing,
    and lamp lens it's mounted in -- positioned and aimed together, roadside
    like the traffic signal rather than planted in the lane -- so the glare
    visibly comes from somewhere instead of thin air.

    Wide cutoff + low exponent spread the beam into a broad, even flood
    rather than a tight hotspot, and gentle attenuation lets it stay bright
    over the straight's full length, so the washed-out area is large enough
    to actually obscure the lane edge lines, not just glint off them.
    """
    px, py, heading = offset_pose(track.SHINE[1] + 0.2, W + 0.22)
    tx, ty = math.cos(heading), math.sin(heading)
    tower_h = 0.85
    lamp = (px - tx * 0.045, py - ty * 0.045, tower_h)
    aim = (-tx, -ty, -0.30)
    rotation = f'euler="0 0 {heading:.4f}"'
    return [
        f'<geom name="shine_pole" type="cylinder" size="0.02 {tower_h / 2:.4f}" '
        f'pos="{px:.4f} {py:.4f} {tower_h / 2:.4f}" rgba="0.15 0.15 0.15 1" '
        f'contype="0" conaffinity="0"/>',
        f'<geom name="shine_housing" type="box" size="0.045 0.065 0.03" '
        f'pos="{px:.4f} {py:.4f} {tower_h:.4f}" {rotation} '
        f'rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/>',
        f'<geom name="shine_lamp" type="sphere" size="0.032" '
        f'pos="{lamp[0]:.4f} {lamp[1]:.4f} {lamp[2]:.4f}" '
        f'rgba="1.0 0.97 0.85 1" contype="0" conaffinity="0"/>',
        f'<light name="shine_light" pos="{lamp[0]:.4f} {lamp[1]:.4f} {lamp[2]:.4f}" '
        f'dir="{aim[0]:.4f} {aim[1]:.4f} {aim[2]:.4f}" '
        f'diffuse="3.6 3.4 3.0" specular="4.0 4.0 3.6" ambient="0 0 0" '
        f'attenuation="1 0 0.05" cutoff="65" exponent="1" castshadow="false"/>',
    ]


HATCH_COLOR = 'rgba="0.95 0.82 0.12 1"'
HATCH_HALF_WIDTH = 0.018   # stripe thickness
HATCH_ANGLE = math.radians(45)
HATCH_SPACING = 0.26        # centreline distance between successive stripes


def cross_hatch(name: str, s_range: tuple[float, float]) -> list[str]:
    """Diagonal "keep clear"-style hatch stripes painted across the road.

    Purely a paint layer -- like the edge lines, non-collidable and drawn
    above the plain road material -- so it adds visual clutter for the
    camera pipeline without changing the drivable width.
    """
    lo, hi = s_range
    length = track.s_delta(hi, lo)
    n = max(round(length / HATCH_SPACING), 2)
    half_len = W * math.sqrt(2) * 0.92
    g = []
    for i in range(n + 1):
        s = lo + length * i / n
        x, y, heading = track.path_point(s)
        stripe_heading = heading + HATCH_ANGLE
        g.append(
            f'<geom name="{name}_{i}" type="box" size="{half_len:.4f} {HATCH_HALF_WIDTH:.4f} 0.0004" '
            f'pos="{x:.4f} {y:.4f} {EDGE_Z}" euler="0 0 {stripe_heading:.4f}" '
            f'{HATCH_COLOR} contype="0" conaffinity="0"/>'
        )
    return g


def tunnel(name: str, s_range: tuple[float, float]) -> list[str]:
    """Dark tunnel walls (interior width 0.56 m, height 0.22 m) plus a full roof."""
    lo, hi = s_range
    half_len = track.s_delta(hi, lo) / 2
    center_s = lo + half_len
    x_center, y_center, heading = track.path_point(center_s)
    nx, ny = -math.sin(heading), math.cos(heading)
    rotation = f'euler="0 0 {heading:.4f}"'
    g = []
    for side, sy in (("l", 1), ("r", -1)):
        x, y = x_center + nx * sy * 0.295, y_center + ny * sy * 0.295
        g.append(f'<geom name="{name}_wall_{side}" type="box" '
                 f'size="{half_len} 0.015 0.1" '
                 f'pos="{x:.4f} {y:.4f} 0.12" {rotation} {DARK_WALL}/>')
    g.append(f'<geom name="{name}_roof" type="box" size="{half_len} 0.31 0.006" '
             f'pos="{x_center:.4f} {y_center:.4f} 0.228" {rotation} {DARK_WALL}/>')
    return g


def checkerboard_gate(name: str, s: float) -> list[str]:
    """Semicircular checkerboard arch standing over the road at arc-length s."""
    x0, y0, heading = track.path_point(s)
    nx, ny = -math.sin(heading), math.cos(heading)
    tx, ty = math.cos(heading), math.sin(heading)

    r_inner, r_outer = 0.28, 0.36
    n_theta, n_radial, n_depth = 14, 2, 2
    depth_pitch = 0.045
    g = []
    idx = 0
    for i in range(n_theta):
        th = (i + 0.5) * math.pi / n_theta
        dth = math.pi / n_theta
        for ri in range(n_radial):
            radius = r_inner + (ri + 0.5) * (r_outer - r_inner) / n_radial
            half_r = 0.5 * (r_outer - r_inner) / n_radial + 0.001
            for di in range(n_depth):
                depth = (di - 0.5 * (n_depth - 1)) * depth_pitch
                lat = -radius * math.cos(th)
                z = radius * math.sin(th)
                x = x0 + nx * lat + tx * depth
                y = y0 + ny * lat + ty * depth

                # local x = arc tangent, y = along-track, z = x × y (radial)
                xa = (nx * math.sin(th), ny * math.sin(th), math.cos(th))
                ya = (tx, ty, 0.0)
                za = (
                    xa[1] * ya[2] - xa[2] * ya[1],
                    xa[2] * ya[0] - xa[0] * ya[2],
                    xa[0] * ya[1] - xa[1] * ya[0],
                )
                qw, qx, qy, qz = _quat_from_axes(xa, ya, za)

                half_arc = radius * dth / 2 + 0.001
                half_d = depth_pitch / 2 + 0.001
                colour = CHECKER_BLACK if (i + ri + di) % 2 == 0 else CHECKER_WHITE
                g.append(
                    f'<geom name="{name}_{idx}" type="box" '
                    f'size="{half_arc:.4f} {half_d:.4f} {half_r:.4f}" '
                    f'pos="{x:.4f} {y:.4f} {z:.4f}" '
                    f'quat="{qw:.5f} {qx:.5f} {qy:.5f} {qz:.5f}" {colour}/>'
                )
                idx += 1
    return g


def offset_pose(s: float, lateral: float = 0.0) -> tuple[float, float, float]:
    """Return a path pose shifted along its left normal."""
    x, y, heading = track.path_point(s)
    return (
        x - math.sin(heading) * lateral,
        y + math.cos(heading) * lateral,
        heading,
    )


FORK_DIM = 'rgba="0.15 0.14 0.13 1"'
SIGN_POST = 'rgba="0.15 0.15 0.15 1"'
SIGN_BOARD = 'rgba="0.10 0.10 0.12 1"'


def fork_lanes(name: str, s_range: tuple[float, float]) -> list[str]:
    """Two paved lanes plus a divider wall spanning a straight fork section.

    The divider physically blocks the middle of the road, so the car cannot
    drive straight through -- it must commit to the left or right lane. Each
    lane keeps its own pair of white edge lines; env.py judges which lane was
    actually taken against the sign at FORK_SIGN_S.
    """
    lo, hi = s_range
    half_len = track.s_delta(hi, lo) / 2
    center_s = lo + half_len
    x0, y0, heading = track.path_point(center_s)
    nx, ny = -math.sin(heading), math.cos(heading)
    rotation = f'euler="0 0 {heading:.4f}"'
    lane_half = track.FORK_LANE_HALF_WIDTH
    lane_offset = track.FORK_LANE_OFFSET
    g = []
    for side, sign in (("l", 1), ("r", -1)):
        cx, cy = x0 + nx * sign * lane_offset, y0 + ny * sign * lane_offset
        g.append(f'<geom name="{name}_road_{side}" type="box" '
                 f'size="{half_len:.4f} {lane_half:.4f} 0.0006" '
                 f'pos="{cx:.4f} {cy:.4f} {ROAD_Z:.5f}" {rotation} '
                 f'material="road" contype="0" conaffinity="0"/>')
        for edge, radius in (("outer", lane_offset + lane_half), ("inner", lane_offset - lane_half)):
            ex, ey = x0 + nx * sign * radius, y0 + ny * sign * radius
            g.append(f'<geom name="{name}_edge_{side}_{edge}" type="box" '
                     f'size="{half_len + 0.01:.4f} {LINE_W} 0.0004" '
                     f'pos="{ex:.4f} {ey:.4f} {EDGE_Z}" {rotation} '
                     f'material="line" contype="0" conaffinity="0"/>')
    g.append(f'<geom name="{name}_divider" type="box" '
              f'size="{half_len:.4f} {track.FORK_DIVIDER_HALF_WIDTH:.4f} 0.05" '
              f'pos="{x0:.4f} {y0:.4f} 0.07" {rotation} {DARK_WALL}/>')
    return g


def _circular_ease(u: float, run: float, rise: float) -> float:
    """Value at distance ``u`` (0..run) along a smooth circular 'S' curve from
    0 to ``rise``, flat (zero slope) at both ends.

    Built from two tangent circular arcs -- like the fillets that round the
    track's own corners -- rather than a straight ramp: each half is an arc
    through the endpoint with a horizontal tangent, meeting its mirror image
    at the midpoint with matching slope, so the whole curve is a genuine
    circular sweep with no kink where it meets the flat sections on either
    side.
    """
    if run <= 0 or rise == 0:
        return rise if u >= run else 0.0
    sign = 1.0 if rise > 0 else -1.0
    rise = abs(rise)
    half_run, half_rise = run / 2, rise / 2
    radius = (half_run**2 + half_rise**2) / (2 * half_rise)
    if u <= half_run:
        x = max(min(u, radius), 0.0)
        val = radius - math.sqrt(max(radius**2 - x**2, 0.0))
    else:
        x = max(min(run - u, radius), 0.0)
        val = rise - (radius - math.sqrt(max(radius**2 - x**2, 0.0)))
    return sign * val


FORK_TAPER_STEP = 0.05  # length of each staircase segment approximating the sweep


def fork_taper(name: str, s_lo: float, s_hi: float, widening: bool) -> list[str]:
    """Flare the road between one-lane and two-lane width over ``[s_lo, s_hi]``.

    MuJoCo boxes can't themselves curve in cross-section, so -- exactly like
    the arc road sweep in road_and_edges() -- this approximates a smooth
    flare with a staircase of short constant-width segments, each sized by
    _circular_ease() so the boundary is a circular sweep rather than a
    straight-line ramp with sharp bends at either end. ``widening=True`` eases
    from single-lane width up to the fork (the narrow end is s_lo);
    ``widening=False`` eases back down (the narrow end is s_hi). The divider's
    height eases with its width, so it reads as a low, pointed gore nose that
    grows into the full wall.
    """
    length = s_hi - s_lo
    n = max(math.ceil(length / FORK_TAPER_STEP), 1)
    target_inner = track.FORK_LANE_OFFSET - track.FORK_LANE_HALF_WIDTH
    target_outer_delta = track.FORK_LANE_OFFSET + track.FORK_LANE_HALF_WIDTH - W
    g = []
    for i in range(n):
        seg_lo = s_lo + length * i / n
        seg_hi = s_lo + length * (i + 1) / n
        s_mid = (seg_lo + seg_hi) / 2
        u = (s_mid - s_lo) if widening else (s_hi - s_mid)
        inner = _circular_ease(u, length, target_inner)
        outer = W + _circular_ease(u, length, target_outer_delta)
        divider_half = max(_circular_ease(u, length, track.FORK_DIVIDER_HALF_WIDTH), 0.0015)
        x, y, heading = track.path_point(s_mid)
        nx, ny = -math.sin(heading), math.cos(heading)
        rotation = f'euler="0 0 {heading:.4f}"'
        half_len = (seg_hi - seg_lo) / 2 + 0.003
        for side, sign in (("l", 1), ("r", -1)):
            lane_half = (outer - inner) / 2
            lane_center = (outer + inner) / 2
            cx, cy = x + nx * sign * lane_center, y + ny * sign * lane_center
            g.append(f'<geom name="{name}_road_{side}_{i}" type="box" '
                     f'size="{half_len:.4f} {lane_half:.4f} 0.0006" '
                     f'pos="{cx:.4f} {cy:.4f} {ROAD_Z:.5f}" {rotation} '
                     f'material="road" contype="0" conaffinity="0"/>')
            for edge, radius in (("outer", outer), ("inner", inner)):
                ex, ey = x + nx * sign * radius, y + ny * sign * radius
                g.append(f'<geom name="{name}_edge_{side}_{edge}_{i}" type="box" '
                         f'size="{half_len:.4f} {LINE_W} 0.0004" '
                         f'pos="{ex:.4f} {ey:.4f} {EDGE_Z}" {rotation} '
                         f'material="line" contype="0" conaffinity="0"/>')
        # bottom stays flush with the core divider's 0.02 clearance; only the
        # top grows, so the nose looks like it rises out of the road
        divider_half_z = max(_circular_ease(u, length, 0.05), 0.006)
        divider_z = 0.02 + divider_half_z
        g.append(f'<geom name="fork_divider_{name}_{i}" type="box" '
                 f'size="{half_len:.4f} {divider_half:.4f} {divider_half_z:.4f}" '
                 f'pos="{x:.4f} {y:.4f} {divider_z:.4f}" {rotation} {DARK_WALL}/>')
    return g


ARROW_CENTER = 0.085    # each arrow's own lateral anchor, offset from the panel centre
ARROW_HALF_SPAN = 0.028  # each arrow's own tip-to-tail half-extent -- kept well under
                          # ARROW_CENTER so the two arrows read as separate glyphs
                          # rather than merging into a single hourglass at the middle


def _arrow_chevron(name: str, pos: tuple[float, float, float], heading: float,
                    side: float, colour: str) -> list[str]:
    """A two-stroke chevron ("<" or ">") lying flat on a sign panel facing the road.

    ``side`` is +1 for the left half of the panel (arrow points further left)
    or -1 for the right half (points further right) -- each arrow lives
    entirely within its own half so the two never overlap into an hourglass.
    Both strokes share a tip and fan back along the vertical axis, built from
    explicit local axes (as with checkerboard_gate) since the strokes sit at
    an angle within the panel plane rather than along a world axis.
    """
    sx, sy, sz = pos
    tx, ty = math.cos(heading), math.sin(heading)      # panel normal (thickness axis)
    nx, ny = -math.sin(heading), math.cos(heading)     # panel lateral axis (left = +)
    head_angle = math.radians(35)
    half_len = ARROW_HALF_SPAN / math.cos(head_angle)  # stroke half-length
    tip_lat = side * (ARROW_CENTER + ARROW_HALF_SPAN)
    depth = 0.007        # stroke half-thickness
    half_w = 0.009       # stroke half-width
    # panel board is 0.01 half-thick (see fork_sign); stand the stroke clear
    # of its UPSTREAM face (-tangent), the one an approaching car sees
    face_offset = -(0.01 + depth + 0.003)
    g = []
    for i, updown in enumerate((1.0, -1.0)):
        ya = (
            side * math.cos(head_angle) * nx,
            side * math.cos(head_angle) * ny,
            updown * math.sin(head_angle),
        )
        xa = (tx, ty, 0.0)
        za = (
            xa[1] * ya[2] - xa[2] * ya[1],
            xa[2] * ya[0] - xa[0] * ya[2],
            xa[0] * ya[1] - xa[1] * ya[0],
        )
        qw, qx, qy, qz = _quat_from_axes(xa, ya, za)
        tip = (
            sx + nx * tip_lat + tx * face_offset,
            sy + ny * tip_lat + ty * face_offset,
            sz,
        )
        cx = tip[0] + ya[0] * half_len
        cy = tip[1] + ya[1] * half_len
        cz = tip[2] + ya[2] * half_len
        g.append(
            f'<geom name="{name}_{i}" type="box" size="{depth:.4f} {half_len:.4f} {half_w:.4f}" '
            f'pos="{cx:.4f} {cy:.4f} {cz:.4f}" '
            f'quat="{qw:.5f} {qx:.5f} {qy:.5f} {qz:.5f}" {colour} '
            f'contype="0" conaffinity="0"/>'
        )
    return g


def fork_sign(s: float) -> list[str]:
    """Roadside post with a left and a right arrow, both dim by default.

    env.py picks a random direction each attempt and brightens the matching
    pair of geoms (named fork_sign_left_0/1 and fork_sign_right_0/1) the same
    way it brightens the active traffic-light lamp.
    """
    px, py, heading = offset_pose(s, track.ROAD_HALF_WIDTH + 0.15)
    g = [
        f'<geom name="fork_sign_post" type="cylinder" size="0.012 0.16" '
        f'pos="{px:.4f} {py:.4f} 0.18" {SIGN_POST} contype="0" conaffinity="0"/>',
        f'<geom name="fork_sign_board" type="box" size="0.01 0.14 0.09" '
        f'pos="{px:.4f} {py:.4f} 0.32" euler="0 0 {heading:.4f}" {SIGN_BOARD} '
        f'contype="0" conaffinity="0"/>',
    ]
    g += _arrow_chevron("fork_sign_left", (px, py, 0.32), heading, 1.0, FORK_DIM)
    g += _arrow_chevron("fork_sign_right", (px, py, 0.32), heading, -1.0, FORK_DIM)
    return g


def build() -> str:
    body = []
    body += road_and_edges()

    # start/finish line: a white bar across the full road width
    sx, sy, start_heading = track.path_point(track.START_S)
    body.append(f'<geom name="start_line" type="box" size="0.02 {W} 0.0004" '
                f'pos="{sx:.4f} {sy:.4f} {EDGE_Z + 0.0004}" '
                f'euler="0 0 {start_heading:.4f}" material="line" contype="0" conaffinity="0"/>')

    body += fork_sign(track.FORK_SIGN_S)
    body += fork_taper("taperin", FORK_FULL[0], track.FORK_S[0], True)
    body += fork_lanes("fork", track.FORK_S)
    body += fork_taper("taperout", track.FORK_S[1], FORK_FULL[1], False)

    # Choke point, aligned to its local centerline tangent.
    ch_lo, ch_hi = track.CHOKE
    ch_half = track.s_delta(ch_hi, ch_lo) / 2
    cx, cy, ch_heading = track.path_point(ch_lo + ch_half)
    nx, ny = -math.sin(ch_heading), math.cos(ch_heading)
    gap = track.CHOKE_GAP_HALF
    for side, sign in (("l", 1), ("r", -1)):
        wx, wy = cx + nx * sign * (gap + 0.03), cy + ny * sign * (gap + 0.03)
        body.append(f'<geom name="choke_wall_{side}" type="box" '
                    f'size="{ch_half} 0.03 0.05" '
                    f'pos="{wx:.4f} {wy:.4f} 0.07" euler="0 0 {ch_heading:.4f}" {DARK_WALL}/>')

    body += checkerboard_gate("gate", track.GATE_S)
    body += shine_light()
    body += cross_hatch("hatch", track.HATCH)
    body += tunnel("tunnel2", track.TUNNEL_2)

    # Mocap bodies: dynamic obstacle and tunnel #2 obstacle.
    # Positions are placeholders — env.py (re)places them every reset.
    dx, dy, bicycle_heading = track.path_point(track.DYN_OBSTACLE_S)
    bicycle_heading += track.BICYCLE_YAW_OFFSET
    body.append(f'''<body name="dyn_obstacle" mocap="true" pos="{dx:.3f} {dy:.3f} 0.02"
            quat="{math.cos(bicycle_heading / 2):.5f} 0 0 {math.sin(bicycle_heading / 2):.5f}">
      <geom name="dyn_obstacle_rear_wheel" type="cylinder" size="0.045 0.008"
            pos="-0.07 0 0.045" zaxis="0 1 0" rgba="0.06 0.06 0.06 1"/>
      <geom name="dyn_obstacle_front_wheel" type="cylinder" size="0.045 0.008"
            pos="0.07 0 0.045" zaxis="0 1 0" rgba="0.06 0.06 0.06 1"/>
      <geom name="dyn_obstacle_frame_rear" type="capsule" size="0.008"
            fromto="-0.07 0 0.045 -0.02 0 0.115" rgba="0.85 0.20 0.08 1"/>
      <geom name="dyn_obstacle_frame_front" type="capsule" size="0.008"
            fromto="0.07 0 0.045 -0.02 0 0.115" rgba="0.85 0.20 0.08 1"/>
      <geom name="dyn_obstacle_frame_base" type="capsule" size="0.008"
            fromto="-0.07 0 0.045 0.07 0 0.045" rgba="0.85 0.20 0.08 1"/>
      <geom name="dyn_obstacle_seat" type="box" size="0.025 0.018 0.006"
            pos="-0.025 0 0.125" rgba="0.08 0.08 0.08 1"/>
      <geom name="dyn_obstacle_handlebar" type="box" size="0.01 0.055 0.006"
            pos="0.055 0 0.125" rgba="0.12 0.12 0.12 1"/>
    </body>''')

    ox, oy, _ = track.path_point(track.T2_OBSTACLE_S)
    body.append(f'''<body name="t2_obstacle" mocap="true" pos="{ox:.3f} {oy:.3f} 0.09">
      <geom name="t2_obstacle_box" type="box" size="0.055 0.055 0.07" rgba="0.3 0.55 0.8 1"/>
    </body>''')

    # Roadside traffic signal. The lenses are visual-only collision-wise;
    # env.py switches their emission/colour and judges the stop-line crossing.
    lx, ly, light_heading = offset_pose(track.TRAFFIC_LIGHT_S, W + 0.12)
    body.append(
        f'<geom name="traffic_pole" type="cylinder" size="0.012 0.14" '
        f'pos="{lx:.4f} {ly:.4f} 0.16" rgba="0.15 0.15 0.15 1" '
        f'contype="0" conaffinity="0"/>'
    )
    body.append(
        f'<geom name="traffic_housing" type="box" size="0.025 0.035 0.09" '
        f'pos="{lx:.4f} {ly:.4f} 0.36" euler="0 0 {light_heading:.4f}" '
        f'rgba="0.08 0.08 0.08 1" contype="0" conaffinity="0"/>'
    )
    # Each lamp also carries a real <light> (off by default; env.py drives
    # both its geom colour and its light intensity together), so the active
    # colour actually casts a glow on the pole and road below it instead of
    # just being a flat-shaded sphere.
    nx, ny = -math.sin(light_heading), math.cos(light_heading)   # left normal
    lamp_aim = (-nx, -ny, -0.35)                                 # back toward the road
    # the housing's road-facing half-width is 0.035; push each lens out past
    # that face (radius 0.018 alone would sit entirely inside the box, so
    # nothing would ever be visible no matter the colour) so it reads as a
    # lit lens standing proud of the housing, facing oncoming traffic
    lens_x = lx - nx * 0.048
    lens_y = ly - ny * 0.048
    for name, z, colour in (
        ("red", 0.415, "0.20 0.01 0.01 1"),
        ("yellow", 0.360, "0.18 0.12 0.01 1"),
        ("green", 0.305, "0.01 0.20 0.01 1"),
    ):
        body.append(
            f'<geom name="traffic_{name}" type="sphere" size="0.018" '
            f'pos="{lens_x:.4f} {lens_y:.4f} {z}" rgba="{colour}" '
            f'contype="0" conaffinity="0"/>'
        )
        body.append(
            f'<light name="traffic_{name}_light" pos="{lens_x:.4f} {lens_y:.4f} {z}" '
            f'dir="{lamp_aim[0]:.4f} {lamp_aim[1]:.4f} {lamp_aim[2]:.4f}" '
            f'diffuse="0 0 0" specular="0 0 0" attenuation="1 0 6" '
            f'cutoff="60" castshadow="false"/>'
        )

    stop_x, stop_y, stop_heading = track.path_point(track.TRAFFIC_STOP_S)
    body.append(
        f'<geom name="traffic_stop_line" type="box" size="0.018 {W} 0.0005" '
        f'pos="{stop_x:.4f} {stop_y:.4f} {EDGE_Z + 0.0005}" '
        f'euler="0 0 {stop_heading:.4f}" rgba="0.95 0.25 0.18 1" '
        f'contype="0" conaffinity="0"/>'
    )

    floor_cx = (track.MIN_X + track.MAX_X) / 2
    floor_cy = (track.MIN_Y + track.MAX_Y) / 2
    floor_hx = (track.MAX_X - track.MIN_X) / 2
    floor_hy = (track.MAX_Y - track.MIN_Y) / 2

    worldbody = "\n    ".join(body)

    return f'''<!--
  The EI Robotics Challenge - lane-keeping competition track (GENERATED FILE).

  Edit challenge/track.py or challenge/generate_track.py and rerun
      python -m challenge.generate_track
  instead of editing this file by hand.

  Diagram-shaped multi-turn loop. A paved road ~0.36 m wide on dark ground,
  with white edge lines and no center line: the car stays BETWEEN the edges.
  Sections: lane keeping/choke -> dynamic obstacle -> checkerboard gate/speed ->
  sheen straight -> winding return -> tunnel #2 -> traffic light -> finish.
-->
<mujoco model="challenge">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>

  <visual>
    <!-- offscreen framebuffer: must be >= the largest render requested.
         800x450 onboard camera; view_camera.py's chase panel is 600x450. -->
    <global offwidth="800" offheight="480"/>
    <!-- weak headlight so the dark tunnel is genuinely dark on camera -->
    <headlight diffuse="0.12 0.12 0.12" ambient="0.22 0.22 0.22"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.65 0.9" rgb2="0.9 0.95 1" width="256" height="256"/>
    <!-- dark ground surrounding the road, so the paved strip stands out -->
    <texture type="2d" name="carpet" builtin="checker" rgb1="0.09 0.10 0.09" rgb2="0.11 0.12 0.11"
             width="512" height="512"/>
    <material name="floor" texture="carpet" texrepeat="16 16" texuniform="true" reflectance="0" specular="0.1"/>
    <!-- paved road surface: mid gray, clearly brighter than the ground -->
    <material name="road" rgba="0.40 0.40 0.43 1" reflectance="0.05" specular="0.1"/>
    <material name="line" rgba="0.93 0.93 0.93 1"/>
  </asset>

  <worldbody>
    <light directional="true" pos="3 -4 6" dir="-0.35 0.45 -0.85" diffuse="0.85 0.85 0.8" castshadow="true"/>

    <geom name="ground" type="plane" size="0 0 1" rgba="0.25 0.32 0.25 1"/>
    <geom name="floor" type="box" size="{floor_hx:.2f} {floor_hy:.2f} 0.01"
          pos="{floor_cx:.2f} {floor_cy:.2f} 0.01" material="floor"/>

    {worldbody}

    <include file="car.xml"/>
  </worldbody>

  <contact>
    <exclude body1="car" body2="fl_wheel"/>
    <exclude body1="car" body2="fr_wheel"/>
    <exclude body1="car" body2="rl_wheel"/>
    <exclude body1="car" body2="rr_wheel"/>
  </contact>

  <equality>
    <!-- couple the two steering knuckles so one servo steers both wheels -->
    <joint joint1="fr_steer" joint2="fl_steer" polycoef="0 1 0 0 0"/>
  </equality>

  <actuator>
    <position name="steer" joint="fl_steer" kp="1.5" forcerange="-0.8 0.8" ctrlrange="-0.55 0.55"/>
    <velocity name="drive_l" joint="rl_roll" kv="0.12" forcerange="-0.3 0.3" ctrlrange="-70 70"/>
    <velocity name="drive_r" joint="rr_roll" kv="0.12" forcerange="-0.3 0.3" ctrlrange="-70 70"/>
  </actuator>

  <sensor>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>
    <jointpos name="steer_angle" joint="fl_steer"/>
    <jointvel name="wheel_speed_l" joint="rl_roll"/>
    <jointvel name="wheel_speed_r" joint="rr_roll"/>
    <framepos name="car_pos" objtype="xbody" objname="car"/>
    <framequat name="car_quat" objtype="xbody" objname="car"/>
  </sensor>
</mujoco>
'''


def main():
    out = Path(__file__).parent / "assets" / "challenge.xml"
    out.write_text(build())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
