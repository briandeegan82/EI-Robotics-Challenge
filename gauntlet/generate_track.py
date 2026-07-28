"""Generate gauntlet/assets/gauntlet.xml from the track geometry in track.py.

    python -m gauntlet.generate_track

Rerun after changing track.py (or the wall/tunnel parameters below). The
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
ROAD_STEP = 0.08             # centerline sweep resolution

WALL = 'rgba="0.55 0.55 0.6 1"'
# dark (below the road brightness) so it reads as "not road" to the camera
DARK_WALL = 'rgba="0.13 0.13 0.15 1"'


def road_and_edges() -> list[str]:
    """Sweep the paved strip and edge lines along the shared centerline."""
    g = []
    index = 0
    for segment in track.SEGMENTS:
        divisions = math.ceil(segment.length / ROAD_STEP) if segment.kind == "arc" else 1
        cuts = [segment.s0 + segment.length * i / divisions for i in range(divisions + 1)]
        if segment.kind == "line":
            for boundary in track.SHINE:
                if segment.s0 < boundary < segment.s0 + segment.length:
                    cuts.append(boundary)
            cuts.sort()
        for lo, hi in zip(cuts, cuts[1:]):
            s = (lo + hi) / 2
            x, y, heading = track.path_point(s)
            nx, ny = -math.sin(heading), math.cos(heading)
            material = "glare" if track.in_range(s, track.SHINE) else "road"
            half_len = (hi - lo) / 2 + 0.004
            g.append(
                f'<geom name="road_{index}" type="box" size="{half_len:.4f} {W} 0.0006" '
                f'pos="{x:.4f} {y:.4f} {ROAD_Z}" euler="0 0 {heading:.4f}" '
                f'material="{material}" contype="0" conaffinity="0"/>'
            )
            for side, sign in (("l", 1), ("r", -1)):
                ex, ey = x + nx * sign * W, y + ny * sign * W
                g.append(
                    f'<geom name="edge_{index}_{side}" type="box" '
                    f'size="{half_len:.4f} {LINE_W} 0.0004" '
                    f'pos="{ex:.4f} {ey:.4f} {EDGE_Z}" euler="0 0 {heading:.4f}" '
                    f'material="line" contype="0" conaffinity="0"/>'
                )
            index += 1
    return g


def tunnel(name: str, s_range: tuple[float, float], roof: str) -> list[str]:
    """Tunnel walls (interior width 0.56 m, height 0.22 m) plus a roof.

    roof="full" (dark, tunnel #2) or "split" (center skylight gap, tunnel #1).
    """
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
                 f'pos="{x:.4f} {y:.4f} 0.12" {rotation} {WALL}/>')
    if roof == "full":
        g.append(f'<geom name="{name}_roof" type="box" size="{half_len} 0.31 0.006" '
                 f'pos="{x_center:.4f} {y_center:.4f} 0.228" {rotation} {DARK_WALL}/>')
    else:
        for side, sy in (("l", 1), ("r", -1)):
            x, y = x_center + nx * sy * 0.21, y_center + ny * sy * 0.21
            g.append(f'<geom name="{name}_roof_{side}" type="box" '
                     f'size="{half_len} 0.09 0.006" '
                     f'pos="{x:.4f} {y:.4f} 0.228" {rotation} {WALL}/>')
    return g


def offset_pose(s: float, lateral: float = 0.0) -> tuple[float, float, float]:
    """Return a path pose shifted along its left normal."""
    x, y, heading = track.path_point(s)
    return (
        x - math.sin(heading) * lateral,
        y + math.cos(heading) * lateral,
        heading,
    )


def build() -> str:
    body = []
    body += road_and_edges()

    # start/finish line: a white bar across the full road width
    sx, sy, start_heading = track.path_point(track.START_S)
    body.append(f'<geom name="start_line" type="box" size="0.02 {W} 0.0004" '
                f'pos="{sx:.4f} {sy:.4f} {EDGE_Z + 0.0004}" '
                f'euler="0 0 {start_heading:.4f}" material="line" contype="0" conaffinity="0"/>')

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

    body += tunnel("tunnel1", track.TUNNEL_1, roof="split")
    body += tunnel("tunnel2", track.TUNNEL_2, roof="full")

    # mocap bodies: dynamic obstacle, tunnel #2 obstacle, stop cube.
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

    body.append(f'''<body name="stop_cube" mocap="true" pos="0 -5 -1">
      <geom name="stop_cube_box" type="box" size="{track.CUBE_HALF_LENGTH:.4f} {track.CUBE_HALF_WIDTH:.4f} {track.CUBE_HALF_HEIGHT:.4f}" rgba="0.95 0.95 0.95 1"/>
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
    for name, z, colour in (
        ("red", 0.415, "0.20 0.01 0.01 1"),
        ("yellow", 0.360, "0.18 0.12 0.01 1"),
        ("green", 0.305, "0.01 0.20 0.01 1"),
    ):
        body.append(
            f'<geom name="traffic_{name}" type="sphere" size="0.018" '
            f'pos="{lx:.4f} {ly:.4f} {z}" rgba="{colour}" '
            f'contype="0" conaffinity="0"/>'
        )

    stop_x, stop_y, stop_heading = track.path_point(track.TRAFFIC_STOP_S)
    body.append(
        f'<geom name="traffic_stop_line" type="box" size="0.018 {W} 0.0005" '
        f'pos="{stop_x:.4f} {stop_y:.4f} {EDGE_Z + 0.0005}" '
        f'euler="0 0 {stop_heading:.4f}" rgba="0.95 0.25 0.18 1" '
        f'contype="0" conaffinity="0"/>'
    )

    t1_mid = track.TUNNEL_1[0] + track.s_delta(track.TUNNEL_1[1], track.TUNNEL_1[0]) / 2
    t1x, t1y, _ = track.path_point(t1_mid)
    floor_cx = (track.MIN_X + track.MAX_X) / 2
    floor_cy = (track.MIN_Y + track.MAX_Y) / 2
    floor_hx = (track.MAX_X - track.MIN_X) / 2
    floor_hy = (track.MAX_Y - track.MIN_Y) / 2

    worldbody = "\n    ".join(body)

    return f'''<!--
  The Gauntlet - lane-keeping competition track (GENERATED FILE).

  Edit gauntlet/track.py or gauntlet/generate_track.py and rerun
      python -m gauntlet.generate_track
  instead of editing this file by hand.

  Diagram-shaped multi-turn loop. A paved road ~0.36 m wide on dark ground,
  with white edge lines and no center line: the car stays BETWEEN the edges.
  Sections: lane keeping/choke -> dynamic obstacle -> tunnel #1/speed ->
  glare -> winding return -> tunnel #2 -> traffic light ->
  finish and end-zone stop.
-->
<mujoco model="gauntlet">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>

  <visual>
    <!-- offscreen framebuffer: must be >= the largest render requested.
         800x450 onboard camera; view_camera.py's chase panel is 600x450. -->
    <global offwidth="800" offheight="480"/>
    <!-- weak headlight so the tunnels are genuinely dark on camera -->
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
    <!-- high-glare road: bright + strongly reflective, washes out edge contrast -->
    <material name="glare" rgba="0.82 0.82 0.86 1" reflectance="0.8" specular="1" shininess="1"/>
  </asset>

  <worldbody>
    <light directional="true" pos="3 -4 6" dir="-0.35 0.45 -0.85" diffuse="0.85 0.85 0.8" castshadow="true"/>
    <light name="tunnel1_light" pos="{t1x:.2f} {t1y:.2f} 0.21" dir="0 0 -1" diffuse="0.45 0.45 0.4" cutoff="70"/>

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
    out = Path(__file__).parent / "assets" / "gauntlet.xml"
    out.write_text(build())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
