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

LINE_W = 0.0125          # white line half-width (~25 mm line)
LINE_Z = 0.0220          # line sits just above the floor and glare slab
FLOOR_TOP = 0.02

WALL = 'rgba="0.55 0.55 0.6 1"'
DARK_WALL = 'rgba="0.4 0.4 0.45 1"'


def line_segments() -> list[str]:
    """White line as long boxes on straights and short segments on curves."""
    g = []
    y = track.RADIUS
    g.append(f'<geom name="line_bottom" type="box" size="{track.STRAIGHT_HALF} {LINE_W} 0.0007" '
             f'pos="0 {-y} {LINE_Z}" material="line" contype="0" conaffinity="0"/>')
    g.append(f'<geom name="line_top" type="box" size="{track.STRAIGHT_HALF} {LINE_W} 0.0007" '
             f'pos="0 {y} {LINE_Z}" material="line" contype="0" conaffinity="0"/>')
    n = 60
    for curve, cx in (("r", track.STRAIGHT_HALF), ("l", -track.STRAIGHT_HALF)):
        for i in range(n):
            phi = -math.pi / 2 + math.pi * (i + 0.5) / n
            if curve == "l":
                phi += math.pi
            x = cx + track.RADIUS * math.cos(phi)
            yy = track.RADIUS * math.sin(phi)
            seg = track.RADIUS * math.pi / n / 2 + 0.004   # tiny overlap
            g.append(f'<geom name="line_{curve}{i}" type="box" '
                     f'size="{seg:.4f} {LINE_W} 0.0007" '
                     f'pos="{x:.4f} {yy:.4f} {LINE_Z}" euler="0 0 {phi + math.pi / 2:.4f}" '
                     f'material="line" contype="0" conaffinity="0"/>')
    return g


def tunnel(name: str, x_center: float, y_center: float, half_len: float,
           roof: str) -> list[str]:
    """Tunnel walls (interior width 0.56 m, height 0.22 m) plus a roof.

    roof="full" (dark, tunnel #2) or "split" (center skylight gap, tunnel #1).
    """
    g = []
    for side, sy in (("l", 1), ("r", -1)):
        g.append(f'<geom name="{name}_wall_{side}" type="box" '
                 f'size="{half_len} 0.015 0.1" '
                 f'pos="{x_center} {y_center + sy * 0.295} 0.12" {WALL}/>')
    if roof == "full":
        g.append(f'<geom name="{name}_roof" type="box" size="{half_len} 0.31 0.006" '
                 f'pos="{x_center} {y_center} 0.228" {DARK_WALL}/>')
    else:
        for side, sy in (("l", 1), ("r", -1)):
            g.append(f'<geom name="{name}_roof_{side}" type="box" '
                     f'size="{half_len} 0.09 0.006" '
                     f'pos="{x_center} {y_center + sy * 0.21} 0.228" {WALL}/>')
    return g


def build() -> str:
    R, y_bot, y_top = track.RADIUS, -track.RADIUS, track.RADIUS

    body = []
    body += line_segments()

    # start/finish line
    sx, sy, _ = track.path_point(track.START_S)
    body.append(f'<geom name="start_line" type="box" size="0.02 0.18 0.0008" '
                f'pos="{sx} {sy} {LINE_Z + 0.0002}" material="line" contype="0" conaffinity="0"/>')

    # choke point: tight corridor walls at the end of the bottom straight
    ch_lo, ch_hi = track.CHOKE
    cx = (-track.STRAIGHT_HALF + ch_lo + -track.STRAIGHT_HALF + ch_hi) / 2
    ch_half = (ch_hi - ch_lo) / 2
    for side, sy_ in (("l", 1), ("r", -1)):
        body.append(f'<geom name="choke_wall_{side}" type="box" '
                    f'size="{ch_half} 0.015 0.05" '
                    f'pos="{cx} {y_bot + sy_ * 0.125} 0.07" {WALL}/>')

    # tunnels (both on straights; centers derived from their s-ranges)
    t1x = track.STRAIGHT_HALF - ((track.TUNNEL_1[0] + track.TUNNEL_1[1]) / 2 - track.S_TOP)
    t1_half = (track.TUNNEL_1[1] - track.TUNNEL_1[0]) / 2
    body += tunnel("tunnel1", round(t1x, 3), y_top, t1_half, roof="split")

    t2x = -track.STRAIGHT_HALF + (track.TUNNEL_2[0] + track.TUNNEL_2[1]) / 2
    t2_half = (track.TUNNEL_2[1] - track.TUNNEL_2[0]) / 2
    body += tunnel("tunnel2", round(t2x, 3), y_bot, t2_half, roof="full")

    # mocap bodies: dynamic obstacle, tunnel #2 obstacle, stop cube.
    # Positions are placeholders — env.py (re)places them every reset.
    dx, dy, _ = track.path_point(track.DYN_OBSTACLE_S)
    body.append(f'''<body name="dyn_obstacle" mocap="true" pos="{dx:.3f} {dy:.3f} 0.11">
      <geom name="dyn_obstacle_box" type="box" size="0.06 0.06 0.09" rgba="0.85 0.35 0.1 1"/>
    </body>''')

    ox, oy, _ = track.path_point(track.T2_OBSTACLE_S)
    body.append(f'''<body name="t2_obstacle" mocap="true" pos="{ox:.3f} {oy + 0.14:.3f} 0.09">
      <geom name="t2_obstacle_box" type="box" size="0.055 0.055 0.07" rgba="0.3 0.55 0.8 1"/>
    </body>''')

    body.append('''<body name="stop_cube" mocap="true" pos="0 -5 -1">
      <geom name="stop_cube_box" type="box" size="0.1 0.1 0.1" rgba="0.95 0.95 0.95 1"/>
    </body>''')

    worldbody = "\n    ".join(body)

    return f'''<!--
  The Gauntlet - line-following competition track (GENERATED FILE).

  Edit gauntlet/track.py or gauntlet/generate_track.py and rerun
      python -m gauntlet.generate_track
  instead of editing this file by hand.

  Stadium loop, driven counter-clockwise. White ~25 mm line on dark floor.
  Sections: start -> lane keeping -> choke point -> curve -> tunnel #1 ->
  speed section (dynamic obstacle) -> high-glare curve -> tunnel #2 ->
  end zone (stop cube appears after the lap).
-->
<mujoco model="gauntlet">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>

  <visual>
    <global offwidth="640" offheight="480"/>
    <!-- weak headlight so the tunnels are genuinely dark on camera -->
    <headlight diffuse="0.12 0.12 0.12" ambient="0.22 0.22 0.22"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.65 0.9" rgb2="0.9 0.95 1" width="256" height="256"/>
    <texture type="2d" name="carpet" builtin="checker" rgb1="0.16 0.16 0.18" rgb2="0.18 0.18 0.2"
             width="512" height="512"/>
    <material name="floor" texture="carpet" texrepeat="16 16" texuniform="true" reflectance="0" specular="0.1"/>
    <material name="line" rgba="0.93 0.93 0.93 1"/>
    <!-- high-glare slab: bright + strongly reflective, washes out line contrast -->
    <material name="glare" rgba="0.8 0.8 0.85 1" reflectance="0.8" specular="1" shininess="1"/>
  </asset>

  <worldbody>
    <light directional="true" pos="3 -4 6" dir="-0.35 0.45 -0.85" diffuse="0.85 0.85 0.8" castshadow="true"/>
    <light name="tunnel1_light" pos="{t1x:.2f} {y_top} 0.21" dir="0 0 -1" diffuse="0.45 0.45 0.4" cutoff="70"/>

    <geom name="ground" type="plane" size="0 0 1" rgba="0.35 0.4 0.35 1"/>
    <!-- dark track floor, 7 x 4 m footprint -->
    <geom name="floor" type="box" size="3.5 2.0 0.01" pos="0 0 0.01" material="floor"/>
    <!-- high-glare area covering the left curve; ends before the tunnel #2
         entrance so the glare and low-light challenges stay distinct -->
    <geom name="glare_slab" type="box" size="0.675 2.0 0.0006" pos="-2.725 0 0.0206"
          material="glare" contype="0" conaffinity="0"/>

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
