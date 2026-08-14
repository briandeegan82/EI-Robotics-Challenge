"""ChallengeEnv: MuJoCo simulation of the EI Robotics Challenge.

The API follows the Gymnasium convention (``reset`` / ``step``) but has no
dependency on Gymnasium — plain numpy in, numpy out — so it works equally well
for hand-written controllers and RL training loops.

Typical use::

    from challenge import ChallengeEnv

    env = ChallengeEnv(render_mode="human")
    obs, info = env.reset(seed=0)
    while True:
        action = my_controller(obs, info)   # [steer, throttle], each in [-1, 1]
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print(env.score.summary())

An attempt ends when the car completes one full lap.

Action (2 floats, each in [-1, 1]):
    action[0]  steering: +1 = full left, -1 = full right (max ±0.55 rad)
    action[1]  throttle: +1 ≈ 2.2 m/s forward, negative reverses/brakes

Observation (8 floats) — only what the real robot can sense on-board
(depth sensors are prohibited by the rules, so there are none):
    obs[0]  forward speed estimated from wheel encoders (m/s)
    obs[1]  longitudinal acceleration (m/s^2, body frame)
    obs[2]  lateral acceleration (m/s^2, body frame)
    obs[3]  yaw rate (rad/s)
    obs[4]  steering angle (rad)
    obs[5]  left rear wheel angular velocity (rad/s)
    obs[6]  right rear wheel angular velocity (rad/s)
    obs[7]  elapsed time (s)

The PRIMARY sensor is the camera: ``env.camera_image()`` returns the onboard
RGB frame for line following and obstacle detection.

info dict:
    info["score"]       current points (see scoring.py)
    info["events"]      scoring events so far
    info["lap_time"]    adjusted lap time once finished (raw time + 1 s per
                        penalty point), else None
    info["privileged"]  ground-truth state the real robot will NOT have
                        (pose, arc length s, lateral offset, obstacle info).
                        Great for getting started — wean yourself off it and
                        use the camera before the real competition.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import mujoco
import numpy as np

from . import track
from .scoring import ScoreKeeper

_ASSETS = Path(__file__).parent / "assets"

MAX_STEER = 0.55       # rad
MAX_WHEEL_SPEED = 70   # rad/s (~2.2 m/s with 0.032 m wheels)
WHEEL_RADIUS = 0.032
CAR_Z = 0.052
CONTROL_HZ = 50
LAP_TIME_LIMIT = 120.0   # s to finish the lap
# The signal starts red and only turns (permanently) green once the car has
# sat stopped at the line for TRAFFIC_WAIT_SECONDS -- every competitor pays
# exactly the same toll, rather than some getting lucky with a pre-timed
# cycle and others eating a near-full-cycle wait. TRAFFIC_STOP_SPEED reuses
# the stall detector's "basically stopped" threshold, and TRAFFIC_WAIT_ZONE
# is how close to the line (in arc-length) the stop has to be to count.
TRAFFIC_WAIT_SECONDS = 5.0
TRAFFIC_STOP_SPEED = 0.03
TRAFFIC_WAIT_ZONE = (-0.05, 0.5)

# dynamic bicycle behaviour
DYN_MOVE_SPAN = 0.22     # moving mode: slides across +/- this lateral range
DYN_MOVE_SPEED = 0.12    # m/s

T2_OBSTACLE_LAT = 0.14   # closer to one wall, leaving a wider route on the free side

# fork sign: bright/dim lamp colours, same trick as the traffic light lenses
FORK_BRIGHT = [1.0, 0.75, 0.05, 1.0]
FORK_DIM = [0.20, 0.17, 0.06, 1.0]
# lateral offset past which the car counts as having committed to a lane (the
# divider wall forces this well before the fork's own lane centre at
# track.FORK_LANE_OFFSET)
FORK_COMMIT_LAT = 0.05

# Off-track checks are suspended around obstacles/the fork because bypassing
# them requires leaving the (single-lane) line.
_NO_LANE_CHECK = (
    (track.DYN_OBSTACLE_S - 0.9, track.DYN_OBSTACLE_S + 0.9),
    (track.TUNNEL_2[0] - 0.5, track.TUNNEL_2[1] + 0.3),
    (track.FORK_S[0] - 1.4, track.FORK_S[1] + 0.9),
)


class ChallengeEnv:
    """MuJoCo simulation of the EI Robotics Challenge course."""

    def __init__(self, render_mode: str | None = None):
        """render_mode: None (headless) or "human" (opens the MuJoCo viewer)."""
        self.model = mujoco.MjModel.from_xml_path(str(_ASSETS / "challenge.xml"))
        # MuJoCo's viewer handles Backspace by restoring model.qpos0 directly.
        # Keep that built-in reset pose aligned with the environment start.
        x0, y0, heading0 = track.path_point(track.START_S - 0.25)
        self.model.qpos0[0:3] = [x0, y0, CAR_Z]
        self.model.qpos0[3:7] = [
            np.cos(heading0 / 2),
            0,
            0,
            np.sin(heading0 / 2),
        ]
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._viewer = None
        self._renderer = None
        self._rng = np.random.default_rng()

        self._frame_skip = int(round(1 / (CONTROL_HZ * self.model.opt.timestep)))

        self._sensor = {self.model.sensor(i).name: i for i in range(self.model.nsensor)}
        self._car_body = self.model.body("car").id
        self._dyn_mocap = self.model.body("dyn_obstacle").mocapid[0]
        self._t2_mocap = self.model.body("t2_obstacle").mocapid[0]

        def gid(name):
            return self.model.geom(name).id

        self._car_geoms = {gid(n) for n in ("chassis", "board", "camera_mount", "camera_body",
                                            "fl_tire", "fr_tire", "rl_tire", "rr_tire")}
        self._traffic_geoms = {
            "red": gid("traffic_red"),
            "yellow": gid("traffic_yellow"),
            "green": gid("traffic_green"),
        }
        self._traffic_lights = {
            "red": self.model.light("traffic_red_light").id,
            "yellow": self.model.light("traffic_yellow_light").id,
            "green": self.model.light("traffic_green_light").id,
        }
        self._fork_geoms = {
            "left": [gid("fork_sign_left_0"), gid("fork_sign_left_1")],
            "right": [gid("fork_sign_right_0"), gid("fork_sign_right_1")],
        }
        self._hazard_geoms = {
            gid(name): "obstacle"
            for name in (
                "dyn_obstacle_rear_wheel",
                "dyn_obstacle_front_wheel",
                "dyn_obstacle_frame_rear",
                "dyn_obstacle_frame_front",
                "dyn_obstacle_frame_base",
                "dyn_obstacle_seat",
                "dyn_obstacle_handlebar",
                "t2_obstacle_box",
            )
        }
        for i in range(self.model.ngeom):
            name = self.model.geom(i).name
            if name.startswith(("choke_wall", "gate_", "tunnel2_", "fork_divider")):
                self._hazard_geoms[i] = "wall"

        self.reset(seed=0)

    # ------------------------------------------------------------------ API

    def reset(self, seed: int | None = None):
        """Reset the attempt. Returns (obs, info)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        # staging: within 30 cm of the start line, small placement noise
        s0 = track.START_S - 0.25
        x, y, heading = track.path_point(s0)
        self.data.qpos[0] = x + self._rng.uniform(-0.02, 0.02)
        self.data.qpos[1] = y + self._rng.uniform(-0.03, 0.03)
        self.data.qpos[3:7] = [
            np.cos(heading / 2),
            0,
            0,
            np.sin(heading / 2),
        ]

        # dynamic bicycle always crosses; only its starting phase is randomized
        self._dyn_moving = True
        self._dyn_phase = self._rng.uniform(0, 2 * np.pi)

        # tunnel #2 obstacle: random side of the line
        self.t2_side = int(self._rng.choice([-1, 1]))
        ox, oy, heading = track.path_point(track.T2_OBSTACLE_S)
        nx, ny = -np.sin(heading), np.cos(heading)
        self.data.mocap_pos[self._t2_mocap] = [
            ox + nx * self.t2_side * T2_OBSTACLE_LAT,
            oy + ny * self.t2_side * T2_OBSTACLE_LAT,
            0.09,
        ]

        # fork sign: randomize which lane is required this attempt, and light
        # up the matching arrow (the other stays dim)
        self.fork_direction = "left" if self._rng.integers(0, 2) == 0 else "right"
        for direction, geom_ids in self._fork_geoms.items():
            colour = FORK_BRIGHT if direction == self.fork_direction else FORK_DIM
            for geom_id in geom_ids:
                self.model.geom_rgba[geom_id] = colour
        self._fork_side = None

        # traffic light: always starts red; see step()'s wait-timer logic for
        # when it switches to green.
        self._traffic_state = "red"
        self._traffic_wait_since = None

        self.score = ScoreKeeper()
        self.lap_time = None
        self._progress = 0.0
        self._prev_s = track.frenet(self.data.qpos[0], self.data.qpos[1])[0]
        self._off_track = False
        self._moved = False
        self._stall_since = None
        self._colliding = set()
        self._hit = set()          # hazards already penalized this attempt
        self._traffic_violated = False
        self._done = False

        self._update_world(0.0)
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), self._info()

    def step(self, action):
        """Advance one control step (1/50 s). Returns (obs, reward, terminated, truncated, info)."""
        if self._done:
            raise RuntimeError("Attempt is over — call reset() first.")
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.data.ctrl[0] = action[0] * MAX_STEER
        self.data.ctrl[1] = action[1] * MAX_WHEEL_SPEED
        self.data.ctrl[2] = action[1] * MAX_WHEEL_SPEED

        step_hazards = set()
        for _ in range(self._frame_skip):
            self._update_world(self.data.time)
            mujoco.mj_step(self.model, self.data)
            step_hazards |= self._contact_hazards()

        t = self.data.time
        x, y = self.data.qpos[0], self.data.qpos[1]
        s, lat = track.frenet(x, y)
        prev_s = self._prev_s
        ds = np.clip(track.s_delta(s, prev_s), -0.5, 0.5)
        self._progress += ds
        self._prev_s = s
        speed = self._obs()[0]

        terminated = truncated = False

        # ---- traffic light ---------------------------------------------
        # Red until the car has sat stopped at the line for
        # TRAFFIC_WAIT_SECONDS, then permanently green -- a fixed toll every
        # competitor pays, instead of a pre-timed cycle some get lucky on.
        if self._traffic_state == "red":
            light_gap = track.s_delta(track.TRAFFIC_STOP_S, s)
            waiting_here = TRAFFIC_WAIT_ZONE[0] < light_gap < TRAFFIC_WAIT_ZONE[1]
            if waiting_here and speed < TRAFFIC_STOP_SPEED:
                if self._traffic_wait_since is None:
                    self._traffic_wait_since = t
                elif t - self._traffic_wait_since >= TRAFFIC_WAIT_SECONDS:
                    self._traffic_state = "green"
            else:
                self._traffic_wait_since = None

        crossed_stop_line = (
            ds > 0
            and 0 <= track.s_delta(track.TRAFFIC_STOP_S, prev_s) <= ds + 1e-6
        )
        if (crossed_stop_line and not self._traffic_violated
                and self._traffic_state == "red"):
            self.score.traffic_light_violation(t)
            self._traffic_violated = True

        # ---- lane fork ---------------------------------------------------
        # The divider physically forces a side well before FORK_S[1], so by
        # the time the car leaves the fork it has always committed to one
        # lane; compare that lane against the sign shown this attempt.
        if track.in_range(s, track.FORK_S) and self._fork_side is None and abs(lat) > FORK_COMMIT_LAT:
            self._fork_side = "left" if lat > 0 else "right"
        crossed_fork_end = (
            ds > 0
            and 0 <= track.s_delta(track.FORK_S[1], prev_s) <= ds + 1e-6
        )
        if crossed_fork_end and self._fork_side != self.fork_direction:
            self.score.wrong_lane(t)

        # ---- collisions ------------------------------------------------
        for hazard in step_hazards - self._colliding:
            if hazard == "wall":
                # official rules: collision with track structure forfeits
                self.score.forfeit("wall_collision", t)
                terminated = True
            elif hazard == "obstacle" and "obstacle" not in self._hit:
                self.score.obstacle_contact(t)
                self._hit.add("obstacle")
        self._colliding = step_hazards

        # ---- lane keeping ----------------------------------------------
        checked = not any(track.in_range(s, r) for r in _NO_LANE_CHECK)
        if checked and abs(lat) > track.OFFTRACK_MINOR and not self._off_track:
            self.score.off_track_minor(t)
            self._off_track = True
        elif abs(lat) < track.OFFTRACK_MINOR - 0.03:
            self._off_track = False

        # ---- stalling (official: hesitation > 2 s) ---------------------
        light_gap = track.s_delta(track.TRAFFIC_STOP_S, s)
        waiting_at_signal = (
            self._traffic_state == "red"
            and -0.05 < light_gap < 0.8
        )
        # Suppress the stall penalty across the whole region where a controller
        # has to hold for the crossing bicycle — it can legitimately come to a
        # stop anywhere it first sees the bike blocking the road, which is a
        # wider band than the old [-0.8, -0.2] window. Gated on the bike
        # actually blocking (|lat| < 0.20) so it can't be abused to sit idle.
        waiting_for_bicycle = (
            self._dyn_moving
            and track.in_range(
                s,
                (track.DYN_OBSTACLE_S - 1.0, track.DYN_OBSTACLE_S + 0.2),
            )
            and abs(self._dyn_lat) < 0.20
        )
        if waiting_at_signal or waiting_for_bicycle:
            self._stall_since = None
        elif speed > 0.1:
            self._moved = True
            self._stall_since = None
        elif self._moved and speed < 0.03:
            if self._stall_since is None:
                self._stall_since = t
            elif t - self._stall_since > 2.0:
                self.score.stall(t)
                self._stall_since = t  # re-arm; repeated stalls repeat the penalty
        # ---- lap completion --------------------------------------------
        # Staging sits 0.25 m behind START_S, so a full loop of travel back to
        # the painted start/finish line is exactly TOTAL + 0.25 of progress.
        if self._progress >= track.TOTAL + 0.25:
            self.score.lap_complete(t)
            self.lap_time = self.score.lap_time
            terminated = True

        # ---- global termination ----------------------------------------
        if self._flipped():
            self.score.forfeit("flipped", t)
            terminated = True
        if t >= LAP_TIME_LIMIT:
            self.score.forfeit("lap_timeout", t)
            truncated = True

        reward = ds + self.score.consume_reward()
        self._done = terminated or truncated

        if self.render_mode == "human":
            self.render()

        return self._obs(), reward, terminated, truncated, self._info()

    def render(self):
        """Sync the interactive viewer (render_mode="human")."""
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._viewer.cam.trackbodyid = self._car_body
            self._viewer.cam.distance = 1.6
            self._viewer.cam.elevation = -30
            self._viewer.cam.azimuth = 0
        if self._viewer.is_running():
            self._viewer.sync()

    def camera_image(self, camera: str = "onboard", width: int = 800, height: int = 450):
        """Return an RGB image (H, W, 3 uint8) from the onboard camera.

        Defaults to 800x450 — 16:9, matching the Raspberry Pi Camera v3 aspect
        ratio. This is the competition's primary sensor.
        Requires OpenGL; on a headless machine set MUJOCO_GL=egl or osmesa.

        Note: the renderer is created once at the first call and reuses that
        size afterward, so pass the resolution you want on the first call.
        """
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._scene_option = mujoco.MjvOption()
        self._renderer.update_scene(self.data, camera=camera, scene_option=self._scene_option)
        return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            if self._viewer.is_running():
                self._viewer.close()
                # launch_passive's render loop runs on a daemon thread; close()
                # only requests an exit, it doesn't block until the thread has
                # actually torn down its GL context. If the process exits
                # first, that teardown can race the interpreter shutdown and
                # print a spurious (harmless) X/GLX error to stderr. Give it a
                # moment to finish on its own terms.
                deadline = time.monotonic() + 1.0
                while self._viewer.is_running() and time.monotonic() < deadline:
                    time.sleep(0.01)
        self._viewer = None
        gc.collect()  # drop the Handle's C++ side synchronously, here, not at
                       # an unpredictable later GC pass or interpreter exit
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------- course dynamics

    def _update_world(self, t: float):
        """Move dynamic course elements and update signal lamps."""
        ox, oy, heading = track.path_point(track.DYN_OBSTACLE_S)
        # slide back and forth across the track, sinusoidally
        period = 4 * DYN_MOVE_SPAN / DYN_MOVE_SPEED
        lat = DYN_MOVE_SPAN * np.sin(2 * np.pi * t / period + self._dyn_phase)
        nx, ny = -np.sin(heading), np.cos(heading)   # left normal
        self._dyn_lat = float(lat)
        self.data.mocap_pos[self._dyn_mocap] = [ox + nx * lat, oy + ny * lat, 0.02]
        bicycle_heading = heading + track.BICYCLE_YAW_OFFSET
        self.data.mocap_quat[self._dyn_mocap] = [
            np.cos(bicycle_heading / 2),
            0,
            0,
            np.sin(bicycle_heading / 2),
        ]
        self._update_traffic_light()

    def _update_traffic_light(self):
        # the amber lamp is a physical fixture (a real 3-light signal head)
        # but this course only ever drives it red<->green -- see step()'s
        # wait-timer logic for self._traffic_state.
        state = self._traffic_state
        colours = {
            "red": [1.0, 0.02, 0.02, 1.0] if state == "red" else [0.20, 0.01, 0.01, 1.0],
            "yellow": [1.0, 0.75, 0.02, 1.0] if state == "yellow" else [0.18, 0.12, 0.01, 1.0],
            "green": [0.02, 1.0, 0.02, 1.0] if state == "green" else [0.01, 0.20, 0.01, 1.0],
        }
        # only the active lamp actually lights up; the other two stay off
        # (rather than merely dim) so there's a genuine glow to switch, not
        # just a recoloured sphere
        light_colours = {
            "red": [2.5, 0.05, 0.05] if state == "red" else [0.0, 0.0, 0.0],
            "yellow": [2.2, 1.6, 0.05] if state == "yellow" else [0.0, 0.0, 0.0],
            "green": [0.05, 2.5, 0.05] if state == "green" else [0.0, 0.0, 0.0],
        }
        for name, geom_id in self._traffic_geoms.items():
            self.model.geom_rgba[geom_id] = colours[name]
        for name, light_id in self._traffic_lights.items():
            self.model.light_diffuse[light_id] = light_colours[name]
            self.model.light_specular[light_id] = light_colours[name]

    def _contact_hazards(self):
        out = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if pair & self._car_geoms:
                for g in pair - self._car_geoms:
                    if g in self._hazard_geoms:
                        out.add(self._hazard_geoms[g])
        return out

    def _flipped(self):
        return self.data.xmat[self._car_body].reshape(3, 3)[2, 2] < 0.2

    # -------------------------------------------------------------- readouts

    def _sensordata(self, name):
        i = self._sensor[name]
        adr = self.model.sensor_adr[i]
        dim = self.model.sensor_dim[i]
        return self.data.sensordata[adr:adr + dim].copy()

    def _obs(self):
        wl = self._sensordata("wheel_speed_l")[0]
        wr = self._sensordata("wheel_speed_r")[0]
        accel = self._sensordata("accel")
        gyro = self._sensordata("gyro")
        return np.array([
            (wl + wr) / 2 * WHEEL_RADIUS,   # encoder-based speed estimate
            accel[0], accel[1],
            gyro[2],
            self._sensordata("steer_angle")[0],
            wl, wr,
            self.data.time,
        ], dtype=np.float32)

    def _info(self):
        pos = self._sensordata("car_pos")
        qw, qx, qy, qz = self._sensordata("car_quat")
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        s, lat = track.frenet(pos[0], pos[1])
        return {
            "score": self.score.total(),
            "events": list(self.score.events),
            "time": self.data.time,
            "lap_time": self.lap_time,
            "privileged": {
                "car_pos": pos,
                "car_yaw": float(yaw),
                "s": float(s),
                "lateral": float(lat),
                "progress": float(self._progress),
                "dyn_obstacle": {"moving": self._dyn_moving, "lateral": self._dyn_lat},
                "t2_side": self.t2_side,
                "fork_direction": self.fork_direction,
                "traffic_light": {
                    "state": self._traffic_state,
                    "stop_s": track.TRAFFIC_STOP_S,
                },
            },
        }
