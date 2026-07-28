"""GauntletEnv: MuJoCo simulation of the Gauntlet line-following challenge.

The API follows the Gymnasium convention (``reset`` / ``step``) but has no
dependency on Gymnasium — plain numpy in, numpy out — so it works equally well
for hand-written controllers and RL training loops.

Typical use::

    from gauntlet import GauntletEnv

    env = GauntletEnv(render_mode="human")
    obs, info = env.reset(seed=0)
    while True:
        action = my_controller(obs, info)   # [steer, throttle], each in [-1, 1]
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print(env.score.summary())

An attempt = one full lap plus the end-zone stop: after the car crosses the
finish line, the white stop cube is placed on the track and the car must stop
within 10 cm of it without touching it (30 s allowed), exactly as in the
official rules.

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
RGB frame — that's what line following, obstacle detection, and the stop cube
are meant to be driven by.

info dict:
    info["score"]       current points (see scoring.py)
    info["events"]      scoring events so far
    info["phase"]       "lap" or "stop"
    info["lap_time"]    lap time once the finish line is crossed, else None
    info["privileged"]  ground-truth state the real robot will NOT have
                        (pose, arc length s, lateral offset, obstacle info,
                        cube gap). Great for getting started — wean yourself
                        off it and use the camera before the real competition.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from . import track
from .scoring import ScoreKeeper

_ASSETS = Path(__file__).parent / "assets"

MAX_STEER = 0.55       # rad
MAX_WHEEL_SPEED = 70   # rad/s (~2.2 m/s with 0.032 m wheels)
WHEEL_RADIUS = 0.032
CAR_FRONT = 0.115      # front bumper distance from body origin
CAR_Z = 0.052
CONTROL_HZ = 50
LAP_TIME_LIMIT = 120.0   # s to finish the lap
STOP_TIME_LIMIT = 30.0   # s from finish line to full stop (official rule)
TRAFFIC_CYCLE = 10.0
TRAFFIC_GREEN_TIME = 6.0

# dynamic obstacle behaviour (mode chosen randomly each reset)
DYN_STATIC_LAT = 0.09    # static mode: parked this far off the line
DYN_MOVE_SPAN = 0.22     # moving mode: slides across +/- this lateral range
DYN_MOVE_SPEED = 0.12    # m/s

T2_OBSTACLE_LAT = 0.09   # tunnel #2 obstacle offset (half-road: dodge stays on the road)

# off-track checks are suspended around obstacles (bypassing them requires
# leaving the line) and during the end-zone stop
_NO_LANE_CHECK = (
    (track.DYN_OBSTACLE_S - 0.9, track.DYN_OBSTACLE_S + 0.9),
    (track.TUNNEL_2[0] - 0.5, track.TUNNEL_2[1] + 0.3),
)


class GauntletEnv:
    """MuJoCo simulation of the Gauntlet course."""

    def __init__(self, render_mode: str | None = None):
        """render_mode: None (headless) or "human" (opens the MuJoCo viewer)."""
        self.model = mujoco.MjModel.from_xml_path(str(_ASSETS / "gauntlet.xml"))
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
        self._cube_mocap = self.model.body("stop_cube").mocapid[0]

        def gid(name):
            return self.model.geom(name).id

        self._car_geoms = {gid(n) for n in ("chassis", "board", "camera_mount", "camera_body",
                                            "fl_tire", "fr_tire", "rl_tire", "rr_tire")}
        self._traffic_geoms = {
            "red": gid("traffic_red"),
            "yellow": gid("traffic_yellow"),
            "green": gid("traffic_green"),
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
        self._hazard_geoms[gid("stop_cube_box")] = "cube"
        for i in range(self.model.ngeom):
            name = self.model.geom(i).name
            if name.startswith(("choke_wall", "tunnel1_", "tunnel2_")):
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

        # dynamic obstacle: static on a random side, or slowly crossing
        self._dyn_moving = bool(self._rng.random() < 0.5)
        self._dyn_lat0 = float(self._rng.choice([-1, 1]) * DYN_STATIC_LAT)
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

        # stop cube occupies one half of the road after the lap is complete
        self.cube_side = int(self._rng.choice([-1, 1]))
        self.data.mocap_pos[self._cube_mocap] = [0, -5, -1]

        self.score = ScoreKeeper()
        self.phase = "lap"
        self.lap_time = None
        self._progress = 0.0
        self._prev_s = track.frenet(self.data.qpos[0], self.data.qpos[1])[0]
        self._off_track = False
        self._moved = False
        self._stall_since = None
        self._stopped_since = None
        self._colliding = set()
        self._hit = set()          # hazards already penalized this attempt
        self._minor_offtrack_s = []
        self._speed_max_in_speed_section = 0.0
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
        crossed_stop_line = (
            ds > 0
            and 0 <= track.s_delta(track.TRAFFIC_STOP_S, prev_s) <= ds + 1e-6
        )
        if (self.phase == "lap" and crossed_stop_line and not self._traffic_violated
                and self._traffic_light_state(t) == "red"):
            self.score.traffic_light_violation(t)
            self._traffic_violated = True

        # ---- collisions ------------------------------------------------
        for hazard in step_hazards - self._colliding:
            if hazard == "wall":
                # official rules: collision with track structure forfeits
                self.score.forfeit("wall_collision", t)
                terminated = True
            elif hazard == "obstacle" and "obstacle" not in self._hit:
                self.score.obstacle_contact(t)
                self._hit.add("obstacle")
            elif hazard == "cube":
                # touching the stop cube ends the attempt (-15, lap time kept)
                self.score.cube_contact(t)
                terminated = True
        self._colliding = step_hazards

        # ---- lane keeping ----------------------------------------------
        if self.phase == "lap":
            checked = not any(track.in_range(s, r) for r in _NO_LANE_CHECK)
            if checked and abs(lat) > track.OFFTRACK_MINOR and not self._off_track:
                self.score.off_track_minor(t)
                self._off_track = True
                self._minor_offtrack_s.append(s)
            elif abs(lat) < track.OFFTRACK_MINOR - 0.03:
                self._off_track = False

        # ---- stalling (official: hesitation > 2 s) ---------------------
        if self.phase == "lap":
            light_gap = track.s_delta(track.TRAFFIC_STOP_S, s)
            waiting_at_red = (
                self._traffic_light_state(t) == "red"
                and -0.05 < light_gap < 0.8
            )
            waiting_for_bicycle = (
                self._dyn_moving
                and track.in_range(
                    s,
                    (track.DYN_OBSTACLE_S - 0.8, track.DYN_OBSTACLE_S - 0.2),
                )
                and abs(self._dyn_lat) < 0.20
            )
            if waiting_at_red or waiting_for_bicycle:
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
            if track.in_range(s, track.SPEED):
                self._speed_max_in_speed_section = max(self._speed_max_in_speed_section, speed)

        # ---- lap completion & the end-zone stop ------------------------
        if self.phase == "lap" and self._progress >= track.TOTAL + 0.20:
            self.phase = "stop"
            self.lap_time = t
            self.score.lap_complete(t)
            self._award_lap_bonuses(t)
            cx, cy, heading = track.path_point(track.CUBE_S)
            offset = self.cube_side * (track.ROAD_HALF_WIDTH - track.CUBE_HALF_WIDTH)
            nx, ny = -np.sin(heading), np.cos(heading)
            self.data.mocap_pos[self._cube_mocap] = [
                cx + nx * offset,
                cy + ny * offset,
                0.02 + track.CUBE_HALF_HEIGHT,
            ]
        elif self.phase == "stop" and not terminated:
            gap = self._cube_gap()
            # judge's view: the car itself must be at rest (encoders can read
            # zero while the car is still sliding with locked wheels)
            body_speed = float(np.hypot(self.data.qvel[0], self.data.qvel[1]))
            if body_speed < 0.02:
                if self._stopped_since is None:
                    self._stopped_since = t
                elif t - self._stopped_since > 0.5:
                    self.score.stop_result(gap, t)
                    terminated = True
            else:
                self._stopped_since = None
            if t - self.lap_time > STOP_TIME_LIMIT:
                self.score.stop_timeout(t)
                terminated = True

        # ---- global termination ----------------------------------------
        if self._flipped():
            self.score.forfeit("flipped", t)
            terminated = True
        if self.phase == "lap" and t >= LAP_TIME_LIMIT:
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
            self._viewer.cam.azimuth = 180
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
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.close()
        self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------- course dynamics

    def _update_world(self, t: float):
        """Move dynamic course elements and update signal lamps."""
        ox, oy, heading = track.path_point(track.DYN_OBSTACLE_S)
        if self._dyn_moving:
            # slides back and forth across the track, sinusoidal
            period = 4 * DYN_MOVE_SPAN / DYN_MOVE_SPEED
            lat = DYN_MOVE_SPAN * np.sin(2 * np.pi * t / period + self._dyn_phase)
        else:
            lat = self._dyn_lat0
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
        self._update_traffic_light(t)

    def _traffic_light_state(self, t: float) -> str:
        return "green" if t % TRAFFIC_CYCLE < TRAFFIC_GREEN_TIME else "red"

    def _update_traffic_light(self, t: float):
        state = self._traffic_light_state(t)
        colours = {
            "red": [1.0, 0.02, 0.02, 1.0] if state == "red" else [0.20, 0.01, 0.01, 1.0],
            "yellow": [0.18, 0.12, 0.01, 1.0],
            "green": [0.02, 1.0, 0.02, 1.0] if state == "green" else [0.01, 0.20, 0.01, 1.0],
        }
        for name, geom_id in self._traffic_geoms.items():
            self.model.geom_rgba[geom_id] = colours[name]

    def _award_lap_bonuses(self, t: float):
        """Official bonus tiers that can be judged automatically."""
        if "obstacle" not in self._hit:
            self.score.bonus("obstacle_avoidance", t)
            self.score.bonus("low_light", t)   # cleared tunnel #2 untouched
        if not any(track.in_range(s, track.SHINE) for s in self._minor_offtrack_s):
            self.score.bonus("high_glare", t)
        if (self._speed_max_in_speed_section > 1.5
                and not any(track.in_range(s, track.SPEED) for s in self._minor_offtrack_s)):
            self.score.bonus("speed_section", t)

    def _cube_gap(self) -> float:
        """Gap (m) between the car's front bumper and the stop cube's face."""
        s, _ = track.frenet(self.data.qpos[0], self.data.qpos[1])
        return track.s_delta(track.CUBE_S - track.CUBE_HALF_LENGTH, s) - CAR_FRONT

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
            "phase": self.phase,
            "lap_time": self.lap_time,
            "privileged": {
                "car_pos": pos,
                "car_yaw": float(yaw),
                "s": float(s),
                "lateral": float(lat),
                "progress": float(self._progress),
                "dyn_obstacle": {"moving": self._dyn_moving, "lateral": self._dyn_lat},
                "t2_side": self.t2_side,
                "traffic_light": {
                    "state": self._traffic_light_state(self.data.time),
                    "stop_s": track.TRAFFIC_STOP_S,
                },
                "cube_gap": self._cube_gap() if self.phase == "stop" else None,
            },
        }
