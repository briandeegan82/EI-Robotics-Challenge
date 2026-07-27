"""GauntletEnv: the simulation environment for the Gauntlet competition.

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
    print(info["score"])

Action (2 floats, each in [-1, 1]):
    action[0]  steering: +1 = full left, -1 = full right (max ±0.6 rad)
    action[1]  throttle: +1 = full speed ahead (~3 m/s), -1 = full reverse

Observation (18 floats) — everything the *real* robot could plausibly sense:
    obs[0:11]  rangefinder distances in meters, right (-60 deg) to left (+60 deg),
               clipped to 5 m (5.0 = nothing in range)
    obs[11]    forward speed (m/s, body frame)
    obs[12]    lateral speed (m/s, body frame)
    obs[13]    yaw rate (rad/s)
    obs[14]    steering angle (rad)
    obs[15]    left rear wheel angular velocity (rad/s)
    obs[16]    right rear wheel angular velocity (rad/s)
    obs[17]    elapsed time (s)

info dict:
    info["score"]       current score (see scoring.py)
    info["events"]      list of scoring events so far
    info["privileged"]  ground-truth state the real robot will NOT have
                        (car pose, light state, bicycle position, obstacle side).
                        Great for getting started; wean yourself off it before
                        the real competition.

An onboard camera image is available via ``env.camera_image()`` (requires
working OpenGL; everything else runs fully headless).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .scoring import ScoreKeeper

_ASSETS = Path(__file__).parent / "assets"

# --- course constants (keep in sync with gauntlet.xml) ---
STOP_LINE_X = 5.5
LIGHT_X = 6.0
CROSSING_X = 12.0
TUNNEL_X = (18.0, 24.0)
OBSTACLE_X = 21.0
FINISH_X = 28.0
CHECKPOINTS = {"traffic_light": 7.0, "crossing": 13.5, "tunnel": 24.5}

# traffic light cycle (seconds)
GREEN_S, YELLOW_S, RED_S = 5.0, 2.0, 5.0
CYCLE_S = GREEN_S + YELLOW_S + RED_S

# bicycle motion: crosses between y=-2.2 and y=+2.2, pausing at each end
BIKE_SPAN = 2.2
BIKE_SPEED = 0.8      # m/s
BIKE_DWELL = 2.0      # s pause at each end

MAX_STEER = 0.6       # rad
MAX_WHEEL_SPEED = 50  # rad/s  (~3 m/s with 0.06 m wheels)
CONTROL_HZ = 50
TIME_LIMIT = 120.0    # s

_LAMP_COLORS = {
    "lamp_red": ((0.3, 0.05, 0.05, 1), (1.0, 0.1, 0.1, 1)),
    "lamp_yellow": ((0.3, 0.25, 0.05, 1), (1.0, 0.85, 0.1, 1)),
    "lamp_green": ((0.05, 0.3, 0.05, 1), (0.1, 1.0, 0.15, 1)),
}


class GauntletEnv:
    """MuJoCo simulation of the Gauntlet course."""

    def __init__(self, render_mode: str | None = None):
        """render_mode: None (headless) or "human" (opens the MuJoCo viewer)."""
        self.model = mujoco.MjModel.from_xml_path(str(_ASSETS / "gauntlet.xml"))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._viewer = None
        self._renderer = None
        self._rng = np.random.default_rng()

        self._frame_skip = int(round(1 / (CONTROL_HZ * self.model.opt.timestep)))

        # cache ids
        self._sensor = {
            self.model.sensor(i).name: i for i in range(self.model.nsensor)
        }
        self._lamp_geoms = {
            name: self.model.geom(name).id for name in _LAMP_COLORS
        }
        self._bike_mocap = self.model.body("bicycle").mocapid[0]
        self._obstacle_mocap = self.model.body("obstacle").mocapid[0]
        self._car_body = self.model.body("car").id

        def gid(name):
            return self.model.geom(name).id

        self._car_geoms = {
            gid(n) for n in ("chassis", "cabin", "fl_tire", "fr_tire", "rl_tire", "rr_tire")
        }
        self._hazard_geoms = {
            gid("bicycle_hitbox"): "bicycle",
            gid("obstacle_box"): "obstacle",
        }
        for n in ("curb_l1", "curb_r1", "curb_l2", "curb_r2", "curb_l3", "curb_r3",
                  "tunnel_wall_l", "tunnel_wall_r", "tunnel_roof",
                  "pole_l", "pole_r", "flag_pole_l", "flag_pole_r"):
            self._hazard_geoms[gid(n)] = "wall"

        self.reset(seed=0)

    # ------------------------------------------------------------------ API

    def reset(self, seed: int | None = None):
        """Reset the course. Returns (obs, info)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        # small random start offset, like an imperfect staging box
        self.data.qpos[1] = self._rng.uniform(-0.1, 0.1)   # y
        self.data.qpos[0] = self._rng.uniform(-0.05, 0.05)  # x

        # randomize which side of the tunnel the obstacle is on
        self.obstacle_side = self._rng.choice([-1, 1])
        self.data.mocap_pos[self._obstacle_mocap] = [OBSTACLE_X, 0.4 * self.obstacle_side, 0.17]

        # random phases so every run is a little different
        self._light_phase = self._rng.uniform(0, CYCLE_S)
        self._bike_phase = self._rng.uniform(0, 2 * (2 * BIKE_SPAN / BIKE_SPEED + BIKE_DWELL))

        self.score = ScoreKeeper()
        self._prev_x = self.data.qpos[0]
        self._colliding = set()        # hazards in contact last step
        self._last_collision_t = {}    # hazard -> time of last scored event
        self._done = False

        mujoco.mj_forward(self.model, self.data)
        self._update_world(0.0)
        return self._obs(), self._info()

    def step(self, action):
        """Advance one control step (1/50 s). Returns (obs, reward, terminated, truncated, info)."""
        if self._done:
            raise RuntimeError("Episode is over — call reset() first.")
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.data.ctrl[0] = action[0] * MAX_STEER
        self.data.ctrl[1] = action[1] * MAX_WHEEL_SPEED
        self.data.ctrl[2] = action[1] * MAX_WHEEL_SPEED

        step_hazards = set()
        for _ in range(self._frame_skip):
            self._update_world(self.data.time)
            mujoco.mj_step(self.model, self.data)
            step_hazards |= self._contact_hazards()

        # collision events fire once per new contact, with a 1 s cooldown per
        # hazard so a scraping contact isn't scored dozens of times
        for hazard in step_hazards - self._colliding:
            if self.data.time - self._last_collision_t.get(hazard, -10) > 1.0:
                self.score.collision(hazard, self.data.time)
                self._last_collision_t[hazard] = self.data.time
        self._colliding = step_hazards

        x = self.data.qpos[0]
        t = self.data.time

        # red light violation: crossing the stop line while the light is red
        if self._prev_x < STOP_LINE_X <= x and self.light_state == "red":
            self.score.red_light(t)

        for name, cx in CHECKPOINTS.items():
            if self._prev_x < cx <= x:
                self.score.checkpoint(name, t)

        terminated, truncated = False, False
        if self._prev_x < FINISH_X <= x:
            self.score.finish(t)
            terminated = True
        if self._flipped():
            self.score.flipped(t)
            terminated = True
        if t >= TIME_LIMIT:
            self.score.timeout(t)
            truncated = True

        # simple shaped reward for RL users: forward progress plus scoring events
        reward = (x - self._prev_x) + self.score.consume_reward()
        self._prev_x = x
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
            self._viewer.cam.distance = 3.0
            self._viewer.cam.elevation = -25
            self._viewer.cam.azimuth = 180
        if self._viewer.is_running():
            self._viewer.sync()

    def camera_image(self, camera: str = "onboard", width: int = 320, height: int = 240):
        """Return an RGB image (H, W, 3 uint8) from the named camera.

        Requires OpenGL. On a headless machine set MUJOCO_GL=egl or osmesa.
        """
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._scene_option = mujoco.MjvOption()
            # hide the rangefinder debug rays — a real camera wouldn't see them
            self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
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
        """Advance the scripted parts of the world: traffic light and bicycle."""
        # traffic light
        phase = (t + self._light_phase) % CYCLE_S
        if phase < GREEN_S:
            self.light_state = "green"
        elif phase < GREEN_S + YELLOW_S:
            self.light_state = "yellow"
        else:
            self.light_state = "red"
        for name, (dim, bright) in _LAMP_COLORS.items():
            on = name == f"lamp_{self.light_state}"
            self.model.geom_rgba[self._lamp_geoms[name]] = bright if on else dim

        # bicycle: constant-speed crossing with a dwell at each end
        travel = 2 * BIKE_SPAN / BIKE_SPEED
        period = 2 * (travel + BIKE_DWELL)
        p = (t + self._bike_phase) % period
        if p < travel:                       # heading +y
            y = -BIKE_SPAN + BIKE_SPEED * p
        elif p < travel + BIKE_DWELL:        # dwell at +y end
            y = BIKE_SPAN
        elif p < 2 * travel + BIKE_DWELL:    # heading -y
            y = BIKE_SPAN - BIKE_SPEED * (p - travel - BIKE_DWELL)
        else:                                # dwell at -y end
            y = -BIKE_SPAN
        self.data.mocap_pos[self._bike_mocap] = [CROSSING_X, y, 0.02]
        self._bike_y = y

    def _contact_hazards(self):
        """Names of hazards the car is currently touching."""
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
        # world z component of the car's local z axis
        return self.data.xmat[self._car_body].reshape(3, 3)[2, 2] < 0.2

    # -------------------------------------------------------------- readouts

    def _sensordata(self, name):
        i = self._sensor[name]
        adr = self.model.sensor_adr[i]
        dim = self.model.sensor_dim[i]
        return self.data.sensordata[adr:adr + dim].copy()

    def _obs(self):
        rf = np.array([self._sensordata(f"rf_{i}")[0] for i in range(11)])
        rf[rf < 0] = 5.0          # -1 means "no hit" -> report max range
        vel = self._sensordata("velocity")
        gyro = self._sensordata("gyro")
        return np.concatenate([
            rf,
            [vel[0], vel[1], gyro[2]],
            self._sensordata("steer_angle"),
            self._sensordata("wheel_speed_l"),
            self._sensordata("wheel_speed_r"),
            [self.data.time],
        ]).astype(np.float32)

    def _info(self):
        pos = self._sensordata("car_pos")
        qw, qx, qy, qz = self._sensordata("car_quat")
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        return {
            "score": self.score.total(),
            "events": list(self.score.events),
            "time": self.data.time,
            "privileged": {
                "car_pos": pos,
                "car_yaw": float(yaw),
                "light_state": self.light_state,
                "bike_y": float(self._bike_y),
                "obstacle_side": int(self.obstacle_side),
            },
        }
