"""Camera-based lane keeping — the way the real competition works.

    python examples/vision_lane_keeper.py             # with viewer
    python examples/vision_lane_keeper.py --headless

The track is a paved road just over two car widths wide, with white edge
lines and no center line: the car must stay BETWEEN the edges. Steering comes
ONLY from the onboard camera. The pipeline here is deliberately simple:

    1. Look at a strip of floor just ahead of the car (the ROI).
    2. The paved road is brighter than the dark ground on either side, so
       threshold between the darkest (ground) and brightest (edge line)
       pixels to isolate the road region.
    3. Steer toward the horizontal centroid of that region — i.e. keep the
       road centered in front of the car.

It works, but leans on that brightness gap, which the course attacks on
purpose: a bright light at the end of the "sheen" straight throws a specular
highlight down the road that blows part of it out toward white, the
checkerboard gate adds high-contrast structure over the road, and the dark
tunnel crushes it toward black. Making detection robust there (and reading
the edge lines directly, rather than the whole road blob) is your project.

Honesty note (marked CHEAT below): obstacle dodging and reading the fork sign
still come from info["privileged"], because doing it from vision is real
project work, not example code. Replace it with camera logic.

Requires OpenGL for offscreen rendering; on a headless machine run with
MUJOCO_GL=egl or MUJOCO_GL=osmesa.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from challenge import ChallengeEnv, track

CAM_W, CAM_H = 800, 450    # 16:9, Raspberry Pi Camera v3 aspect ratio
# A tall strip of floor ahead of the bumper. The ROI must look far enough
# ahead to SEE a turn coming: too near (e.g. 380, just in front of the bumper)
# and the road curves away *above* the strip, so the car keeps reading
# "straight" and drives off the first bend. Reaching to ~mid-frame lets it
# anticipate the course's turns; the trade-off is a noisier road-center
# estimate through narrow gaps (the choke) — try raising/lowering it and watch.
ROI_TOP = 240          # ~0.53 of the way down the frame
BASE_SPEED = 1.1        # m/s
DARK_SPEED = 0.7        # m/s when the road can't be found confidently
MAX_SPEED = 2.2
DODGE_RESPONSE = 0.08
WHEELBASE = 0.15
MAX_STEER_RAD = 0.55
DODGE_LOOKAHEAD = 0.35
STOP_MARGIN = 0.15    # brake target: chassis half-length (~0.11 m) + clearance
# See example_controller.py's FORK_LEAD: the smoothed dodge_bias needs real
# distance to converge before the car's bumper (which leads its frenet
# reference point) reaches the divider's narrow entrance.
FORK_LEAD = 1.3
FORK_EXIT_FADE = 0.8
FORK_SPEED = 0.7        # m/s through the fork -- more real time per metre to
                         # steer gently, so recentring doesn't build up yaw
# Aim inside the lane rather than dead-centre (track.FORK_LANE_OFFSET):
# still clears the divider with margin, but the shorter swing keeps the
# recentring steer (and the yaw it builds up) small enough for the choke
# just past the fork to forgive.
FORK_TARGET_OFFSET = 0.15


class VisionLaneKeeper:
    # ROI rows (start, end) the pipeline actually looks at, exposed so
    # view_camera.py can draw exactly what the algorithm sees.
    ROI = (ROI_TOP, CAM_H)

    def __init__(self, env):
        self.env = env
        self.err = 0.0            # last normalized road-center offset [-1, 1]
        self.road_seen = False
        # debug state for visualization (view_camera.py)
        self.frame = None        # last camera frame (CAM_H, CAM_W, 3)
        self.mask = None         # road pixels within the ROI, or None
        self.cx = None           # detected road-center column, or None
        self.dodge_bias = 0.0    # smoothed obstacle-avoidance steering bias

    def process_frame(self):
        """Update self.err from the camera. Returns True if the road was found."""
        rgb = self.env.camera_image(width=CAM_W, height=CAM_H)
        self.frame = rgb
        self.mask = None
        self.cx = None
        roi = rgb[ROI_TOP:, :, :].mean(axis=2)          # grayscale floor strip

        # The paved road is brighter than the dark ground beside it, at every
        # lighting level: road ~100 vs ground ~23 in daylight, ~30 vs ~6 in
        # the dark tunnel, up to ~235 in the sheen straight's specular streak.
        # A threshold placed a fixed FRACTION of the way from the darkest to
        # the brightest pixel tracks all of those; a fixed absolute value
        # would not. Naive, though — specular glare or a pale obstacle can
        # masquerade as road.
        lo, hi = float(roi.min()), float(roi.max())
        if hi - lo < 20:                                 # no road/ground contrast
            self.road_seen = False
            return False
        mask = roi > lo + 0.22 * (hi - lo)               # the road (+ its edges)
        if mask.sum() < 400:                             # too little road in view
            self.road_seen = False
            return False

        cols = np.nonzero(mask)[1]
        cx = cols.mean()
        self.mask = mask
        self.cx = float(cx)
        # low-pass the error so a one-frame detection glitch (e.g. a wall
        # edge sweeping through view at the choke) doesn't jerk the steering
        new_err = (cx - CAM_W / 2) / (CAM_W / 2)
        self.err = 0.6 * self.err + 0.4 * new_err if self.road_seen else new_err
        self.road_seen = True
        return True

    def act(self, obs, info):
        p = info["privileged"]
        s = p["s"]

        offset_err = 0.0
        path_offset = None
        fork_blend = None   # None = hard cutover; else 0..1 weight toward pursuit_steer
        speed = BASE_SPEED

        # CHEAT: read the fork sign from privileged info — replace with camera
        # detection of the lit arrow. A divider splits the road view into two
        # separate blobs here, so the brightness-centroid steer (self.err)
        # would aim between them; pursue the commanded lane's centre instead.
        # The divider's own circular-ease taper stays close to full width just
        # past FORK_S[1] (zero slope where it meets the core), so both the
        # target offset AND the steering law itself fade back to the camera
        # estimate over FORK_EXIT_FADE, rather than snapping at the boundary
        # -- a hard cutover leaves a residual yaw kick right where the wall
        # is still substantial, which the choke just past it does not forgive.
        fork_exit = track.FORK_S[1] + FORK_EXIT_FADE
        if track.in_range(s, (track.FORK_S[0] - FORK_LEAD, fork_exit)):
            target = FORK_TARGET_OFFSET if p["fork_direction"] == "left" else -FORK_TARGET_OFFSET
            if s > track.FORK_S[1]:
                fade = max(1.0 - track.s_delta(s, track.FORK_S[1]) / FORK_EXIT_FADE, 0.0)
                target *= fade
                fork_blend = fade
            path_offset = target
            speed = FORK_SPEED

        # CHEAT: dodge offsets from privileged info — replace with camera
        # detection of the obstacles (color blobs are a good start).
        if track.in_range(s, (track.TUNNEL_2[0] - 0.9, track.TUNNEL_2[1])):
            speed = 0.5                                  # slow for the dark tunnel
        if track.in_range(s, (track.TUNNEL_2[0] + 0.1, track.T2_OBSTACLE_S + 0.35)):
            path_offset = -p["t2_side"] * 0.10         # aim for the free half,
            # once inside where the walls are parallel — swerving at the mouth
            # while still yawed from the curve clips the tunnel's leading edge
        # CHEAT: the bicycle continuously crosses the road; wait for a gap
        # (real robot: detect it with the camera).
        dyn = p["dyn_obstacle"]
        if track.in_range(s, (track.DYN_OBSTACLE_S - 1.0, track.DYN_OBSTACLE_S + 0.45)):
            if abs(dyn["lateral"]) < 0.20:
                speed = 0.0                              # wait for the gap

        # CHEAT: obey the simulated signal from privileged state.
        # light_gap tracks the car's body origin, but the chassis extends
        # ~0.11 m ahead of it -- braking to zero at gap=0.08 (as if the car
        # were a point) left the front bumper straddling the line. STOP_MARGIN
        # adds that overhang back plus a little clearance.
        light = p["traffic_light"]
        light_gap = track.s_delta(light["stop_s"], s)
        if light["state"] in ("red", "yellow") and 0 < light_gap < 0.9:
            speed = min(speed, float(np.clip(1.5 * (light_gap - STOP_MARGIN), 0.0, 0.6)))

        if not self.road_seen:
            speed = min(speed, DARK_SPEED)               # lost the road: ease off

        self.dodge_bias += DODGE_RESPONSE * (offset_err - self.dodge_bias)
        steer = np.clip(-2.0 * (self.err + self.dodge_bias), -1, 1)
        if path_offset is not None:
            # The camera's road-centroid estimate is unreliable here (skewed by
            # the dark tunnel wall/block, or split into two blobs by the fork
            # divider). Since both cases already use privileged information,
            # pursue an explicit offset point instead of a steering bias that
            # could be cancelled out by that skewed/split estimate.
            tx, ty, target_heading = track.path_point(s + DODGE_LOOKAHEAD)
            tx += -np.sin(target_heading) * path_offset
            ty += np.cos(target_heading) * path_offset
            x, y, _ = p["car_pos"]
            yaw = p["car_yaw"]
            dx, dy = tx - x, ty - y
            local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
            local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
            curvature = 2 * local_y / max(local_x**2 + local_y**2, 1e-6)
            pursuit_steer = np.clip(
                np.arctan(WHEELBASE * curvature) / MAX_STEER_RAD,
                -1,
                1,
            )
            steer = fork_blend * pursuit_steer + (1 - fork_blend) * steer if fork_blend is not None else pursuit_steer
        return np.array([steer, speed / MAX_SPEED])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    env = ChallengeEnv(render_mode=None if args.headless else "human")
    obs, info = env.reset(seed=args.seed)
    keeper = VisionLaneKeeper(env)

    last_events, i = 0, 0
    term = trunc = False
    while not (term or trunc):
        if i % 2 == 0:                    # vision at 25 Hz, control at 50 Hz
            keeper.process_frame()
        obs, _, term, trunc, info = env.step(keeper.act(obs, info))
        for e in info["events"][last_events:]:
            print(f"[t={e['t']:6.2f}s] {e['event']} {e['detail']} ({e['points']:+d})")
        last_events = len(info["events"])
        if not args.headless:
            time.sleep(env.model.opt.timestep * 10)
        i += 1

    print()
    print(env.score.summary())
    env.close()


if __name__ == "__main__":
    main()
