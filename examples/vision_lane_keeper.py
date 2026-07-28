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
purpose: the high-glare curve blows the road out toward white, and the dark
tunnels crush it toward black. Making detection robust there (and reading the
edge lines directly, rather than the whole road blob) is your project.

Honesty notes (marked CHEAT below): obstacle dodging and the stop-cube
distance still come from info["privileged"], because doing them from vision
is real project work, not example code. Replace them with camera logic.

Requires OpenGL for offscreen rendering; on a headless machine run with
MUJOCO_GL=egl or MUJOCO_GL=osmesa.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gauntlet import GauntletEnv, track

CAM_W, CAM_H = 800, 450    # 16:9, Raspberry Pi Camera v3 aspect ratio
# Only the bottom rows: the floor just ahead of the bumper. Keeping it near
# makes the road-center estimate steady where it matters most (the choke, the
# tunnel mouths): a taller ROI sees further but is pulled off by the road
# *beyond* a narrow gap and by upcoming curves — try raising it and watch.
ROI_TOP = 380          # ~0.84 of the way down the frame
BASE_SPEED = 1.1        # m/s
DARK_SPEED = 0.7        # m/s when the road can't be found confidently
MAX_SPEED = 2.2


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

    def process_frame(self):
        """Update self.err from the camera. Returns True if the road was found."""
        rgb = self.env.camera_image(width=CAM_W, height=CAM_H)
        self.frame = rgb
        self.mask = None
        self.cx = None
        roi = rgb[ROI_TOP:, :, :].mean(axis=2)          # grayscale floor strip

        # The paved road is brighter than the dark ground beside it, at every
        # lighting level: road ~100 vs ground ~23 in daylight, ~30 vs ~6 in
        # the dark tunnel, ~207 in the glare. A threshold placed a fixed
        # FRACTION of the way from the darkest to the brightest pixel tracks
        # all of those; a fixed absolute value would not. Naive, though —
        # specular glare or a pale obstacle can masquerade as road.
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
        speed = BASE_SPEED

        # CHEAT: dodge offsets from privileged info — replace with camera
        # detection of the obstacles (color blobs are a good start).
        if track.in_range(s, (track.TUNNEL_2[0] - 0.9, track.TUNNEL_2[1])):
            speed = 0.5                                  # slow for the dark tunnel
        if track.in_range(s, (track.TUNNEL_2[0] + 0.1, track.T2_OBSTACLE_S + 0.35)):
            offset_err = p["t2_side"] * 0.16             # dodge to the free half,
            # once inside where the walls are parallel — swerving at the mouth
            # while still yawed from the curve clips the tunnel's leading edge
        dyn = p["dyn_obstacle"]
        if track.in_range(s, (track.DYN_OBSTACLE_S - 1.0, track.DYN_OBSTACLE_S + 0.45)):
            if dyn["moving"] and abs(dyn["lateral"]) < 0.20:
                speed = 0.0                              # wait for the gap
            elif not dyn["moving"]:
                offset_err = np.sign(dyn["lateral"]) * 0.50
                speed = 0.6

        # CHEAT: stop-cube range from privileged info — replace with the
        # cube's apparent size/position in the image.
        if info["phase"] == "stop":
            gap = p["cube_gap"]
            speed = float(np.clip(1.2 * (gap - 0.03), 0.0, 0.6))
            if gap < 0.048:
                speed = 0.0

        # CHEAT: obey the simulated signal from privileged state.
        light = p["traffic_light"]
        light_gap = track.s_delta(light["stop_s"], s)
        if info["phase"] == "lap" and light["state"] == "red" and 0 < light_gap < 0.9:
            speed = min(speed, float(np.clip(1.5 * (light_gap - 0.08), 0.0, 0.6)))

        if not self.road_seen:
            speed = min(speed, DARK_SPEED)               # lost the road: ease off

        steer = np.clip(-2.0 * (self.err + offset_err), -1, 1)
        return np.array([steer, speed / MAX_SPEED])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    env = GauntletEnv(render_mode=None if args.headless else "human")
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
