"""Camera-based line following — the way the real competition works.

    python examples/vision_line_follower.py             # with viewer
    python examples/vision_line_follower.py --headless

Steering comes ONLY from the onboard camera: threshold the bright line in the
lower part of the image, take the centroid, steer toward it. This is the
core pipeline the real robot needs — and it fails in exactly the interesting
places (glare washing out contrast, dark tunnels), which is your challenge to
solve with better image processing.

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

CAM_W, CAM_H = 600, 600
# Only the bottom rows: the floor from ~0.15 to ~0.5 m ahead of the bumper.
# A taller ROI previews the track further out but is easily fooled at the
# tunnel mouths, where sunlit walls outshine the line — try it and see.
ROI_TOP = 425          # ~0.71 of the way down the frame
BASE_SPEED = 1.1        # m/s
DARK_SPEED = 0.7        # m/s when the image is too dark to be confident
MAX_SPEED = 2.2


class VisionLineFollower:
    # ROI rows (start, end) the pipeline actually looks at, exposed so
    # view_camera.py can draw exactly what the algorithm sees.
    ROI = (ROI_TOP, CAM_H)

    def __init__(self, env):
        self.env = env
        self.err = 0.0            # last normalized line offset [-1, 1]
        self.line_seen = False
        # debug state for visualization (view_camera.py)
        self.frame = None        # last camera frame (CAM_H, CAM_W, 3)
        self.mask = None         # line pixels within the ROI, or None
        self.cx = None           # detected line column, or None

    def process_frame(self):
        """Update self.err from the camera. Returns True if the line was found."""
        rgb = self.env.camera_image(width=CAM_W, height=CAM_H)
        self.frame = rgb
        self.mask = None
        self.cx = None
        roi = rgb[ROI_TOP:, :, :].mean(axis=2)          # grayscale floor strip

        # The line is brighter than the floor around it — but the absolute
        # levels swing wildly: normal floor ~40, the glare slab ~200, tunnel
        # #2 ~12. Thresholding relative to the median (the floor) instead of
        # a fixed value survives all three. Still naive: specular highlights
        # or a white object ahead can fool it — your problem to solve.
        floor = np.median(roi)
        peak = roi.max()
        if peak < 30 or peak - floor < 25:               # no line contrast
            self.line_seen = False
            return False
        mask = roi > max(floor + 0.6 * (peak - floor), 30)
        if mask.sum() < 120:            # too few bright pixels to be the line
            self.line_seen = False
            return False

        cols = np.nonzero(mask)[1]
        cx = cols.mean()
        self.mask = mask
        self.cx = float(cx)
        self.err = (cx - CAM_W / 2) / (CAM_W / 2)
        self.line_seen = True
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
            offset_err = p["t2_side"] * 0.25             # dodge once inside,
            # where the walls are parallel — swerving at the entrance while
            # still yawed from the curve clips the tunnel's leading edge
        dyn = p["dyn_obstacle"]
        if track.in_range(s, (track.DYN_OBSTACLE_S - 0.7, track.DYN_OBSTACLE_S + 0.35)):
            if dyn["moving"] and abs(dyn["lateral"]) < 0.16:
                speed = 0.0                              # wait for the gap
            elif not dyn["moving"]:
                offset_err = np.sign(dyn["lateral"]) * 0.35
                speed = 0.6

        # CHEAT: stop-cube range from privileged info — replace with the
        # cube's apparent size/position in the image.
        if info["phase"] == "stop":
            gap = p["cube_gap"]
            speed = float(np.clip(1.2 * (gap - 0.03), 0.0, 0.6))
            if gap < 0.048:
                speed = 0.0

        if not self.line_seen:
            speed = min(speed, DARK_SPEED)               # lost the line: ease off

        steer = np.clip(-2.0 * (self.err + offset_err), -1, 1)
        return np.array([steer, speed / MAX_SPEED])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    env = GauntletEnv(render_mode=None if args.headless else "human")
    obs, info = env.reset(seed=args.seed)
    follower = VisionLineFollower(env)

    last_events, i = 0, 0
    term = trunc = False
    while not (term or trunc):
        if i % 2 == 0:                    # vision at 25 Hz, control at 50 Hz
            follower.process_frame()
        obs, _, term, trunc, info = env.step(follower.act(obs, info))
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
