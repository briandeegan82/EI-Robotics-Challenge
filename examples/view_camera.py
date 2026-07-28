"""Watch what the onboard camera sees — with the road-detection overlay.

    python examples/view_camera.py                 # lane keeper drives
    python examples/view_camera.py --seed 3
    python examples/view_camera.py --record run.mp4 # also save the window to a file

Opens one window with two panels:

    LEFT   the onboard camera feed (exactly what the algorithm receives,
           800x450), with the vision pipeline drawn on top:
             * green  = pixels kept as road (the mask)
             * yellow = the region of interest it looks at
             * cyan   = the road-center column it steers toward
             * white  = image center (zero-error reference)
    RIGHT  a third-person chase view of the car on the track.

This is the tool to reach for when the car drifts toward an edge: the overlay
shows *why* — e.g. the road mask collapses in the dark tunnel, or the glare
curve blooms the whole strip toward white. Press q or Esc to quit.

Requires OpenCV (pip install -e ".[viz]") and, on a headless machine,
MUJOCO_GL=egl or MUJOCO_GL=osmesa for the offscreen rendering.
"""

import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from challenge import ChallengeEnv, track
from vision_lane_keeper import CAM_H, CAM_W, VisionLaneKeeper

SCALE = 1                       # camera is already 800x450; no upscaling
PANEL_H = CAM_H * SCALE         # both panels share this height


def draw_overlay(keeper):
    """Return a BGR image of the camera frame with the pipeline drawn on it."""
    frame = keeper.frame
    if frame is None:
        return np.zeros((PANEL_H, CAM_W * SCALE, 3), np.uint8)

    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.float32)
    roi_top, roi_bot = keeper.ROI

    # tint the detected road pixels green
    if keeper.mask is not None:
        band = img[roi_top:roi_bot]
        green = np.zeros_like(band)
        green[..., 1] = 255
        m = keeper.mask[..., None]
        img[roi_top:roi_bot] = np.where(m, 0.45 * band + 0.55 * green, band)

    img = cv2.resize(img.astype(np.uint8), (CAM_W * SCALE, CAM_H * SCALE),
                     interpolation=cv2.INTER_NEAREST)

    # ROI band edges (yellow)
    cv2.line(img, (0, roi_top * SCALE), (CAM_W * SCALE, roi_top * SCALE), (0, 220, 220), 1)
    # image center = zero-error reference (white)
    cx0 = CAM_W * SCALE // 2
    cv2.line(img, (cx0, roi_top * SCALE), (cx0, roi_bot * SCALE), (255, 255, 255), 1)
    # detected road-center column (cyan)
    if keeper.cx is not None:
        cx = int(keeper.cx * SCALE)
        cv2.line(img, (cx, roi_top * SCALE), (cx, roi_bot * SCALE), (255, 255, 0), 2)

    status = "ROAD" if keeper.road_seen else "NO ROAD"
    color = (120, 255, 120) if keeper.road_seen else (80, 80, 255)
    cv2.putText(img, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(img, "onboard camera", (8, PANEL_H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return img


def hud(chase_bgr, info, obs, err):
    p = info["privileged"]
    lines = [
        f"lap     {p['progress'] / track.TOTAL * 100:5.1f}%",
        f"speed   {obs[0]:4.2f} m/s",
        f"lane err{err:+5.2f}",
        f"signal  {p['traffic_light']['state']}",
        f"points  {info['score']}",
        f"time    {info['time']:5.1f}s",
    ]
    for i, text in enumerate(lines):
        cv2.putText(chase_bgr, text, (8, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(chase_bgr, text, (8, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    return chase_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--record", metavar="FILE", help="also write the window to an .mp4")
    args = parser.parse_args()

    env = ChallengeEnv(render_mode=None)
    obs, info = env.reset(seed=args.seed)
    keeper = VisionLaneKeeper(env)

    chase = mujoco.Renderer(env.model, height=PANEL_H, width=int(PANEL_H * 4 / 3))
    chase_cam = mujoco.MjvCamera()
    chase_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    chase_cam.trackbodyid = env.model.body("car").id
    chase_cam.distance, chase_cam.elevation, chase_cam.azimuth = 1.4, -35, 0

    writer = None
    win = "EI Robotics Challenge camera view (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    i, last_events = 0, 0
    term = trunc = False
    while True:
        if i % 2 == 0:                         # vision at 25 Hz
            keeper.process_frame()
        if not (term or trunc):
            obs, _, term, trunc, info = env.step(keeper.act(obs, info))
            for e in info["events"][last_events:]:
                print(f"[t={e['t']:6.2f}s] {e['event']} {e['detail']} ({e['points']:+d})")
            last_events = len(info["events"])

        cam_panel = draw_overlay(keeper)
        chase.update_scene(env.data, camera=chase_cam)
        chase_panel = hud(cv2.cvtColor(chase.render(), cv2.COLOR_RGB2BGR), info, obs, keeper.err)
        window = np.hstack([cam_panel, chase_panel])

        if args.record:
            if writer is None:
                h, w = window.shape[:2]
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
            writer.write(window)

        cv2.imshow(win, window)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if term or trunc:
            # freeze on the final frame so the result stays on screen
            print("\n" + env.score.summary())
            if cv2.waitKey(2500) & 0xFF in (ord("q"), 27):
                break
            term = trunc = False
            obs, info = env.reset()
            keeper = VisionLaneKeeper(env)
            last_events = 0
        i += 1

    if writer is not None:
        writer.release()
        print(f"saved {args.record}")
    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
