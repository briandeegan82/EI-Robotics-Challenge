"""Drive the Gauntlet by hand to get a feel for the course.

    python examples/drive_keyboard.py

Click on the viewer window first so it receives key presses.

Controls:
    Up / Down     more / less throttle
    Left / Right  steer
    Space         stop (throttle and steering to zero)
    Backspace     restart the run

The real competition is fully autonomous — this script is just for building
intuition about the course, the car's handling, and the sensors.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gauntlet import GauntletEnv

# GLFW key codes used by the MuJoCo viewer
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_SPACE, KEY_BACKSPACE = 32, 259

action = np.zeros(2)
want_reset = False


def on_key(keycode):
    global want_reset
    if keycode == KEY_UP:
        action[1] = min(1.0, action[1] + 0.15)
    elif keycode == KEY_DOWN:
        action[1] = max(-1.0, action[1] - 0.15)
    elif keycode == KEY_LEFT:
        action[0] = min(1.0, action[0] + 0.25)
    elif keycode == KEY_RIGHT:
        action[0] = max(-1.0, action[0] - 0.25)
    elif keycode == KEY_SPACE:
        action[:] = 0
    elif keycode == KEY_BACKSPACE:
        want_reset = True


def main():
    global want_reset
    import mujoco.viewer

    env = GauntletEnv()
    obs, info = env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.model.body("car").id
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 180

        last_events, done = 0, False
        while viewer.is_running():
            step_start = time.time()

            if want_reset or done:
                obs, info = env.reset()
                action[:] = 0
                last_events, done, want_reset = 0, False, False
                print("\n--- new run ---")

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            for e in info["events"][last_events:]:
                print(f"[t={e['t']:6.2f}s] {e['event']} {e['detail']} ({e['points']:+d})")
            last_events = len(info["events"])
            if done:
                print(env.score.summary())
                print("Backspace to go again.")
                while viewer.is_running() and not want_reset:
                    viewer.sync()
                    time.sleep(0.05)
                continue

            print(f"\rspeed {obs[11]:5.2f} m/s   light: {info['privileged']['light_state']:<7}"
                  f"score {info['score']:5d}   t {info['time']:6.1f}s ", end="")

            viewer.sync()
            leftover = env.model.opt.timestep * 10 - (time.time() - step_start)
            if leftover > 0:
                time.sleep(leftover)

    env.close()


if __name__ == "__main__":
    main()
