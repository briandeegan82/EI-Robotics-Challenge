# The Gauntlet — Simulation Template

A MuJoCo simulation of the Gauntlet competition course. The real competition
runs on physical robots — this simulator exists so your team can understand
the course, prototype control strategies, and practice long before you touch
hardware. Expect details (dimensions, timings, scoring) to be tuned as the
official rules are finalized.

```
 START      TRAFFIC LIGHT       BICYCLE           TUNNEL             FINISH
  |              |             CROSSING     ┌────────────────┐         |
  |              ▼                |         │    ┌──┐        │         |
  ▼         ═════╗════       ← ← ▼ → →      │    │██│obstacle│         ▼
  ▓         stop ║           ﻿   🚲         │    └──┘        │        ▓
 ─────────────────────────────────────────────────────────────────────────►
  x=0          x=5.5           x=12        x=18    x=21    x=24      x=28
```

The car drives a ~28 m straight road with three challenges:

1. **Traffic light** (x=6): green → yellow → red on a fixed cycle. Crossing
   the stop line on red costs points.
2. **Bicycle crossing** (x=12): a cyclist repeatedly crosses the road.
   Hitting them costs a lot of points — time your approach.
3. **Tunnel with obstacle** (x=18–24): a box sits on a random side of the
   lane inside the tunnel. The tunnel fits the car *and* the obstacle —
   detect which side it's on (rangefinders!) and drive around it.

The run is **fully autonomous**: your code gets sensor readings and returns
steering + throttle, 50 times per second.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# feel the course with your keyboard (arrows, space, backspace)
python examples/drive_keyboard.py

# watch the reference autonomous controller complete the course
python examples/run_example_controller.py

# run it headless (fast, no graphics)
python examples/run_example_controller.py --headless --seed 3
```

## Writing your controller

Copy `examples/example_controller.py` and start editing. The loop looks like:

```python
from gauntlet import GauntletEnv

env = GauntletEnv(render_mode="human")
obs, info = env.reset(seed=0)
while True:
    action = my_brain(obs, info)          # -> [steer, throttle] in [-1, 1]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(env.score.summary())
```

**Action** — 2 floats in [-1, 1]:

| index | meaning |
|---|---|
| 0 | steering (+1 full left, -1 full right, ±0.6 rad at the wheels) |
| 1 | throttle (+1 ≈ 3 m/s forward, negative reverses/brakes) |

**Observation** — 18 floats, matching sensors the real robot will have:

| index | meaning |
|---|---|
| 0–10 | rangefinder distances (m), fanned right (−60°) to left (+60°), max 5 m |
| 11–12 | forward / lateral speed (m/s, body frame) |
| 13 | yaw rate (rad/s) |
| 14 | steering angle (rad) |
| 15–16 | rear wheel angular velocities (rad/s) |
| 17 | elapsed time (s) |

**Camera** — `env.camera_image()` returns the onboard RGB frame
(320×240×3). That's how the real robot will read the traffic light. On a
machine without a display run with `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa`.

**`info["privileged"]`** — ground truth (car pose, light state, bicycle
position, obstacle side) that the real robot will **not** have. Use it to get
moving on day one, then replace each lookup with perception. The reference
controller marks exactly where it cheats.

## Scoring

Defined in `gauntlet/scoring.py` (placeholder values until official rules land):

| event | points |
|---|---|
| checkpoint passed (light / crossing / tunnel) | +50 each |
| finish line | +500 |
| running a red light | −100 |
| hitting the bicycle | −150 |
| hitting the tunnel obstacle | −75 |
| wall / curb strike | −10 each |
| flipping the car (ends run) | −200 |

Time limit 120 s; ties break on elapsed time. The reference controller scores
650 (clean run) in ~17–27 s depending on light/bicycle timing.

## Project layout

```
gauntlet/
  env.py           GauntletEnv — reset/step API, course logic (light, bicycle,
                   obstacle placement), collision & rule detection
  scoring.py       all scoring rules and point values in one place
  assets/
    gauntlet.xml   the course (road, light, crossing, tunnel, scoring geoms)
    car.xml        the robot (Ackermann car, camera, IMU, 11-ray rangefinder)
examples/
  drive_keyboard.py          drive manually to learn the course
  example_controller.py      readable reference autonomous controller
  run_example_controller.py  runs the reference controller, prints the score
```

## Notes for teams

- **Every run differs.** The light phase, bicycle timing, obstacle side, and
  your exact start position are randomized on `reset()`. Pass a `seed` for
  reproducible debugging, but test across many seeds — judging will not use
  your favorite seed.
- **The env is Gymnasium-shaped** (`reset`/`step` with the standard 5-tuple)
  but has no Gymnasium dependency. RL teams can wrap it in
  `gymnasium.Env` in a few lines; `step` already returns a shaped reward
  (forward progress + scaled scoring events).
- **Physics runs at 500 Hz, control at 50 Hz** — the same control rate
  planned for the real robot.
- **Customizing:** course geometry lives entirely in `gauntlet/assets/*.xml`
  (positions/sizes in meters), rule timings in the constants at the top of
  `gauntlet/env.py`, and points in `gauntlet/scoring.py`.
