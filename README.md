# The Gauntlet — Line-Following Challenge Simulator

A MuJoCo simulation of the Qualcomm Student Hackathon autonomous vehicle
line-following challenge. The real competition runs on physical Rubik Pi
vehicles — this simulator exists so your team can understand the course,
prototype the vision pipeline and control logic, and practice long before you
touch hardware.

The track mirrors the official layout: a white line on a dark surface around
a loop that fits the official 7m × 4m footprint, with every challenge section
from the guidelines:

```
                       ┌─ tunnel #1 (lit) ─┐   30
        ◄──────────────┤███████████████████├───▼─────◄──────────
      ┌─                 speed section  [dyn obstacle]           ─┐
  HIGH GLARE                                                    curve
  (reflective)                                                    │
      └─   tunnel #2 (dark)                choke ▲│▲              ─┘
        ───┤███ [obstacle] ███├──┃━►──■───────────┴──────────►
                              start  stop      lane keeping
                              /finish cube (after lap)
```

Driving order: **start → lane keeping → choke point → curve → tunnel #1 →
speed section (dynamic obstacle) → high-glare curve → tunnel #2 (obstacle
inside) → finish → stop cube**.

An attempt, exactly as in the official rules: complete one lap (fastest lap
of 3 attempts wins), then the white 20 cm stop cube is placed on the track
and the car must **stop within 10 cm of it without touching it** (30 s
allowed). Off-track = forfeit; hitting track structure = forfeit.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# feel the course with your keyboard (arrows; space = stop; backspace = reset)
python examples/drive_keyboard.py

# watch the reference controller drive a clean lap (uses privileged info)
python examples/run_example_controller.py

# the real thing: camera-based line following
python examples/vision_line_follower.py

# SEE what the camera sees, with the line-detection overlay
pip install -e ".[viz]"        # one-time: adds OpenCV
python examples/view_camera.py

# headless (no graphics window; camera rendering still needs OpenGL)
MUJOCO_GL=osmesa python examples/vision_line_follower.py --headless --seed 3
```

## The rules, in simulation

Faithful to the official guidelines wherever they can be auto-judged:

- **Primary metric: lap time.** Teams get 3 attempts; the fastest valid lap
  wins. Run 3 episodes and keep the best — `env.score.result()` reports
  `lap_time` (or `None` if the attempt was forfeited).
- **Sensors match the hardware rules.** Camera (primary), IMU, wheel
  encoders, steering feedback. **Depth sensors are prohibited** at the real
  event, so the sim has none — obstacle detection must come from the camera.
- **Vehicle matches the Rubik Pi class**: ~22 × 16 cm, ~1 kg, Ackermann
  steering, rear-wheel drive, ~2.2 m/s top speed.

| points | event (auto-judged subset of the official tables) |
|---|---|
| +5 | dynamic obstacle passed without contact |
| +5 | tunnel #2 (low light) cleared without contact |
| +5 | high-glare curve without lane violations |
| +5 | speed section at >1.5 m/s without losing the line |
| +10 | stop within 5 cm of the cube, no contact |
| −2 | minor off-track (wheel on the boundary), per incident |
| −5 | stalling >2 s |
| −15 | contact with an obstacle or the stop cube |
| forfeit | complete loss of track, collision with track structure, flip |

Smoothness/precision bonuses from the official table are judged by humans at
the event and aren't scored here. Values live in `gauntlet/scoring.py`.

## Writing your controller

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

**Action** — 2 floats in [-1, 1]: `[steering (+1 = left), throttle (+1 ≈ 2.2 m/s)]`.

**Observation** — 8 floats, only what the real robot senses on-board:

| index | meaning |
|---|---|
| 0 | forward speed from wheel encoders (m/s) |
| 1–2 | longitudinal / lateral acceleration (m/s², body frame) |
| 3 | yaw rate (rad/s) |
| 4 | steering angle (rad) |
| 5–6 | rear wheel angular velocities (rad/s) |
| 7 | elapsed time (s) |

**Camera** — `env.camera_image()` returns the onboard RGB frame. This is the
competition's primary sensor: line following, obstacle detection, and ranging
the stop cube are all meant to be done from it. Run
`python examples/view_camera.py` to *watch* that feed live with the
line-detection overlay drawn on it (green = detected line pixels, cyan =
the column the follower steers toward) next to a chase view — the fastest way
to see why a follower drifts. The course is deliberately hostile to naive
vision, just like the real event:

- the **glare curve** washes the floor out to near-white (fixed thresholds die
  here — see the median-relative threshold in `vision_line_follower.py`),
- **tunnel #1** is dim, **tunnel #2** is genuinely dark,
- both tunnel mouths mix bright and dark content in one frame.

**`info["privileged"]`** — ground truth (pose, arc length, lateral offset,
obstacle positions, cube distance) that the real robot will **not** have.
Use it to get moving on day one; replace each use with perception before the
real event. Both examples label exactly where they cheat.

## Project layout

```
gauntlet/
  track.py           track geometry: centerline math, section positions —
                     shared by the XML generator and the env
  generate_track.py  writes assets/gauntlet.xml (python -m gauntlet.generate_track)
  env.py             GauntletEnv — reset/step API, rules enforcement, phases
  scoring.py         all point values and the attempt result
  assets/
    gauntlet.xml     the generated course
    car.xml          the Rubik Pi-class car (camera, IMU, encoders)
examples/
  drive_keyboard.py          drive manually to learn the course
  example_controller.py      pure-pursuit reference (privileged info; 15/15
                             clean laps ~10.5 s, max auto-judged points)
  run_example_controller.py  runs the reference controller
  vision_line_follower.py    camera-based line following — the real approach
                             (~18 s laps; obstacle/stop logic still privileged)
  view_camera.py             live onboard-camera view with the line-detection
                             overlay + chase view (needs the [viz] extra)
```

## Notes for teams

- **Every attempt differs**: obstacle placement/behaviour (parked on a random
  side, or crossing the track) and your staging position randomize on
  `reset()`. The official track layout is only revealed on competition day —
  don't overfit; `track.py` makes it easy to build variant layouts.
- **The env is Gymnasium-shaped** (`reset`/`step`, 5-tuple) with no Gymnasium
  dependency; RL teams can wrap it in a few lines. `step` returns a shaped
  reward (arc-length progress + scaled scoring events).
- **Physics at 500 Hz, control at 50 Hz.** Vision at 25 Hz (every other
  control step) is a realistic processing budget for embedded hardware.
- **Customizing**: geometry in `gauntlet/track.py` + `generate_track.py`
  (rerun the generator), rules/timing constants at the top of
  `gauntlet/env.py`, point values in `gauntlet/scoring.py`.
