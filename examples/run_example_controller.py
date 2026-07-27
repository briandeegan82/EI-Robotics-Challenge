"""Run the reference controller on the Gauntlet.

    python examples/run_example_controller.py             # with viewer
    python examples/run_example_controller.py --headless  # no graphics
    python examples/run_example_controller.py --seed 3
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gauntlet import GauntletEnv
from example_controller import ExampleController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="run without the viewer")
    parser.add_argument("--seed", type=int, default=None, help="course randomization seed")
    args = parser.parse_args()

    env = GauntletEnv(render_mode=None if args.headless else "human")
    obs, info = env.reset(seed=args.seed)
    controller = ExampleController()

    last_event_count = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(controller.act(obs, info))

        for event in info["events"][last_event_count:]:
            print(f"[t={event['t']:6.2f}s] {event['event']} {event['detail']} ({event['points']:+d})")
        last_event_count = len(info["events"])

        if not args.headless:
            time.sleep(env.model.opt.timestep * 10)  # roughly real time
        if terminated or truncated:
            break

    print()
    print(env.score.summary())
    print(f"Finished in {info['time']:.1f}s with score {info['score']}")
    env.close()


if __name__ == "__main__":
    main()
