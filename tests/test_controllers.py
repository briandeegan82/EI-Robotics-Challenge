import unittest

import numpy as np

from examples.vision_lane_keeper import VisionLaneKeeper
from challenge import track


class ObstacleAvoidanceTests(unittest.TestCase):
    def test_vision_tunnel_dodge_steers_away_and_recovers(self):
        obs = np.zeros(8)
        for obstacle_side in (-1, 1):
            with self.subTest(obstacle_side=obstacle_side):
                keeper = VisionLaneKeeper(env=None)
                keeper.road_seen = True
                s = track.TUNNEL_2[0] + 0.2
                x, y, heading = track.path_point(s)
                info = {
                    "privileged": {
                        "car_pos": (x, y, 0.052),
                        "car_yaw": heading,
                        "s": s,
                        "dyn_obstacle": {"moving": True, "lateral": 0.22},
                        "t2_side": obstacle_side,
                        "traffic_light": {
                            "state": "green",
                            "stop_s": track.TRAFFIC_STOP_S,
                        },
                    },
                }

                avoidance_steer = [keeper.act(obs, info)[0] for _ in range(25)]
                self.assertLess(avoidance_steer[-1] * obstacle_side, 0.0)
                self.assertLess(max(abs(value) for value in avoidance_steer), 0.50)

                info["privileged"]["s"] = track.TUNNEL_2[1] + 1.0
                recovery_steer = [keeper.act(obs, info)[0] for _ in range(60)]
                self.assertLess(abs(recovery_steer[-1]), 0.01)


if __name__ == "__main__":
    unittest.main()
