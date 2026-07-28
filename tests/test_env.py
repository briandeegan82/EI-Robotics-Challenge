import math
import unittest

import mujoco

from gauntlet import GauntletEnv, track


class ExtendedCourseTests(unittest.TestCase):
    def setUp(self):
        self.env = GauntletEnv()

    def tearDown(self):
        self.env.close()

    def test_generated_course_features_load(self):
        self.assertGreater(self.env.model.ngeom, 200)
        for name in (
            "dyn_obstacle_rear_wheel",
            "dyn_obstacle_front_wheel",
            "dyn_obstacle_frame_base",
            "dyn_obstacle_handlebar",
            "traffic_red",
            "traffic_green",
            "traffic_stop_line",
            "tunnel1_roof_l",
            "tunnel2_roof",
        ):
            self.assertGreaterEqual(self.env.model.geom(name).id, 0)

    def test_signal_cycle(self):
        self.assertEqual(self.env._traffic_light_state(0.0), "green")
        self.assertEqual(self.env._traffic_light_state(5.99), "green")
        self.assertEqual(self.env._traffic_light_state(6.0), "red")
        self.assertEqual(self.env._traffic_light_state(9.99), "red")
        self.assertEqual(self.env._traffic_light_state(10.0), "green")

    def test_mujoco_default_reset_returns_to_start(self):
        self.env.data.qpos[0:2] = [0.0, 0.0]
        mujoco.mj_resetData(self.env.model, self.env.data)
        s, lateral = track.frenet(
            float(self.env.data.qpos[0]),
            float(self.env.data.qpos[1]),
        )
        self.assertAlmostEqual(
            track.s_delta(s, track.START_S - 0.25),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(lateral, 0.0, places=6)

    def test_crossing_on_red_is_penalized_once(self):
        self.env.reset(seed=0)
        before = track.TRAFFIC_STOP_S - 0.05
        after = track.TRAFFIC_STOP_S + 0.05
        x, y, _ = track.path_point(after)
        self.env.data.qpos[0:2] = [x, y]
        self.env.data.time = 7.0
        self.env._prev_s = before
        mujoco.mj_forward(self.env.model, self.env.data)

        _, _, terminated, truncated, info = self.env.step([0.0, 0.0])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        violations = [
            event for event in info["events"]
            if event["event"] == "traffic_light_violation"
        ]
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["points"], -10)

    def test_leaving_track_penalizes_without_terminating(self):
        self.env.reset(seed=0)
        s = track.START_S + 0.6

        def place(lateral):
            x, y, heading = track.path_point(s)
            nx, ny = -math.sin(heading), math.cos(heading)
            self.env.data.qpos[0:3] = [
                x + nx * lateral,
                y + ny * lateral,
                0.052,
            ]
            self.env.data.qpos[3:7] = [
                math.cos(heading / 2),
                0,
                0,
                math.sin(heading / 2),
            ]
            self.env.data.qvel[:] = 0
            self.env._prev_s = s
            mujoco.mj_forward(self.env.model, self.env.data)

        place(0.35)
        _, _, terminated, truncated, info = self.env.step([0.0, 0.0])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(
            sum(event["event"] == "off_track_minor" for event in info["events"]),
            1,
        )

        # Remaining off-track does not repeatedly charge the same incident.
        self.env.step([0.0, 0.0])
        self.assertEqual(
            sum(event["event"] == "off_track_minor" for event in self.env.score.events),
            1,
        )

        # Returning to the road rearms the penalty for a later excursion.
        place(0.0)
        self.env.step([0.0, 0.0])
        place(-0.35)
        _, _, terminated, truncated, info = self.env.step([0.0, 0.0])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(
            sum(event["event"] == "off_track_minor" for event in info["events"]),
            2,
        )

    def test_tunnel_obstacle_uses_path_normal(self):
        self.env.reset(seed=2)
        pos = self.env.data.mocap_pos[self.env._t2_mocap]
        s, lateral = track.frenet(float(pos[0]), float(pos[1]))
        self.assertAlmostEqual(track.s_delta(s, track.T2_OBSTACLE_S), 0.0, places=6)
        self.assertAlmostEqual(abs(lateral), 0.09, places=6)

    def test_dynamic_bicycle_is_rotated_across_track(self):
        self.env.reset(seed=0)
        quat = self.env.data.mocap_quat[self.env._dyn_mocap]
        actual_yaw = 2 * math.atan2(float(quat[3]), float(quat[0]))
        expected_yaw = (
            track.path_point(track.DYN_OBSTACLE_S)[2]
            + track.BICYCLE_YAW_OFFSET
        )
        error = (actual_yaw - expected_yaw + math.pi) % (2 * math.pi) - math.pi
        self.assertAlmostEqual(error, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
