import math
import unittest

from challenge import track


class TrackGeometryTests(unittest.TestCase):
    def test_centerline_closes(self):
        start = track.path_point(0.0)
        end = track.path_point(track.TOTAL)
        self.assertAlmostEqual(start[0], end[0], places=9)
        self.assertAlmostEqual(start[1], end[1], places=9)

    def test_path_frenet_round_trip(self):
        for i in range(1000):
            s = track.TOTAL * i / 1000
            x, y, _ = track.path_point(s)
            projected_s, lateral = track.frenet(x, y)
            self.assertAlmostEqual(track.s_delta(projected_s, s), 0.0, places=7)
            self.assertAlmostEqual(lateral, 0.0, places=7)

    def test_signed_lateral_offsets(self):
        for i in range(200):
            s = track.TOTAL * (i + 0.5) / 200
            x, y, heading = track.path_point(s)
            nx, ny = -math.sin(heading), math.cos(heading)
            for lateral in (-0.05, 0.05):
                projected_s, projected_lateral = track.frenet(
                    x + nx * lateral,
                    y + ny * lateral,
                )
                self.assertLess(abs(track.s_delta(projected_s, s)), 0.015)
                self.assertAlmostEqual(projected_lateral, lateral, delta=0.002)

    def test_features_follow_driving_order(self):
        sequence = [
            track.START_S,
            track.CHOKE[0],
            track.SPEED[0],
            track.DYN_OBSTACLE_S,
            track.GATE_S,
            track.SHINE[0],
            track.TUNNEL_2[0],
            track.TRAFFIC_STOP_S,
        ]
        self.assertEqual(sequence, sorted(sequence))
        self.assertGreater(track.TOTAL, sequence[-1])

    def test_course_has_safe_road_and_turn_dimensions(self):
        self.assertGreaterEqual(track.CORNER_RADIUS, 0.35)
        self.assertGreater(track.ROAD_HALF_WIDTH * 2, track.CAR_HALF_WIDTH * 2)
        for segment in track.SEGMENTS:
            self.assertGreater(segment.length, 0.01)

    def test_dynamic_bicycle_is_clear_of_gate(self):
        x, y, heading = track.path_point(track.DYN_OBSTACLE_S)
        self.assertAlmostEqual(x, 0.80, places=6)
        self.assertAlmostEqual(y, 0.20, places=6)
        self.assertAlmostEqual(abs(heading), math.pi, places=6)
        self.assertGreater(abs(track.s_delta(track.GATE_S, track.DYN_OBSTACLE_S)), 1.0)


if __name__ == "__main__":
    unittest.main()
