"""A reference autonomous controller that completes the Gauntlet.

This is deliberately simple — a hand-tuned state machine — so every team can
read it end to end. It is a starting point, not a benchmark: it uses
``info["privileged"]`` ground truth (car pose, light state, bicycle position)
that the REAL robot will not have. Your job is to replace those lookups with
perception: the camera for the traffic light, rangefinders/camera for the
bicycle and the tunnel obstacle.

Course plan:
    1. Lane-keep down the middle of the road.
    2. Stop at the stop line (x=5.5) unless the light is green.
    3. Wait before the bicycle crossing (x=12) until the bicycle is clear.
    4. In the tunnel, use the rangefinder fan to spot the obstacle and pick
       the side with more clearance.
    5. Cross the finish line at x=28.
"""

import numpy as np

STOP_LINE_X = 5.5
CROSSING_STOP_X = 10.9
TUNNEL = (18.0, 24.0)
CRUISE = 0.65        # throttle for open road
TUNNEL_SPEED = 0.4   # slower while avoiding the obstacle


class ExampleController:
    def __init__(self):
        self.dodge_y = None   # lateral target while passing the tunnel obstacle

    def act(self, obs, info):
        """Return [steer, throttle], each in [-1, 1]."""
        rf = obs[0:11]                     # rangefinders, right (-60d) to left (+60d)
        x, y, _ = info["privileged"]["car_pos"]
        yaw = info["privileged"]["car_yaw"]
        light = info["privileged"]["light_state"]
        bike_y = info["privileged"]["bike_y"]

        target_y = 0.0
        throttle = CRUISE

        # --- 1. traffic light: stop at the line unless green -------------
        if x < STOP_LINE_X and light != "green":
            throttle = min(throttle, self._creep_to(STOP_LINE_X - 0.25, x))

        # --- 2. bicycle crossing: wait until the bike is off the road ----
        if x < CROSSING_STOP_X + 0.4 and abs(bike_y) < 1.6:
            throttle = min(throttle, self._creep_to(CROSSING_STOP_X, x))

        # --- 3. tunnel obstacle: dodge to the clearer side ---------------
        # The obstacle sits at x=21 on one side of the lane. Approaching down
        # the middle, the +/-12 deg rays (rf[4] right, rf[6] left) hit the
        # obstacle at ~2-3 m while the tunnel walls are >3.5 m away at that
        # angle, so whichever ray reads short tells us the obstacle's side.
        if TUNNEL[0] - 1.0 < x < TUNNEL[1]:
            throttle = min(throttle, TUNNEL_SPEED)
            if self.dodge_y is None and 18.2 < x < 19.9:
                right, left = rf[4], rf[6]
                if min(right, left) < 3.0:
                    self.dodge_y = 0.42 if right < left else -0.42
            if self.dodge_y is not None:
                target_y = self.dodge_y if x < 22.0 else 0.0
        elif x >= TUNNEL[1]:
            self.dodge_y = None

        # --- lane keeping: steer toward target_y, damped by heading ------
        steer = np.clip(-2.0 * (y - target_y) - 1.6 * yaw, -1, 1)
        return np.array([steer, throttle])

    @staticmethod
    def _creep_to(stop_x, x):
        """Throttle that slows smoothly and stops just before stop_x."""
        gap = stop_x - x
        if gap < 0.05:
            return -0.05          # gentle brake / hold
        return float(np.clip(0.55 * gap, 0.0, CRUISE))
