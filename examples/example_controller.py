"""A reference autonomous controller that completes the EI Robotics Challenge.

Deliberately simple — pure pursuit on the track centerline plus a small state
machine — so every team can read it end to end. It is a starting point, not a
benchmark: it steers using ``info["privileged"]`` ground truth (pose, arc
length, obstacle positions) that the REAL robot will not have. Your job is to
replace those lookups with perception:

  * line following        -> camera (see vision_lane_keeper.py)
  * obstacle detection    -> camera (depth sensors are prohibited!)

Course plan:
    1. Pure-pursuit the white line; slow down for the choke point and gate.
    2. Take the lane the fork sign indicates (a wall blocks the other one).
    3. Dodge the tunnel #2 obstacle to the free side.
    4. Handle the dynamic bicycle, which continuously crosses the road: hold
       short of it until a gap opens on the far side, then drive through.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from challenge import track

WHEELBASE = 0.15
LOOKAHEAD = 0.35
CRUISE = 2.0          # m/s on open track
CHOKE_SPEED = 0.9
GATE_SPEED = 0.9
DODGE_SPEED = 0.7
CURVE_SPEED = 0.8
MAX_SPEED = 2.2       # matches env throttle mapping
MAX_STEER = 0.55

DYN_ZONE = (track.DYN_OBSTACLE_S - 1.0, track.DYN_OBSTACLE_S + 0.30)
OFFSET_RESPONSE = 0.08
# Lead distance for the fork lane change: the car's chassis extends ~0.11 m
# ahead of the frenet reference point (its bumper reaches FORK_S[0] before
# its center does), and the smoothed dodge_offset (OFFSET_RESPONSE=0.08 per
# 50 Hz step) needs real distance to converge -- a short lead leaves the
# offset well under the lane target when the bumper reaches the divider's
# narrow entrance, clipping it. 1.3 m gives the filter time to settle.
FORK_LEAD = 1.3


class ExampleController:
    def __init__(self):
        self.dodge_offset = 0.0
        self.bike_cleared = False   # latched once we commit to crossing
        self.prev_bike_lat = 0.0

    def act(self, obs, info):
        """Return [steer, throttle], each in [-1, 1]."""
        p = info["privileged"]
        x, y, _ = p["car_pos"]
        yaw = p["car_yaw"]
        s = p["s"]

        target_speed = CRUISE
        offset = 0.0          # lateral offset from the line to aim for

        # --- section speed limits -----------------------------------------
        if track.in_range(s, (track.CHOKE[0] - 0.5, track.CHOKE[1])):
            target_speed = CHOKE_SPEED
        if abs(track.s_delta(s, track.GATE_S)) < 0.8:
            target_speed = GATE_SPEED

        # --- lane fork: take the side the sign indicates -------------------
        # A divider wall blocks the middle, so start aiming for the lane
        # centre a bit before the fork itself to avoid clipping its lead edge.
        if track.in_range(s, (track.FORK_S[0] - FORK_LEAD, track.FORK_S[1])):
            offset = track.FORK_LANE_OFFSET if p["fork_direction"] == "left" else -track.FORK_LANE_OFFSET
            target_speed = min(target_speed, DODGE_SPEED)

        # --- tunnel #2 obstacle: dodge to the free side -------------------
        # (real robot: detect the box with the camera instead of p["t2_side"]).
        # Start dodging only inside the lane-check-suspended zone so the swerve
        # itself never trips an off-track penalty.
        if track.in_range(s, (track.TUNNEL_2[0] - 0.5, track.T2_OBSTACLE_S + 0.35)):
            offset = -p["t2_side"] * 0.15
            target_speed = DODGE_SPEED

        # --- dynamic bicycle: it continuously crosses the road ------------
        # (real robot: detect it with the camera). Hold ~0.45 m short until a
        # gap opens on the far side and is still widening, then latch a commit
        # and drive through, hugging the open side for lateral margin (a
        # centered car barely clears the bike at its peak, so if we arrive
        # fast just as it starts to close we would otherwise clip it).
        dyn = p["dyn_obstacle"]
        approach = track.in_range(s, DYN_ZONE)
        if approach and not self.bike_cleared:
            moving_out = abs(dyn["lateral"]) > abs(self.prev_bike_lat)
            if abs(dyn["lateral"]) > 0.19 and moving_out:
                self.bike_cleared = True
            else:
                gap = track.s_delta(track.DYN_OBSTACLE_S - 0.45, s)
                target_speed = min(target_speed, float(np.clip(1.8 * gap, 0.0, CRUISE)))
        elif not approach:
            self.bike_cleared = False           # re-arm once past the obstacle
        if self.bike_cleared:
            offset = -np.sign(dyn["lateral"]) * 0.09
        self.prev_bike_lat = dyn["lateral"]

        # --- functional traffic light -------------------------------------
        # The signal starts red and only turns (permanently) green once the
        # car has sat stopped at the line for a few seconds -- the same
        # fixed toll every attempt, not a lucky pre-timed cycle. Brake to a
        # full stop on red and hold; the environment handles the wait and
        # flips the state once it's satisfied. Creep off once green.
        light = p["traffic_light"]
        light_gap = track.s_delta(light["stop_s"], s)
        if 0 < light_gap < 1.3:
            if light["state"] in ("red", "yellow"):
                target_speed = min(target_speed, float(np.clip(1.6 * (light_gap - 0.04), 0.0, 0.6)))
            else:
                # creep the last stretch so a flip to red right at the line can
                # still be braked to a stop before crossing it
                target_speed = min(target_speed, float(np.clip(0.55 * light_gap, 0.12, CRUISE)))

        # --- pure pursuit toward a point ahead on the (offset) line -------
        self.dodge_offset += OFFSET_RESPONSE * (offset - self.dodge_offset)
        s_t = s + LOOKAHEAD
        tx, ty, theading = track.path_point(s_t)
        _, _, path_heading = track.path_point(s)
        heading_change = (theading - path_heading + np.pi) % (2 * np.pi) - np.pi
        if abs(heading_change) > 0.12:
            target_speed = min(target_speed, CURVE_SPEED)
        tx += -np.sin(theading) * self.dodge_offset
        ty += np.cos(theading) * self.dodge_offset
        dx, dy = tx - x, ty - y
        local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
        curvature = 2 * local_y / max(local_x**2 + local_y**2, 1e-6)
        steer = np.arctan(WHEELBASE * curvature) / MAX_STEER

        return np.array([np.clip(steer, -1, 1), target_speed / MAX_SPEED])
