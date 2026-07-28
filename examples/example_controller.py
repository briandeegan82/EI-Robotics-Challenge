"""A reference autonomous controller that completes the EI Robotics Challenge.

Deliberately simple — pure pursuit on the track centerline plus a small state
machine — so every team can read it end to end. It is a starting point, not a
benchmark: it steers using ``info["privileged"]`` ground truth (pose, arc
length, obstacle positions) that the REAL robot will not have. Your job is to
replace those lookups with perception:

  * line following        -> camera (see vision_lane_keeper.py)
  * obstacle detection    -> camera (depth sensors are prohibited!)

Course plan:
    1. Pure-pursuit the white line; slow down for the choke point and tunnels.
    2. Dodge the tunnel #2 obstacle to the free side.
    3. Handle the dynamic bicycle: swerve around it if parked, or wait for a
       gap if it's crossing the track.
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
TUNNEL_SPEED = 0.9
DODGE_SPEED = 0.7
CURVE_SPEED = 0.8
MAX_SPEED = 2.2       # matches env throttle mapping
MAX_STEER = 0.55

DYN_ZONE = (track.DYN_OBSTACLE_S - 1.0, track.DYN_OBSTACLE_S - 0.35)
OFFSET_RESPONSE = 0.08


class ExampleController:
    def __init__(self):
        self.dodge_offset = 0.0

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
        if track.in_range(s, (track.TUNNEL_1[0] - 0.3, track.TUNNEL_1[1])):
            target_speed = TUNNEL_SPEED

        # --- tunnel #2 obstacle: dodge to the free side -------------------
        # (real robot: detect the box with the camera instead of p["t2_side"])
        if track.in_range(s, (track.TUNNEL_2[0] - 0.6, track.T2_OBSTACLE_S + 0.35)):
            offset = -p["t2_side"] * 0.15
            target_speed = DODGE_SPEED

        # --- dynamic bicycle ----------------------------------------------
        # (real robot: detect it with the camera)
        dyn = p["dyn_obstacle"]
        if dyn["moving"]:
            # crossing obstacle: wait before the zone until it's out of the way
            if track.in_range(s, DYN_ZONE) and abs(dyn["lateral"]) < 0.20:
                gap = track.s_delta(track.DYN_OBSTACLE_S - 0.45, s)
                target_speed = min(target_speed, float(np.clip(2.0 * gap, 0.0, CRUISE)))
        else:
            # parked obstacle: swerve around it on the opposite side
            if track.in_range(s, (track.DYN_OBSTACLE_S - 1.0, track.DYN_OBSTACLE_S + 0.45)):
                offset = -np.sign(dyn["lateral"]) * 0.13
                target_speed = min(target_speed, DODGE_SPEED)

        # --- functional traffic light -------------------------------------
        light = p["traffic_light"]
        light_gap = track.s_delta(light["stop_s"], s)
        if light["state"] == "red" and 0 < light_gap < 0.9:
            target_speed = min(target_speed, float(np.clip(1.5 * (light_gap - 0.08), 0.0, 0.8)))

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
