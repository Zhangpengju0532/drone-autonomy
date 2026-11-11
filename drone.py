import math

import numpy as np

MASS = 1.0
GRAVITY = 9.81
INERTIA = np.array([0.010, 0.010, 0.018])
DRAG = 0.25
MAX_THRUST = 2.5 * MASS * GRAVITY
MAX_TORQUE = np.array([1.0, 1.0, 0.3])
MAX_TILT = math.radians(35)
DT = 0.002
TAKEOFF_TIME = 4.0
LAP_TIME = 12.0
LAPS = 2
RAMP_TIME = 2.0
SIZE = 3.0
HEIGHT = 2.0
KP_POS = np.array([6.0, 6.0, 10.0])
KD_POS = np.array([4.5, 4.5, 6.0])
KI_POS = np.array([2.0, 2.0, 3.0])
KP_ATT = np.array([120.0, 120.0, 40.0])
KD_ATT = np.array([22.0, 22.0, 12.0])
DOWN = np.array([0.0, 0.0, -GRAVITY])


def flight_time():
    return TAKEOFF_TIME + RAMP_TIME / 2 + LAPS * LAP_TIME


def reference(t):
    if t < TAKEOFF_TIME:
        s = t / TAKEOFF_TIME
        blend = 3 * s ** 2 - 2 * s ** 3
        rate = (6 * s - 6 * s ** 2) / TAKEOFF_TIME
        return np.array([0.0, 0.0, HEIGHT * blend]), np.array([0.0, 0.0, HEIGHT * rate]), np.zeros(3)
    w = 2 * math.pi / LAP_TIME
    tau = t - TAKEOFF_TIME
    if tau < RAMP_TIME:
        a = w * tau * tau / (2 * RAMP_TIME)
        da = w * tau / RAMP_TIME
        dda = w / RAMP_TIME
    else:
        a = w * (tau - RAMP_TIME / 2)
        da = w
        dda = 0.0
    pos = np.array([SIZE * math.sin(a), SIZE / 2 * math.sin(2 * a), HEIGHT + 0.5 * math.sin(a)])
    first = np.array([SIZE * math.cos(a), SIZE * math.cos(2 * a), 0.5 * math.cos(a)])
    second = np.array([-SIZE * math.sin(a), -2 * SIZE * math.sin(2 * a), -0.5 * math.sin(a)])
    return pos, first * da, first * dda + second * da * da


def rotation(angles):
    roll, pitch, yaw = angles
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


class Vehicle:
    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.angles = np.zeros(3)
        self.rates = np.zeros(3)
        self.accel = np.zeros(3)

    def step(self, thrust, torque, wind=None):
        air = self.vel if wind is None else self.vel - wind
        body_z = rotation(self.angles)[:, 2]
        self.accel = body_z * thrust / MASS + DOWN - DRAG * air / MASS
        rate_dot = (torque - np.cross(self.rates, INERTIA * self.rates)) / INERTIA
        roll, pitch, yaw = self.angles
        cr, sr = math.cos(roll), math.sin(roll)
        cp, tp = math.cos(pitch), math.tan(pitch)
        angle_dot = np.array([
            self.rates[0] + sr * tp * self.rates[1] + cr * tp * self.rates[2],
            cr * self.rates[1] - sr * self.rates[2],
            (sr * self.rates[1] + cr * self.rates[2]) / cp,
        ])
        self.vel = self.vel + self.accel * DT
        self.pos = self.pos + self.vel * DT
        self.rates = self.rates + rate_dot * DT
        self.angles = self.angles + angle_dot * DT
        if self.pos[2] < 0.0:
            self.pos[2] = 0.0
            self.vel[2] = max(0.0, self.vel[2])
            self.accel[2] = max(0.0, self.accel[2])


class Controller:
    def __init__(self):
        self.accumulated = np.zeros(3)

    def command(self, t, pos, vel, angles, rates):
        ref_pos, ref_vel, ref_acc = reference(t)
        error = ref_pos - pos
        self.accumulated = np.clip(self.accumulated + error * DT, -2.0, 2.0)
        wanted = KP_POS * error + KD_POS * (ref_vel - vel) + KI_POS * self.accumulated + ref_acc
        roll, pitch, yaw = angles
        thrust = MASS * (wanted[2] + GRAVITY) / max(0.5, math.cos(roll) * math.cos(pitch))
        thrust = min(MAX_THRUST, max(0.0, thrust))
        want_roll = (wanted[0] * math.sin(yaw) - wanted[1] * math.cos(yaw)) / GRAVITY
        want_pitch = (wanted[0] * math.cos(yaw) + wanted[1] * math.sin(yaw)) / GRAVITY
        want = np.clip(np.array([want_roll, want_pitch, 0.0]), -MAX_TILT, MAX_TILT)
        torque = INERTIA * (KP_ATT * (want - angles) - KD_ATT * rates)
        return thrust, np.clip(torque, -MAX_TORQUE, MAX_TORQUE)