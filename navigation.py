import math

import numpy as np

import drone
import sensors

ASSUMED_ACCEL_NOISE = 0.5
ASSUMED_GYRO_NOISE = 0.02
ASSUMED_ACCEL_BIAS_DRIFT = 0.002
ASSUMED_GYRO_BIAS_DRIFT = 0.0002


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def small_rotation(v):
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3) + skew(v)
    k = skew(v / angle)
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def euler_angles(r):
    return np.array([math.atan2(r[2, 1], r[2, 2]), -math.asin(max(-1.0, min(1.0, r[2, 0]))), math.atan2(r[1, 0], r[0, 0])])


class NavigationFilter:
    def __init__(self, first_gps):
        self.pos = np.array(first_gps, dtype=float)
        self.vel = np.zeros(3)
        self.r = np.eye(3)
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.rates = np.zeros(3)
        doubt = np.concatenate([sensors.GPS_NOISE, [0.5] * 3, np.radians([3.0, 3.0, 10.0]), [0.2] * 3, [0.03] * 3])
        self.P = np.diag(doubt ** 2)

    @property
    def angles(self):
        return euler_angles(self.r)

    def predict(self, accel_reading, gyro_reading):
        dt = drone.DT
        force = accel_reading - self.accel_bias
        self.rates = gyro_reading - self.gyro_bias
        accel = self.r @ force + drone.DOWN
        self.pos = self.pos + self.vel * dt + 0.5 * accel * dt * dt
        self.vel = self.vel + accel * dt
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -self.r @ skew(force) * dt
        F[3:6, 9:12] = -self.r * dt
        F[6:9, 6:9] = np.eye(3) - skew(self.rates) * dt
        F[6:9, 12:15] = -np.eye(3) * dt
        self.r = self.r @ small_rotation(self.rates * dt)
        self.P = F @ self.P @ F.T
        self.P[3:6, 3:6] += np.eye(3) * (ASSUMED_ACCEL_NOISE * dt) ** 2
        self.P[6:9, 6:9] += np.eye(3) * (ASSUMED_GYRO_NOISE * dt) ** 2
        self.P[9:12, 9:12] += np.eye(3) * ASSUMED_ACCEL_BIAS_DRIFT ** 2 * dt
        self.P[12:15, 12:15] += np.eye(3) * ASSUMED_GYRO_BIAS_DRIFT ** 2 * dt

    def apply(self, H, residual, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ residual
        self.P = (np.eye(15) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        self.pos = self.pos + dx[0:3]
        self.vel = self.vel + dx[3:6]
        self.r = self.r @ small_rotation(dx[6:9])
        self.accel_bias = self.accel_bias + dx[9:12]
        self.gyro_bias = self.gyro_bias + dx[12:15]

    def correct_gps(self, reading):
        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3)
        self.apply(H, reading - self.pos, np.diag(sensors.GPS_NOISE ** 2))

    def correct_compass(self, reading):
        H = np.zeros((1, 15))
        H[0, 6:9] = self.r[2, :]
        residual = (reading - self.angles[2] + math.pi) % (2 * math.pi) - math.pi
        self.apply(H, np.array([residual]), np.array([[sensors.COMPASS_NOISE ** 2]]))