import math

import numpy as np

WHEELBASE = 2.7
MAX_STEER = math.radians(30)
LOOKAHEAD = 4.0
ROAD_START = 2.0
ROAD_END = 38.0
BRIDGE = (19.4, 3.0, 2.4, 20.6, 17.0, 3.2)


def road():
    xs = np.arange(ROAD_START, ROAD_END + 0.01, 0.25)
    return np.column_stack([xs, 10.0 + 5.0 * np.sin((xs - ROAD_START) / 6.0)])


def wanted_speed(x):
    if x < 12.0:
        return 3.0
    if x < 24.0:
        return 1.8
    return 4.0


class Car:
    def __init__(self):
        self.path = road()
        self.x, self.y = self.path[0]
        self.yaw = math.atan2(self.path[1, 1] - self.path[0, 1], self.path[1, 0] - self.path[0, 0])
        self.speed = 0.0
        self.steer = 0.0

    @property
    def finished(self):
        return self.x >= ROAD_END - 5.0

    @property
    def velocity(self):
        return np.array([self.speed * math.cos(self.yaw), self.speed * math.sin(self.yaw), 0.0])

    def step(self, dt):
        target_speed = 0.0 if self.finished else wanted_speed(self.x)
        self.speed += max(-2.0 * dt, min(1.5 * dt, target_speed - self.speed))
        i = int(np.argmin((self.path[:, 0] - self.x) ** 2 + (self.path[:, 1] - self.y) ** 2))
        target = self.path[-1]
        for point in self.path[i:]:
            if math.hypot(point[0] - self.x, point[1] - self.y) >= LOOKAHEAD:
                target = point
                break
        alpha = math.atan2(target[1] - self.y, target[0] - self.x) - self.yaw
        self.steer = max(-MAX_STEER, min(MAX_STEER, math.atan2(2 * WHEELBASE * math.sin(alpha), LOOKAHEAD)))
        self.x += self.speed * math.cos(self.yaw) * dt
        self.y += self.speed * math.sin(self.yaw) * dt
        self.yaw += self.speed / WHEELBASE * math.tan(self.steer) * dt