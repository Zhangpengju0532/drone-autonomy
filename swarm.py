import math

import numpy as np

import drone

HEIGHT = 2.0
MAX_SPEED = 1.2
GOAL_GAIN = 1.0
INFLUENCE = 2.2
REPULSION = 6.0
SIDESTEP = 1.0
COLLISION_DISTANCE = 0.6
CONTROL_EVERY = 5


class Follower(drone.Controller):
    def track(self, ref_pos, ref_vel, vehicle):
        error = ref_pos - vehicle.pos
        self.accumulated = np.clip(self.accumulated + error * drone.DT, -2.0, 2.0)
        wanted = drone.KP_POS * error + drone.KD_POS * (ref_vel - vehicle.vel) + drone.KI_POS * self.accumulated
        roll, pitch, yaw = vehicle.angles
        thrust = drone.MASS * (wanted[2] + drone.GRAVITY) / max(0.5, math.cos(roll) * math.cos(pitch))
        thrust = min(drone.MAX_THRUST, max(0.0, thrust))
        want = np.array([(wanted[0] * math.sin(yaw) - wanted[1] * math.cos(yaw)) / drone.GRAVITY,
                         (wanted[0] * math.cos(yaw) + wanted[1] * math.sin(yaw)) / drone.GRAVITY, 0.0])
        want = np.clip(want, -drone.MAX_TILT, drone.MAX_TILT)
        torque = drone.INERTIA * (drone.KP_ATT * (want - vehicle.angles) - drone.KD_ATT * vehicle.rates)
        return thrust, np.clip(torque, -drone.MAX_TORQUE, drone.MAX_TORQUE)


def avoidance(index, positions):
    push = np.zeros(3)
    for other, position in enumerate(positions):
        if other == index:
            continue
        away = positions[index] - position
        away[2] = 0.0
        distance = float(np.linalg.norm(away))
        if distance < 1e-6 or distance >= INFLUENCE:
            continue
        strength = REPULSION * (1.0 / distance - 1.0 / INFLUENCE)
        unit = away / distance
        push += strength * unit + SIDESTEP * strength * np.array([-unit[1], unit[0], 0.0])
    return push


def simulate(starts, goal_at, duration, avoid):
    count = len(starts)
    vehicles = []
    for start in starts:
        vehicle = drone.Vehicle()
        vehicle.pos = np.array(start, dtype=float)
        vehicles.append(vehicle)
    controllers = [Follower() for _ in range(count)]
    targets = [np.array(start, dtype=float) for start in starts]
    speeds = [np.zeros(3) for _ in range(count)]
    log = []
    steps = int(round(duration / drone.DT))
    for step in range(steps):
        t = step * drone.DT
        if step % CONTROL_EVERY == 0:
            positions = [v.pos.copy() for v in vehicles]
            for i in range(count):
                goal, feed_forward = goal_at(i, t)
                wanted = feed_forward + GOAL_GAIN * (goal - targets[i])
                if avoid:
                    wanted = wanted + avoidance(i, positions)
                speed = float(np.linalg.norm(wanted))
                if speed > MAX_SPEED:
                    wanted = wanted * MAX_SPEED / speed
                speeds[i] = wanted
                targets[i] = targets[i] + wanted * drone.DT * CONTROL_EVERY
            if step % 50 == 0:
                log.append(np.concatenate([[t]] + positions + [v.angles.copy() for v in vehicles]))
        for i in range(count):
            thrust, torque = controllers[i].track(targets[i], speeds[i], vehicles[i])
            vehicles[i].step(thrust, torque)
    return np.array(log)


def positions_of(log, count):
    return log[:, 1:1 + 3 * count].reshape(len(log), count, 3)


def angles_of(log, count):
    return log[:, 1 + 3 * count:1 + 6 * count].reshape(len(log), count, 3)


def closest_pair(log, count):
    positions = positions_of(log, count)
    best = float("inf")
    for i in range(count):
        for j in range(i + 1, count):
            best = min(best, float(np.linalg.norm(positions[:, i] - positions[:, j], axis=1).min()))
    return best