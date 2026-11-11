import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import drone
import ground
import swarm

FOLLOW_HEIGHT = 3.0
BRIDGE_MARGIN = 1.6
CLIMB_DISTANCE = 5.0
CLIMB_RATE = 1.5
DURATION = 18.0


def wanted_height(x):
    x0, y0, z0, x1, y1, z1 = ground.BRIDGE
    if x0 - CLIMB_DISTANCE < x < x1 + 1.5:
        return z1 + BRIDGE_MARGIN
    return FOLLOW_HEIGHT


def main():
    car = ground.Car()
    vehicle = drone.Vehicle()
    vehicle.pos = np.array([car.x, car.y, FOLLOW_HEIGHT])
    controller = swarm.Follower()
    height = FOLLOW_HEIGHT
    rows = []
    steps = int(round(DURATION / drone.DT))
    for step in range(steps):
        t = step * drone.DT
        goal_height = wanted_height(car.x)
        climb = max(-CLIMB_RATE, min(CLIMB_RATE, (goal_height - height) * 3.0))
        height += climb * drone.DT
        ref_pos = np.array([car.x, car.y, height])
        ref_vel = car.velocity + np.array([0.0, 0.0, climb])
        thrust, torque = controller.track(ref_pos, ref_vel, vehicle)
        if step % 10 == 0:
            rows.append(np.concatenate([[t, car.x, car.y, car.yaw, car.speed, car.steer], vehicle.pos, vehicle.angles]))
        vehicle.step(thrust, torque)
        car.step(drone.DT)
    log = np.array(rows)

    t = log[:, 0]
    offset = np.linalg.norm(log[:, 6:8] - log[:, 1:3], axis=1)
    x0, y0, z0, x1, y1, z1 = ground.BRIDGE
    over = (log[:, 6] > x0 - 0.5) & (log[:, 6] < x1 + 0.5)
    print("the car drives %.1f m in %.1f s at 3.0, 1.8 and 4.0 m/s in three sections, then stops" % (
        np.linalg.norm(np.diff(log[:, 1:3], axis=0), axis=1).sum(), t[-1]))
    print("horizontal distance between vehicle and car: rms %.2f m, max %.2f m" % (np.sqrt((offset ** 2).mean()), offset.max()))
    print("bridge deck top at %.1f m: the car passes under it, the vehicle crosses at %.2f m, clearance %.2f m" % (
        z1, log[over, 8].min(), log[over, 8].min() - z1))
    print("height of the vehicle above the car: before the bridge %.2f m, after the bridge %.2f m" % (log[100, 8], log[-1, 8]))
    print("largest tilt of the vehicle: %.1f deg" % np.degrees(np.abs(log[:, 9:11]).max()))

    os.makedirs("results", exist_ok=True)
    np.savez("results/follow_log.npz", log=log)
    fig, (top, side) = plt.subplots(2, 1, figsize=(9.0, 5.0), gridspec_kw={"height_ratios": [1.4, 1]})
    path = ground.road()
    top.plot(path[:, 0], path[:, 1], color="0.75", linewidth=7, solid_capstyle="round")
    top.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="0.55", edgecolor="0.3", linewidth=0.6))
    top.plot(log[:, 1], log[:, 2], color="0.1", linewidth=1.2, label="car")
    top.plot(log[:, 6], log[:, 7], color="#2a78d6", linewidth=1.6, linestyle=(0, (4, 2)), label="vehicle")
    top.set_aspect("equal")
    top.set_xlim(0, 40)
    top.set_ylim(2.5, 17.5)
    top.set_ylabel("y (m)")
    top.set_title("Top view (grey band: road, dark bar: bridge)", fontsize=10, loc="left")
    top.legend(frameon=False, fontsize=9, loc="lower left", ncol=2)
    side.add_patch(plt.Rectangle((x0, z0), x1 - x0, z1 - z0, facecolor="0.55", edgecolor="0.3", linewidth=0.6))
    side.plot(log[:, 1], np.full(len(log), 0.7), color="0.1", linewidth=1.2, label="car roof")
    side.plot(log[:, 6], log[:, 8], color="#2a78d6", linewidth=1.8, label="vehicle")
    side.set_xlim(0, 40)
    side.set_ylim(0, 6)
    side.set_xlabel("x (m)")
    side.set_ylabel("height (m)")
    side.set_title("Side view", fontsize=10, loc="left")
    side.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    for ax in (top, side):
        ax.grid(color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        for edge in ("top", "right"):
            ax.spines[edge].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/s12_follow_car.png", dpi=200)
    print("figure written to results/s12_follow_car.png, log written to results/follow_log.npz")


main()