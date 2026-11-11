import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import swarm

SWAP_COUNT = 6
SWAP_RADIUS = 5.0
SWAP_TIME = 24.0
FORMATION_COUNT = 5
FORMATION_TIME = 40.0
CENTRE = np.array([10.0, 10.0, swarm.HEIGHT])
COLORS = ["#2a78d6", "#eb6834", "#1baf6c", "#8a5cd0", "#d6453d", "#3d3d3d"]
SHAPES = {
    "line": [(0.0, 5.0), (0.0, 2.5), (0.0, 0.0), (0.0, -2.5), (0.0, -5.0)],
    "wedge": [(-3.6, 3.6), (-1.8, 1.8), (0.0, 0.0), (-1.8, -1.8), (-3.6, -3.6)],
    "ring": [(2.2 * math.cos(2 * math.pi * k / 5), 2.2 * math.sin(2 * math.pi * k / 5)) for k in (1, 2, 0, 3, 4)],
}
SCHEDULE = [(0.0, "line"), (8.0, "line"), (16.0, "wedge"), (24.0, "wedge"), (32.0, "ring"), (40.0, "ring")]


def ring_point(k, count, radius):
    angle = 2 * math.pi * k / count
    return CENTRE + np.array([radius * math.cos(angle), radius * math.sin(angle), 0.0])


def swap_goal(i, t):
    return ring_point(i + SWAP_COUNT // 2, SWAP_COUNT, SWAP_RADIUS), np.zeros(3)


def formation_centre(t):
    x = 4.0 + 12.0 * t / FORMATION_TIME
    y = 10.0 + 2.0 * math.sin(2 * math.pi * t / FORMATION_TIME)
    vx = 12.0 / FORMATION_TIME
    vy = 2.0 * 2 * math.pi / FORMATION_TIME * math.cos(2 * math.pi * t / FORMATION_TIME)
    return np.array([x, y, swarm.HEIGHT]), np.array([vx, vy, 0.0])


def slot(i, t):
    for (t0, a), (t1, b) in zip(SCHEDULE, SCHEDULE[1:]):
        if t <= t1:
            s = (t - t0) / (t1 - t0)
            s = 3 * s ** 2 - 2 * s ** 3
            return (1 - s) * np.array(SHAPES[a][i]) + s * np.array(SHAPES[b][i])
    return np.array(SHAPES[SCHEDULE[-1][1]][i])


def formation_goal(i, t):
    centre, velocity = formation_centre(t)
    offset = slot(i, t)
    return centre + np.array([offset[0], offset[1], 0.0]), velocity


def main():
    os.makedirs("results", exist_ok=True)
    starts = [ring_point(k, SWAP_COUNT, SWAP_RADIUS) for k in range(SWAP_COUNT)]
    print("position swap: %d vehicles on a %.0f m circle, each flies to the opposite point" % (SWAP_COUNT, SWAP_RADIUS))
    print("  avoidance  closest pair (m)  collisions (< %.1f m)  farthest from goal at the end (m)" % swarm.COLLISION_DISTANCE)
    logs = {}
    for avoid in (False, True):
        log = swarm.simulate(starts, swap_goal, SWAP_TIME, avoid)
        gap = swarm.closest_pair(log, SWAP_COUNT)
        final = swarm.positions_of(log, SWAP_COUNT)[-1]
        miss = max(float(np.linalg.norm(final[i] - swap_goal(i, 0.0)[0])) for i in range(SWAP_COUNT))
        print("  %-9s  %16.2f  %21s  %32.2f" % ("on" if avoid else "off", gap, "yes" if gap < swarm.COLLISION_DISTANCE else "no", miss))
        logs["swap_on" if avoid else "swap_off"] = log
    starts = [formation_goal(i, 0.0)[0] for i in range(FORMATION_COUNT)]
    log = swarm.simulate(starts, formation_goal, FORMATION_TIME, True)
    logs["formation"] = log
    positions = swarm.positions_of(log, FORMATION_COUNT)
    errors = np.array([[np.linalg.norm(positions[k, i] - formation_goal(i, log[k, 0])[0]) for i in range(FORMATION_COUNT)] for k in range(len(log))])
    print("formation flight: %d vehicles, line -> wedge -> ring while moving" % FORMATION_COUNT)
    print("  distance to the assigned slot: rms %.2f m, max %.2f m | closest pair %.2f m" % (
        np.sqrt((errors ** 2).mean()), errors.max(), swarm.closest_pair(log, FORMATION_COUNT)))
    np.savez("results/swarm_log.npz", **logs)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    for ax, key, title in ((axes[0], "swap_off", "Swap, avoidance off"), (axes[1], "swap_on", "Swap, avoidance on")):
        positions = swarm.positions_of(logs[key], SWAP_COUNT)
        for i in range(SWAP_COUNT):
            ax.plot(positions[:, i, 0], positions[:, i, 1], color=COLORS[i], linewidth=1.6)
            ax.plot(positions[0, i, 0], positions[0, i, 1], marker="o", color=COLORS[i], markersize=6)
            ax.plot(positions[-1, i, 0], positions[-1, i, 1], marker="s", color=COLORS[i], markersize=6, markerfacecolor="white")
        ax.set_xlim(3.5, 16.5)
        ax.set_ylim(3.5, 16.5)
        ax.set_title(title, fontsize=10, loc="left")
    positions = swarm.positions_of(logs["formation"], FORMATION_COUNT)
    for i in range(FORMATION_COUNT):
        axes[2].plot(positions[:, i, 0], positions[:, i, 1], color=COLORS[i], linewidth=1.2)
    for moment in (0.0, 20.0, 39.5):
        k = int(np.argmin(np.abs(logs["formation"][:, 0] - moment)))
        order = [0, 1, 2, 3, 4] if moment < 30 else [2, 0, 1, 3, 4, 2]
        axes[2].plot(positions[k, order, 0], positions[k, order, 1], color="0.3", linewidth=1)
        for i in range(FORMATION_COUNT):
            axes[2].plot(positions[k, i, 0], positions[k, i, 1], marker="o", color=COLORS[i], markersize=6)
    axes[2].set_xlim(0, 20)
    axes[2].set_ylim(2, 18)
    axes[2].set_title("Formation: line, wedge, ring", fontsize=10, loc="left")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.grid(color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig("results/s10_swarm.png", dpi=200)
    print("figure written to results/s10_swarm.png, log written to results/swarm_log.npz")


main()