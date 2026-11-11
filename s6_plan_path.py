import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import planner
import world


def draw_boxes(ax):
    for x0, y0, z0, x1, y1, z1 in world.BOXES:
        ax.bar3d(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0, color="0.6", alpha=0.35, edgecolor="0.35", linewidth=0.4, shade=False)


def main():
    grid = world.occupancy()
    start = world.cell_of(world.START)
    goal = world.cell_of(world.GOAL)
    print("grid: %d x %d x %d cells of %.1f m, %d cells blocked (%.0f%%)" % (*grid.shape, world.CELL, grid.sum(), 100 * grid.mean()))
    print("search          path length (m)  cells examined  time (s)")
    for name, flag in (("Dijkstra", False), ("A*", True)):
        clock = time.time()
        cells, expanded = planner.search(grid, start, goal, use_heuristic=flag)
        elapsed = time.time() - clock
        points = [world.centre_of(c) for c in cells]
        print("%-14s  %15.2f  %14d  %8.2f" % (name, planner.length(points), expanded, elapsed))
    waypoints, raw, expanded = planner.plan()
    print("A* path: %d points, %.2f m | after shortcutting: %d waypoints, %.2f m | straight line: %.2f m" % (
        len(raw), planner.length(raw), len(waypoints), planner.length(waypoints), np.linalg.norm(world.GOAL - world.START)))
    print("closest approach of the waypoint path to an obstacle: %.2f m" % min(
        world.distance_to_obstacles(a + (b - a) * s) for a, b in zip(waypoints, waypoints[1:]) for s in np.linspace(0, 1, 50)))
    for point in waypoints:
        print("  waypoint  x %5.2f  y %5.2f  z %5.2f" % tuple(point))

    os.makedirs("results", exist_ok=True)
    raw = np.array(raw)
    way = np.array(waypoints)
    fig = plt.figure(figsize=(10.0, 4.4))
    space = fig.add_axes([0.0, 0.02, 0.56, 0.96], projection="3d")
    draw_boxes(space)
    space.plot(raw[:, 0], raw[:, 1], raw[:, 2], color="#eb6834", linewidth=1.2, label="A* path")
    space.plot(way[:, 0], way[:, 1], way[:, 2], color="#2a78d6", linewidth=2, marker="o", markersize=4, label="waypoints")
    space.scatter(*world.START, color="0.1", s=30)
    space.scatter(*world.GOAL, color="0.1", s=30, marker="*")
    space.set_xlim(0, 20)
    space.set_ylim(0, 20)
    space.set_zlim(0, 6)
    space.set_box_aspect((20, 20, 6), zoom=0.95)
    space.set_xticks([0, 5, 10, 15, 20])
    space.set_yticks([0, 5, 10, 15, 20])
    space.view_init(elev=38, azim=-58)
    space.set_xlabel("x (m)")
    space.set_ylabel("y (m)")
    space.set_zticks([0, 3, 6])
    space.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.05, 0.98))
    top = fig.add_axes([0.62, 0.12, 0.35, 0.78])
    for x0, y0, z0, x1, y1, z1 in world.BOXES:
        top.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="0.6" if z1 > 3 else "0.82", edgecolor="0.35", linewidth=0.5))
    top.plot(raw[:, 0], raw[:, 1], color="#eb6834", linewidth=1.2)
    top.plot(way[:, 0], way[:, 1], color="#2a78d6", linewidth=2, marker="o", markersize=4)
    top.plot(*world.START[:2], marker="o", color="0.1", markersize=6)
    top.plot(*world.GOAL[:2], marker="*", color="0.1", markersize=10)
    top.set_xlim(0, 20)
    top.set_ylim(0, 20)
    top.set_aspect("equal")
    top.set_xlabel("x (m)")
    top.set_ylabel("y (m)")
    top.set_title("Top view (light grey: low obstacles)", fontsize=10, loc="left")
    fig.savefig("results/s6_plan_path.png", dpi=200)
    print("figure written to results/s6_plan_path.png")


main()