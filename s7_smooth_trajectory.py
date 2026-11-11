import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import planner
import trajectory
import world

WANTED_CLEARANCE = 0.45


def main():
    waypoints, raw, expanded = planner.plan()
    first = trajectory.Trajectory(waypoints)
    gap, when = first.clearance()
    print("waypoints from the planner: %d, path length %.2f m" % (len(waypoints), planner.length(waypoints)))
    print("first smooth trajectory: closest approach to an obstacle %.2f m at t = %.1f s" % (gap, when))
    final, added = trajectory.smooth(waypoints, WANTED_CLEARANCE)
    gap, when = final.clearance()
    ts, pos, vel, acc = final.sample()
    travelled = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    print("after adding %d extra waypoint(s): closest approach %.2f m" % (added, gap))
    print("final trajectory: duration %.1f s, length %.2f m" % (final.duration, travelled))
    print("  top speed %.2f m/s, top acceleration %.2f m/s2" % (np.linalg.norm(vel, axis=1).max(), np.linalg.norm(acc, axis=1).max()))
    print("  tilt needed for the top acceleration: about %.1f deg" % np.degrees(np.arctan(np.linalg.norm(acc[:, :2], axis=1).max() / 9.81)))
    print("  speed at start %.3f m/s, at end %.3f m/s" % (np.linalg.norm(vel[0]), np.linalg.norm(vel[-1])))

    os.makedirs("results", exist_ok=True)
    way = np.array(final.points)
    fig = plt.figure(figsize=(10.0, 4.2))
    top = fig.add_axes([0.05, 0.12, 0.38, 0.8])
    for x0, y0, z0, x1, y1, z1 in world.BOXES:
        top.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="0.6" if z1 > 3 else "0.82", edgecolor="0.35", linewidth=0.5))
    top.plot(way[:, 0], way[:, 1], color="#eb6834", linewidth=1.2, marker="o", markersize=4, label="waypoints")
    top.plot(pos[:, 0], pos[:, 1], color="#2a78d6", linewidth=2, label="smooth trajectory")
    top.set_xlim(0, 20)
    top.set_ylim(0, 20)
    top.set_aspect("equal")
    top.set_xlabel("x (m)")
    top.set_ylabel("y (m)")
    top.set_title("Top view", fontsize=10, loc="left")
    top.legend(frameon=False, fontsize=9, loc="lower right")
    panels = [
        (fig.add_axes([0.52, 0.70, 0.45, 0.22]), pos[:, 2], "height (m)"),
        (fig.add_axes([0.52, 0.41, 0.45, 0.22]), np.linalg.norm(vel, axis=1), "speed (m/s)"),
        (fig.add_axes([0.52, 0.12, 0.45, 0.22]), np.linalg.norm(acc, axis=1), "acceleration (m/s$^2$)"),
    ]
    for ax, values, label in panels:
        for moment in final.starts[1:-1]:
            ax.axvline(moment, color="0.88", linewidth=0.8)
        ax.plot(ts, values, color="#2a78d6", linewidth=1.6)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xlim(0, final.duration)
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    panels[0][0].set_title("Along the trajectory (vertical lines: waypoints)", fontsize=10, loc="left")
    panels[0][0].set_xticklabels([])
    panels[1][0].set_xticklabels([])
    panels[2][0].set_xlabel("time (s)")
    fig.savefig("results/s7_smooth_trajectory.png", dpi=200)
    print("figure written to results/s7_smooth_trajectory.png")


main()