import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import drone
import navigation
import planner
import sensors
import trajectory
import world

PLANNING_MARGIN = 1.0
VEHICLE_RADIUS = 0.3
WAIT_TIME = 6.0
CLIMB_TIME = 3.0
HOVER_TIME = 3.0


def plan_route():
    grid = world.occupancy(safety=PLANNING_MARGIN)
    cells, expanded = planner.search(grid, world.cell_of(world.START), world.cell_of(world.GOAL))
    points = [world.START.copy()] + [world.centre_of(c) for c in cells[1:-1]] + [world.GOAL.copy()]
    waypoints = planner.shortcut(grid, points)
    route, added = trajectory.smooth(waypoints, PLANNING_MARGIN - 0.1)
    return route, len(waypoints), added


def make_reference(route):
    ground = np.array([world.START[0], world.START[1], 0.0])

    def reference(t):
        if t < WAIT_TIME:
            return ground, np.zeros(3), np.zeros(3)
        if t < WAIT_TIME + CLIMB_TIME:
            s = (t - WAIT_TIME) / CLIMB_TIME
            height = world.START[2] * (3 * s ** 2 - 2 * s ** 3)
            rate = world.START[2] * (6 * s - 6 * s ** 2) / CLIMB_TIME
            return np.array([ground[0], ground[1], height]), np.array([0.0, 0.0, rate]), np.zeros(3)
        return route.evaluate(t - WAIT_TIME - CLIMB_TIME)

    return reference


def fly(route):
    drone.reference = make_reference(route)
    total = WAIT_TIME + CLIMB_TIME + route.duration + HOVER_TIME
    vehicle = drone.Vehicle()
    vehicle.pos = np.array([world.START[0], world.START[1], 0.0])
    controller = drone.Controller()
    suite = sensors.Sensors(seed=1)
    nav = navigation.NavigationFilter(suite.gps(vehicle))
    rows = []
    fixes = []
    steps = int(round(total / drone.DT))
    for step in range(steps):
        t = step * drone.DT
        if step > 0 and step % sensors.GPS_EVERY == 0:
            fix = suite.gps(vehicle)
            fixes.append(np.concatenate([[t], fix]))
            nav.correct_gps(fix)
        if step > 0 and step % sensors.COMPASS_EVERY == 0:
            nav.correct_compass(suite.compass(vehicle))
        if t < WAIT_TIME:
            thrust, torque = 0.0, np.zeros(3)
        else:
            thrust, torque = controller.command(t, nav.pos, nav.vel, nav.angles, nav.rates)
        if step % 10 == 0:
            rows.append(np.concatenate([[t], vehicle.pos, drone.reference(t)[0], nav.pos, vehicle.angles]))
        vehicle.step(thrust, torque)
        nav.predict(suite.accelerometer(vehicle), suite.gyroscope(vehicle))
    return np.array(rows), np.array(fixes)


def main():
    route, count, added = plan_route()
    gap, when = route.clearance()
    print("plan: %d waypoints (+%d added), duration %.1f s, closest approach of the plan to an obstacle %.2f m" % (count, added, route.duration, gap))
    log, fixes = fly(route)
    t = log[:, 0]
    true, ref, est = log[:, 1:4], log[:, 4:7], log[:, 7:10]
    airborne = t >= WAIT_TIME + CLIMB_TIME
    tracking = np.linalg.norm(true - ref, axis=1)[airborne]
    estimation = np.linalg.norm(est - true, axis=1)[airborne]
    raw = np.linalg.norm(fixes[:, 1:4] - np.array([true[np.searchsorted(t, f[0]) - 1] for f in fixes]), axis=1)
    gaps = np.array([world.distance_to_obstacles(p) for p in true[airborne]])
    miss = float(np.linalg.norm(true[-1] - world.GOAL))
    print("flight on sensors only:")
    print("  raw GPS error         rms %.2f m" % np.sqrt((raw ** 2).mean()))
    print("  position estimate     rms %.2f m, max %.2f m" % (np.sqrt((estimation ** 2).mean()), estimation.max()))
    print("  tracking error        rms %.2f m, max %.2f m" % (np.sqrt((tracking ** 2).mean()), tracking.max()))
    print("  closest approach of the vehicle to an obstacle: %.2f m (vehicle radius %.1f m)" % (gaps.min(), VEHICLE_RADIUS))
    print("  distance from the goal at the end: %.2f m" % miss)
    print("MISSION", "SUCCEEDED" if gaps.min() > VEHICLE_RADIUS and miss < 0.5 else "FAILED")

    os.makedirs("results", exist_ok=True)
    np.savez("results/mission_log.npz", log=log, fixes=fixes)
    fig = plt.figure(figsize=(10.0, 4.4))
    space = fig.add_axes([0.0, 0.02, 0.56, 0.96], projection="3d")
    for x0, y0, z0, x1, y1, z1 in world.BOXES:
        space.bar3d(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0, color="0.6", alpha=0.3, edgecolor="0.35", linewidth=0.4, shade=False)
    space.plot(ref[:, 0], ref[:, 1], ref[:, 2], color="0.3", linewidth=1, linestyle=(0, (4, 3)), label="planned")
    space.plot(true[:, 0], true[:, 1], true[:, 2], color="#2a78d6", linewidth=2, label="flown")
    space.set_xlim(0, 20)
    space.set_ylim(0, 20)
    space.set_zlim(0, 6)
    space.set_box_aspect((20, 20, 6), zoom=0.95)
    space.set_xticks([0, 5, 10, 15, 20])
    space.set_yticks([0, 5, 10, 15, 20])
    space.set_zticks([0, 3, 6])
    space.view_init(elev=38, azim=-58)
    space.set_xlabel("x (m)")
    space.set_ylabel("y (m)")
    space.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.05, 0.98))
    top = fig.add_axes([0.62, 0.12, 0.35, 0.78])
    for x0, y0, z0, x1, y1, z1 in world.BOXES:
        top.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="0.6" if z1 > 3 else "0.82", edgecolor="0.35", linewidth=0.5))
    top.plot(fixes[:, 1], fixes[:, 2], linestyle="none", marker="o", markersize=2, color="#eb6834", alpha=0.6, label="GPS readings")
    top.plot(ref[:, 0], ref[:, 1], color="0.3", linewidth=1, linestyle=(0, (4, 3)), label="planned")
    top.plot(true[:, 0], true[:, 1], color="#2a78d6", linewidth=2, label="flown")
    top.set_xlim(0, 20)
    top.set_ylim(0, 20)
    top.set_aspect("equal")
    top.set_xlabel("x (m)")
    top.set_ylabel("y (m)")
    top.set_title("Top view", fontsize=10, loc="left")
    top.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=9, loc="upper left")
    fig.savefig("results/s8_mission.png", dpi=200)
    print("figure written to results/s8_mission.png, log written to results/mission_log.npz")


main()