import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import drone
import estimator
import sensors

COLORS = {"true state": "0.1", "raw GPS": "#eb6834", "Kalman filter": "#2a78d6"}


def fly(feedback):
    vehicle = drone.Vehicle()
    controller = drone.Controller()
    suite = sensors.Sensors(seed=1)
    gps = suite.gps(vehicle)
    previous_gps = gps.copy()
    gps_speed = np.zeros(3)
    kalman = estimator.PositionFilter(gps)
    log = []
    steps = int(round(drone.flight_time() / drone.DT))
    for step in range(steps):
        t = step * drone.DT
        if step % sensors.GPS_EVERY == 0 and step > 0:
            previous_gps = gps
            gps = suite.gps(vehicle)
            gps_speed = (gps - previous_gps) / (sensors.GPS_EVERY * drone.DT)
            kalman.correct(gps)
        if feedback == "true state":
            pos, vel = vehicle.pos, vehicle.vel
        elif feedback == "raw GPS":
            pos, vel = gps, gps_speed
        else:
            pos, vel = kalman.pos, kalman.vel
        thrust, torque = controller.command(t, pos, vel, vehicle.angles, vehicle.rates)
        if step % 10 == 0:
            ref = drone.reference(t)[0]
            log.append((t, vehicle.pos.copy(), ref, kalman.pos, np.linalg.norm(vehicle.pos - ref),
                        np.linalg.norm(kalman.pos - vehicle.pos), max(abs(vehicle.angles[0]), abs(vehicle.angles[1]))))
        vehicle.step(thrust, torque)
        kalman.predict(suite.accelerometer(vehicle), vehicle.angles)
    return log, kalman


def main():
    runs = {}
    print("feedback        tracking rms (m)  tracking max (m)  largest tilt (deg)")
    for name in COLORS:
        log, kalman = fly(name)
        late = [row for row in log if row[0] >= drone.TAKEOFF_TIME + 2.0]
        tracking = np.array([row[4] for row in late])
        tilt = np.degrees(max(row[6] for row in log))
        print("%-14s  %16.3f  %16.3f  %18.1f" % (name, np.sqrt((tracking ** 2).mean()), tracking.max(), tilt))
        runs[name] = (log, kalman)
    log, kalman = runs["Kalman filter"]
    late = [row for row in log if row[0] >= drone.TAKEOFF_TIME + 2.0]
    estimation = np.array([row[5] for row in late])
    print("Kalman filter estimation error: rms %.3f m, max %.3f m (single GPS reading: about %.2f m)" % (
        np.sqrt((estimation ** 2).mean()), estimation.max(), np.linalg.norm(sensors.GPS_NOISE)))
    print("accelerometer bias  true:", sensors.ACCEL_BIAS, " estimated:", np.round(kalman.bias, 3))

    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), sharex=True, sharey=True)
    reference = np.array([row[2] for row in log])
    for ax, name in zip(axes, COLORS):
        flown = np.array([row[1] for row in runs[name][0]])
        ax.plot(reference[:, 0], reference[:, 1], color="0.55", linewidth=1, linestyle=(0, (4, 3)))
        ax.plot(flown[:, 0], flown[:, 1], color=COLORS[name], linewidth=1.5)
        ax.set_aspect("equal")
        ax.set_title("controller fed with: " + name, fontsize=10, loc="left")
        ax.set_xlabel("x (m)")
        ax.grid(color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("y (m)")
    axes[0].set_xlim(-4.5, 4.5)
    axes[0].set_ylim(-3.2, 3.2)
    fig.tight_layout()
    fig.savefig("results/s4_fly_on_estimates.png", dpi=200)
    print("figure written to results/s4_fly_on_estimates.png")


main()