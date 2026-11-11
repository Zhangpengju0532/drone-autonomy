import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import drone
import navigation
import sensors


def fly(use_estimates):
    vehicle = drone.Vehicle()
    controller = drone.Controller()
    suite = sensors.Sensors(seed=1)
    nav = navigation.NavigationFilter(suite.gps(vehicle))
    log = []
    steps = int(round(drone.flight_time() / drone.DT))
    for step in range(steps):
        t = step * drone.DT
        if step > 0 and step % sensors.GPS_EVERY == 0:
            nav.correct_gps(suite.gps(vehicle))
        if step > 0 and step % sensors.COMPASS_EVERY == 0:
            nav.correct_compass(suite.compass(vehicle))
        if use_estimates:
            thrust, torque = controller.command(t, nav.pos, nav.vel, nav.angles, nav.rates)
        else:
            thrust, torque = controller.command(t, vehicle.pos, vehicle.vel, vehicle.angles, vehicle.rates)
        if step % 10 == 0:
            ref = drone.reference(t)[0]
            log.append((t, vehicle.pos.copy(), ref, nav.pos.copy(), vehicle.angles.copy(), nav.angles.copy(),
                        nav.accel_bias.copy(), nav.gyro_bias.copy()))
        vehicle.step(thrust, torque)
        nav.predict(suite.accelerometer(vehicle), suite.gyroscope(vehicle))
    return log


def rms(values):
    return float(np.sqrt((np.asarray(values) ** 2).mean()))


def main():
    start = drone.TAKEOFF_TIME + 2.0
    print("controller uses    tracking rms (m)  tracking max (m)")
    logs = {}
    for name, flag in (("true state", False), ("sensors only", True)):
        log = fly(flag)
        late = [row for row in log if row[0] >= start]
        tracking = [np.linalg.norm(row[1] - row[2]) for row in late]
        print("%-17s  %16.3f  %16.3f" % (name, rms(tracking), max(tracking)))
        logs[name] = log
    log = logs["sensors only"]
    late = [row for row in log if row[0] >= start]
    position = [np.linalg.norm(row[3] - row[1]) for row in late]
    tilt = [np.degrees(row[5][:2] - row[4][:2]) for row in late]
    yaw = [np.degrees(row[5][2] - row[4][2]) for row in late]
    print("estimation errors when flying on sensors only:")
    print("  position rms %.3f m | roll and pitch rms %.2f deg | yaw rms %.2f deg" % (rms(position), rms(np.array(tilt).ravel()), rms(yaw)))
    print("accelerometer bias  true:", sensors.ACCEL_BIAS, " estimated:", np.round(log[-1][6], 3))
    print("gyroscope bias      true:", sensors.GYRO_BIAS, " estimated:", np.round(log[-1][7], 4))

    os.makedirs("results", exist_ok=True)
    times = [row[0] for row in log]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    reference = np.array([row[2] for row in log])
    flown = np.array([row[1] for row in log])
    axes[0].plot(reference[:, 0], reference[:, 1], color="0.55", linewidth=1, linestyle=(0, (4, 3)), label="reference")
    axes[0].plot(flown[:, 0], flown[:, 1], color="#2a78d6", linewidth=1.5, label="flown")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_ylim(-2.4, 3.0)
    axes[0].set_title("Flight on sensors only, top view", fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    axes[1].plot(times, np.degrees([row[4][1] for row in log]), color="0.1", linewidth=1.5, label="true pitch")
    axes[1].plot(times, np.degrees([row[5][1] for row in log]), color="#2a78d6", linewidth=1.2, label="estimated pitch")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("pitch (deg)")
    axes[1].set_title("Attitude estimate", fontsize=10, loc="left")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    for k, label in enumerate(("x", "y", "z")):
        axes[2].plot(times, [row[7][k] for row in log], linewidth=1.5, color=("#2a78d6", "#eb6834", "#1baf6c")[k], label="estimated " + label)
        axes[2].axhline(sensors.GYRO_BIAS[k], color="0.45", linewidth=1, linestyle=(0, (4, 3)))
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("gyroscope bias (rad/s)")
    axes[2].set_title("Gyroscope bias (dashed: true)", fontsize=10, loc="left")
    axes[2].legend(frameon=False, fontsize=9, loc="lower right")
    for ax in axes:
        ax.grid(color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/s5_fly_blind.png", dpi=200)
    print("figure written to results/s5_fly_blind.png")


main()