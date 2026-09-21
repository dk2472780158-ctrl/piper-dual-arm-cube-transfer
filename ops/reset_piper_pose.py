#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import sys
import time

import numpy as np

from lerobot.robots.piper_dual.config_piper_dual import PIPERDualConfig
from lerobot.robots.piper_dual.piper_dual import PIPERDual


TRAINING_START_POSE = np.asarray(
    [
        0.0492107794,
        0.0,
        0.0,
        -0.2287141085,
        0.2685223222,
        0.2780382633,
        0.0,
        -0.0443961099,
        0.0,
        0.0,
        0.0536503866,
        0.3397218585,
        -0.0396250561,
        0.0,
    ],
    dtype=np.float64,
)

SINGLE_ARM_LOWER = np.asarray([-1.61, -0.05, -1.93, -1.58, -1.40, -1.58, 0.0])
SINGLE_ARM_UPPER = np.asarray([1.61, 2.10, 0.06, 1.58, 1.40, 1.58, 0.08])
LOWER_LIMITS = np.concatenate((SINGLE_ARM_LOWER, SINGLE_ARM_LOWER))
UPPER_LIMITS = np.concatenate((SINGLE_ARM_UPPER, SINGLE_ARM_UPPER))
# The nominal limits validate the final target. A fallen arm can legitimately
# report a pose outside that command envelope, so recovery starts are checked
# only against a broad single-turn sanity envelope. Generated commands still
# move component-wise from the measured start toward the nominal target and can
# never move farther outward.
SINGLE_ARM_RECOVERY_LOWER = np.asarray([-3.20, -3.20, -3.20, -3.20, -3.20, -3.20, -0.01])
SINGLE_ARM_RECOVERY_UPPER = np.asarray([3.20, 3.20, 3.20, 3.20, 3.20, 3.20, 0.10])
RECOVERY_LOWER_LIMITS = np.concatenate((SINGLE_ARM_RECOVERY_LOWER, SINGLE_ARM_RECOVERY_LOWER))
RECOVERY_UPPER_LIMITS = np.concatenate((SINGLE_ARM_RECOVERY_UPPER, SINGLE_ARM_RECOVERY_UPPER))
JOINT_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER_INDICES = np.asarray([6, 13])
POSITION_KEYS = [
    "joint_1_pos",
    "joint_2_pos",
    "joint_3_pos",
    "joint_4_pos",
    "joint_5_pos",
    "joint_6_pos",
    "gripper_pos",
]

stop_requested = False


def request_stop(_signum=None, _frame=None) -> None:
    global stop_requested
    stop_requested = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slowly move dual Piper followers to the demonstrated start pose without disabling."
    )
    parser.add_argument("--left-can", default="can1")
    parser.add_argument("--right-can", default="can0")
    parser.add_argument(
        "--arm",
        required=True,
        choices=("left", "right", "both"),
        help="Move the left arm, right arm, or both arms toward the target.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--joint-speed", type=float, default=0.20, help="Maximum joint speed in rad/s.")
    parser.add_argument("--gripper-speed", type=float, default=0.006, help="Maximum gripper speed in m/s.")
    parser.add_argument("--tracking-limit", type=float, default=0.12, help="Maximum joint tracking error in rad.")
    parser.add_argument(
        "--gripper-tracking-limit", type=float, default=0.015, help="Maximum gripper tracking error in m."
    )
    parser.add_argument("--settle-tolerance", type=float, default=0.025, help="Final joint tolerance in rad.")
    parser.add_argument("--settle-timeout", type=float, default=10.0)
    parser.add_argument("--feedback-warmup", type=int, default=30)
    parser.add_argument(
        "--target",
        type=float,
        nargs=14,
        default=None,
        metavar=("LJ1", "LJ2", "LJ3", "LJ4", "LJ5", "LJ6", "LG", "RJ1", "RJ2", "RJ3", "RJ4", "RJ5", "RJ6", "RG"),
        help="Optional explicit 14-value target. Default is the training median start pose.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually move the robot; default is dry-run.")
    return parser.parse_args()


def read_bus_positions(bus) -> np.ndarray:
    state = bus.read()
    missing = [key for key in POSITION_KEYS if key not in state]
    if missing:
        raise RuntimeError(f"Missing Piper feedback keys: {missing}")
    values = np.asarray([state[key] for key in POSITION_KEYS], dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid Piper feedback: {values}")
    return values


def read_positions(robot: PIPERDual) -> np.ndarray:
    return np.concatenate((read_bus_positions(robot.left_bus), read_bus_positions(robot.right_bus)))


def read_stable_positions(robot: PIPERDual, reads: int) -> np.ndarray:
    positions = None
    for _ in range(max(1, reads)):
        positions = read_positions(robot)
        time.sleep(0.05)
    if positions is None:
        raise RuntimeError("No Piper feedback received.")
    return positions


def validate_pose(pose: np.ndarray, name: str, allow_recovery: bool = False) -> None:
    if pose.shape != (14,) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must contain 14 finite values, got {pose}")
    lower = RECOVERY_LOWER_LIMITS if allow_recovery else LOWER_LIMITS
    upper = RECOVERY_UPPER_LIMITS if allow_recovery else UPPER_LIMITS
    bad = np.flatnonzero((pose < lower) | (pose > upper))
    if bad.size:
        details = [
            f"index={i}, value={pose[i]:.6f}, range=[{lower[i]:.6f}, {upper[i]:.6f}]"
            for i in bad
        ]
        raise ValueError(f"{name} violates configured limits: " + "; ".join(details))


def print_nominal_limit_warning(pose: np.ndarray) -> None:
    outside = np.flatnonzero((pose < LOWER_LIMITS) | (pose > UPPER_LIMITS))
    if not outside.size:
        return
    print("WARNING: current feedback is outside the nominal command range:")
    for index in outside:
        print(
            f"  index={index}, current={pose[index]:.6f}, "
            f"nominal=[{LOWER_LIMITS[index]:.6f}, {UPPER_LIMITS[index]:.6f}]"
        )
    print("Recovery will move these joints only toward the valid target, never farther outward.")


def arm_is_enabled(bus) -> bool:
    info = bus.piper.GetArmLowSpdInfoMsgs()
    motors = (info.motor_1, info.motor_2, info.motor_3, info.motor_4, info.motor_5, info.motor_6)
    return all(bool(motor.foc_status.driver_enable_status) for motor in motors)


def enable_arm_holding(bus, hold: np.ndarray, name: str, timeout: float = 8.0) -> None:
    bus.write(hold.tolist())
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if arm_is_enabled(bus):
            bus.write(hold.tolist())
            print(f"{name} arm enabled and holding its measured pose.")
            return
        sent = bool(bus.piper.EnablePiper())
        if sent:
            bus.write(hold.tolist())
        time.sleep(0.15)
    raise RuntimeError(f"Timed out enabling {name} arm. No disable command was sent.")


def send_pose(robot: PIPERDual, pose: np.ndarray) -> None:
    validate_pose(pose, "recovery command", allow_recovery=True)
    robot.left_bus.write(pose[:7].tolist())
    robot.right_bus.write(pose[7:].tolist())


def hold_measured_pose(robot: PIPERDual) -> None:
    print("Holding measured pose; both arms will remain ENABLED...")
    try:
        pose = read_stable_positions(robot, 3)
        pose[GRIPPER_INDICES] = np.clip(pose[GRIPPER_INDICES], 0.0, 0.08)
        send_pose(robot, pose)
        time.sleep(0.3)
        print("Current pose is held. No disable command was sent.")
    except Exception as error:
        print(f"WARNING: could not refresh hold target: {error}", file=sys.stderr)
        print("The last valid position command remains active; no disable command was sent.", file=sys.stderr)


def move_slowly(robot: PIPERDual, start: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> None:
    period = 1.0 / args.fps
    joint_step = args.joint_speed / args.fps
    gripper_step = args.gripper_speed / args.fps
    command = start.copy()
    corridor_lower = np.minimum(start, target) - 1e-9
    corridor_upper = np.maximum(start, target) + 1e-9
    step = 0

    while not stop_requested:
        measured = read_positions(robot)
        joint_tracking = float(np.max(np.abs(measured[JOINT_INDICES] - command[JOINT_INDICES])))
        gripper_tracking = float(np.max(np.abs(measured[GRIPPER_INDICES] - command[GRIPPER_INDICES])))
        if joint_tracking > args.tracking_limit:
            raise RuntimeError(
                f"Safety stop: joint tracking error {joint_tracking:.6f} rad exceeds {args.tracking_limit:.6f} rad."
            )
        if gripper_tracking > args.gripper_tracking_limit:
            raise RuntimeError(
                f"Safety stop: gripper tracking error {gripper_tracking:.6f} m exceeds "
                f"{args.gripper_tracking_limit:.6f} m."
            )

        delta = target - command
        next_command = command.copy()
        next_command[JOINT_INDICES] += np.clip(delta[JOINT_INDICES], -joint_step, joint_step)
        next_command[GRIPPER_INDICES] += np.clip(delta[GRIPPER_INDICES], -gripper_step, gripper_step)
        if np.any(next_command < corridor_lower) or np.any(next_command > corridor_upper):
            raise RuntimeError("Internal safety error: recovery command left the start-to-target corridor.")
        send_pose(robot, next_command)
        command = next_command

        if step % max(1, int(args.fps)) == 0:
            remaining = float(np.max(np.abs(target[JOINT_INDICES] - command[JOINT_INDICES])))
            print(
                f"step={step:05d} max_remaining={remaining:.5f} rad "
                f"tracking={joint_tracking:.5f} rad"
            )

        joints_done = np.all(np.abs(target[JOINT_INDICES] - command[JOINT_INDICES]) <= 1e-8)
        grippers_done = np.all(np.abs(target[GRIPPER_INDICES] - command[GRIPPER_INDICES]) <= 1e-8)
        if joints_done and grippers_done:
            break
        step += 1
        time.sleep(period)

    if stop_requested:
        print("Stop requested during movement.")
        return

    print("Target command reached; waiting for the arms to settle...")
    deadline = time.monotonic() + args.settle_timeout
    while time.monotonic() < deadline and not stop_requested:
        measured = read_positions(robot)
        joint_error = float(np.max(np.abs(measured[JOINT_INDICES] - target[JOINT_INDICES])))
        gripper_error = float(np.max(np.abs(measured[GRIPPER_INDICES] - target[GRIPPER_INDICES])))
        send_pose(robot, target)
        if joint_error <= args.settle_tolerance and gripper_error <= args.gripper_tracking_limit:
            print(f"Pose reset complete. Maximum joint error: {joint_error:.6f} rad")
            return
        time.sleep(period)
    if not stop_requested:
        raise RuntimeError("Target did not settle before timeout.")


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.joint_speed <= 0 or args.gripper_speed <= 0:
        raise ValueError("--fps, --joint-speed and --gripper-speed must be positive.")

    requested_target = (
        TRAINING_START_POSE.copy() if args.target is None else np.asarray(args.target, dtype=np.float64)
    )
    validate_pose(requested_target, "target")
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    robot = None
    enabled = False
    try:
        robot = PIPERDual(
            PIPERDualConfig(
                left_port=args.left_can,
                right_port=args.right_can,
                read_only=not args.execute,
                cameras={},
            )
        )
        # Never call robot.connect() or robot.calibrate(): this fork commands all-zero pose there.
        robot._is_connected = True
        robot._is_calibrated = True
        print("Warming up Piper feedback...")
        current = read_stable_positions(robot, args.feedback_warmup)
        validate_pose(current, "current feedback", allow_recovery=True)
        target = requested_target.copy()
        if args.arm == "left":
            target[7:] = current[7:]
        elif args.arm == "right":
            target[:7] = current[:7]
        validate_pose(target, "recovery target", allow_recovery=True)
        if args.arm == "both":
            print("Selected arms: both. Both arms will move toward the target simultaneously.")
        else:
            print(f"Selected arm: {args.arm}. The other arm will hold its measured pose.")
        print("Current:", np.array2string(current, precision=6, separator=", "))
        print("Target: ", np.array2string(target, precision=6, separator=", "))
        print("Delta:  ", np.array2string(target - current, precision=6, separator=", "))
        print_nominal_limit_warning(current)

        if not args.execute:
            print("DRY-RUN only. Re-run with --execute to move slowly to this target.")
            return

        print("Preloading the measured pose before enabling...")
        send_pose(robot, current)
        time.sleep(0.2)
        enable_arm_holding(robot.left_bus, current[:7], "Left")
        enabled = True
        enable_arm_holding(robot.right_bus, current[7:], "Right")
        send_pose(robot, current)
        time.sleep(0.3)
        print("Moving slowly. Press Ctrl+C to stop and hold the measured pose.")
        move_slowly(robot, current, target, args)
    except KeyboardInterrupt:
        print("Keyboard interrupt received.")
    except Exception as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr)
        if robot is None:
            raise
    finally:
        if robot is not None and enabled:
            hold_measured_pose(robot)
            print("Both follower arms remain ENABLED. Physically support them before any later power-off.")


if __name__ == "__main__":
    main()
