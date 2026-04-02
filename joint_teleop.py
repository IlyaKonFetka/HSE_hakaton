"""
Interactive joint angle control.
Use +/- keys to increment/decrement individual joint angles.
Press 'p' to print current TCP position.
"""

import sys
import numpy as np

sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")
from so101_kinematics import forward_kinematics_deg

ROBOT_PORT   = "COM7"
CALIB_DIR    = r"C:\Users\simal\.cache\huggingface\lerobot\calibration\robots\so_follower"
STEP         = 5.0   # degrees per keypress

MOTOR_NAMES  = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINT_LABELS = ["1:shoulder_pan", "2:shoulder_lift", "3:elbow_flex", "4:wrist_flex", "5:wrist_roll", "6:gripper"]


def connect():
    from pathlib import Path
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    cfg = SOFollowerRobotConfig(
        port=ROBOT_PORT, id="my_follower",
        disable_torque_on_disconnect=True, cameras={},
        calibration_dir=Path(CALIB_DIR),
    )
    robot = SOFollower(cfg)
    robot.connect()
    return robot


def get_state(robot):
    obs = robot.get_observation()
    joints = np.array([obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=float)
    gripper = float(obs["gripper.pos"])
    return joints, gripper


def send(robot, joints, gripper):
    action = {f"{m}.pos": float(joints[i]) for i, m in enumerate(MOTOR_NAMES)}
    action["gripper.pos"] = float(gripper)
    robot.send_action(action)


def print_state(joints, gripper):
    pos, _ = forward_kinematics_deg(joints)
    print(f"\n  Joints (deg): pan={joints[0]:.1f}  lift={joints[1]:.1f}  "
          f"elbow={joints[2]:.1f}  wrist_flex={joints[3]:.1f}  wrist_roll={joints[4]:.1f}  "
          f"gripper={gripper:.1f}")
    print(f"  TCP (m):      x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}")


def main():
    print("=== Joint Teleop ===")
    print(f"Step size: {STEP} deg  (change with 's <value>')\n")
    print("Controls:")
    print("  1-5      — select joint (shoulder_pan .. wrist_roll)")
    print("  6        — select gripper")
    print("  +  or =  — increase selected joint")
    print("  -        — decrease selected joint")
    print("  p        — print current position")
    print("  h        — move all joints to 0")
    print("  s <val>  — set step size (e.g. 's 2')")
    print("  q        — quit\n")

    robot = connect()
    joints, gripper = get_state(robot)
    selected = 0  # index 0..5
    step = STEP

    print_state(joints, gripper)
    print(f"\n  Selected: {JOINT_LABELS[selected]}")

    while True:
        try:
            cmd = input("> ").strip()
        except EOFError:
            break
        if not cmd:
            continue

        if cmd == 'q':
            break
        elif cmd == 'p':
            joints, gripper = get_state(robot)
            print_state(joints, gripper)
        elif cmd == 'h':
            joints = np.zeros(5)
            gripper = 50.0
            send(robot, joints, gripper)
            print("  → home (all zeros, gripper=50)")
        elif cmd in ('1','2','3','4','5','6'):
            selected = int(cmd) - 1
            print(f"  Selected: {JOINT_LABELS[selected]}")
        elif cmd in ('+', '='):
            if selected < 5:
                joints[selected] += step
                send(robot, joints, gripper)
                print_state(joints, gripper)
            else:
                gripper = min(100.0, gripper + step)
                send(robot, joints, gripper)
                print(f"  gripper → {gripper:.1f}")
        elif cmd == '-':
            if selected < 5:
                joints[selected] -= step
                send(robot, joints, gripper)
                print_state(joints, gripper)
            else:
                gripper = max(0.0, gripper - step)
                send(robot, joints, gripper)
                print(f"  gripper → {gripper:.1f}")
        elif cmd.startswith('s '):
            try:
                step = float(cmd[2:])
                print(f"  Step = {step} deg")
            except ValueError:
                print("  Usage: s <value>  e.g. s 2")
        else:
            print("  Unknown command")

    robot.disconnect()


if __name__ == "__main__":
    main()
