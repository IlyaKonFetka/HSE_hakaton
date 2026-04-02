

import time
import numpy as np


class SO101Controller:


    def __init__(self, port="COM7", calibration_dir=None):
        self.port = port
        self.calibration_dir = calibration_dir
        self.robot = None

    def connect(self):

        from pathlib import Path
        from lerobot.robots.so_follower.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        config = SOFollowerRobotConfig(
            port=self.port,
            id="my_follower",
            disable_torque_on_disconnect=False,
            cameras={},
            calibration_dir=Path(self.calibration_dir) if self.calibration_dir else None,
        )

        self.robot = SOFollower(config)
        self.robot.connect()
        print(f"Robot connected on {self.port}")

    def disconnect(self):

        if self.robot:
            self.robot.disconnect()
            print("Robot disconnected")

    def get_state(self):

        obs = None
        for attempt in range(5):
            try:
                obs = self.robot.get_observation()
                break
            except Exception:
                time.sleep(0.05)
        if obs is None:
            raise ConnectionError("Failed to read robot state after 5 retries")
        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        state = np.array([obs[f"{m}.pos"] for m in motor_names], dtype=float)
        return state

    def send_joints(self, angles_deg, gripper=None):

        angles = np.asarray(angles_deg, dtype=np.float32)
        if gripper is None:
            current = self.get_state()
            gripper = float(current[5])

        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        action = {f"{m}.pos": float(angles[i]) for i, m in enumerate(motor_names)}
        action["gripper.pos"] = float(gripper)
        self.robot.send_action(action)

    def move_smooth(self, target_deg, gripper=None, duration=1.5, steps=30):

        current = self.get_state()
        current_joints = current[:5]
        current_gripper = current[5]

        target = np.asarray(target_deg, dtype=np.float32)
        target_grip = gripper if gripper is not None else current_gripper

        dt = duration / steps

        for i in range(1, steps + 1):
            alpha = i / steps

            alpha = 0.5 * (1 - np.cos(np.pi * alpha))

            interp_joints = current_joints + alpha * (target - current_joints)
            interp_grip = current_gripper + alpha * (target_grip - current_gripper)

            self.send_joints(interp_joints, interp_grip)
            time.sleep(dt)

        time.sleep(0.15)

    def open_gripper(self, value=80.0, duration=0.5):

        current = self.get_state()
        self.move_smooth(current[:5], gripper=value, duration=duration, steps=15)

    def close_gripper(self, value=10.0, duration=0.5):

        current = self.get_state()
        self.move_smooth(current[:5], gripper=value, duration=duration, steps=15)


if __name__ == "__main__":
    ctrl = SO101Controller(
        port="COM7",
        calibration_dir=r"C:\Users\simal\.cache\huggingface\lerobot\calibration\robots\so_follower",
    )
    ctrl.connect()

    print("Current state:", ctrl.get_state())
    print("\nMoving to home position (all zeros)...")
    ctrl.move_smooth([0, 0, 0, 0, 0], gripper=50, duration=2.0)
    print("Done. Current state:", ctrl.get_state())

    ctrl.disconnect()
