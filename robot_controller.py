"""
Direct motor control for SO-101 follower arm via feetech SDK.

Sends joint angle commands (in degrees) to the real robot.
Uses lerobot's bus interface under the hood.
"""

import time
import numpy as np


class SO101Controller:
    """
    High-level controller for SO-101 follower arm.
    Wraps lerobot's SOFollower for simple joint position control.
    """

    def __init__(self, port="COM7", calibration_dir=None):
        self.port = port
        self.calibration_dir = calibration_dir
        self.robot = None

    def connect(self):
        """Connect to the robot."""
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
        """Disconnect from the robot."""
        if self.robot:
            self.robot.disconnect()
            print("Robot disconnected")

    def get_state(self):
        """
        Read current joint positions.

        Returns
        -------
        state : ndarray, shape (6,)
            [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
            First 5 in degrees, gripper in 0..100 range.
        """
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
        """
        Send joint angles to the robot.

        Parameters
        ----------
        angles_deg : array-like, shape (5,)
            Target angles in degrees for the 5 joints.
        gripper : float or None
            Gripper position 0 (closed) to 100 (open). If None, keeps current.
        """
        angles = np.asarray(angles_deg, dtype=np.float32)
        if gripper is None:
            current = self.get_state()
            gripper = float(current[5])

        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        action = {f"{m}.pos": float(angles[i]) for i, m in enumerate(motor_names)}
        action["gripper.pos"] = float(gripper)
        self.robot.send_action(action)

    def move_smooth(self, target_deg, gripper=None, duration=1.5, steps=30):
        """
        Smoothly interpolate from current position to target.

        Parameters
        ----------
        target_deg : array-like, shape (5,)
            Target joint angles in degrees.
        gripper : float or None
            Target gripper value (0..100).
        duration : float
            Time for the movement in seconds.
        steps : int
            Number of interpolation steps.
        """
        current = self.get_state()
        current_joints = current[:5]
        current_gripper = current[5]

        target = np.asarray(target_deg, dtype=np.float32)
        target_grip = gripper if gripper is not None else current_gripper

        dt = duration / steps

        for i in range(1, steps + 1):
            alpha = i / steps
            # Smooth ease-in-out
            alpha = 0.5 * (1 - np.cos(np.pi * alpha))

            interp_joints = current_joints + alpha * (target - current_joints)
            interp_grip = current_gripper + alpha * (target_grip - current_gripper)

            self.send_joints(interp_joints, interp_grip)
            time.sleep(dt)

        time.sleep(0.15)  # let bus settle after burst of writes

    def open_gripper(self, value=80.0, duration=0.5):
        """Open the gripper."""
        current = self.get_state()
        self.move_smooth(current[:5], gripper=value, duration=duration, steps=15)

    def close_gripper(self, value=10.0, duration=0.5):
        """Close the gripper."""
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
