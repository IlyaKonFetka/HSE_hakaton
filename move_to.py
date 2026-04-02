"""Move robot end-effector to a target XYZ position (meters, robot frame)."""
import sys
import numpy as np
sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")

from so101_kinematics import inverse_kinematics_deg, forward_kinematics_deg, JOINT_LIMITS_RAD
from robot_controller import SO101Controller

TARGET = np.array([0.00, 0.00, 0.42])   # straight up

ctrl = SO101Controller(
    port="COM7",
    calibration_dir=r"C:\Users\simal\.cache\huggingface\lerobot\calibration\robots\so_follower",
)
ctrl.connect()

# Current joints as initial guess for IK
current = ctrl.get_state()
print(f"Current joints: {current[:5]}")
pos, _ = forward_kinematics_deg(current[:5])
print(f"Current TCP:    {pos}")

# Solve IK with multi-start
print(f"\nSolving IK for target {TARGET} ...")
best_joints, best_err = None, np.inf

for i, guess_deg in enumerate([current[:5]] + [
    np.rad2deg([np.random.uniform(lo, hi) for lo, hi in JOINT_LIMITS_RAD])
    for _ in range(50)
]):
    j, _, e = inverse_kinematics_deg(TARGET, initial_guess_deg=guess_deg)
    if e < best_err:
        best_joints, best_err = j, e
    if best_err < 0.002:
        break

joints_deg, err = best_joints, best_err
print(f"IK joints: {np.round(joints_deg, 2)}")
print(f"IK error:  {err*1000:.2f} mm")
if err > 0.010:
    print("WARNING: IK error > 10mm — target may be unreachable")

# Verify with FK
achieved, _ = forward_kinematics_deg(joints_deg)
print(f"FK check:  {achieved}  (error {np.linalg.norm(achieved-TARGET)*1000:.1f}mm)")

ans = input("\nMove? [y/n]: ").strip().lower()
if ans in ('y', 'н', 'yes', 'да'):
    ctrl.move_smooth(joints_deg, duration=2.0)
    final = ctrl.get_state()
    pos2, _ = forward_kinematics_deg(final[:5])
    print(f"Final TCP: {pos2}")

ctrl.disconnect()
