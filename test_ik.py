"""
Test FK/IK by moving +5cm along each axis from current position.
If robot moves in the right direction -> model is correct.
"""
import sys
import numpy as np
sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")

from so101_kinematics import inverse_kinematics_deg, forward_kinematics_deg, JOINT_LIMITS_RAD
from robot_controller import SO101Controller

ctrl = SO101Controller(
    port="COM7",
    calibration_dir=r"C:\Users\simal\.cache\huggingface\lerobot\calibration\robots\so_follower",
)
ctrl.connect()

current = ctrl.get_state()
current_pos, _ = forward_kinematics_deg(current[:5])
print(f"Current joints (deg): {np.round(current[:5], 1)}")
print(f"Current TCP (FK):     {np.round(current_pos*100, 1)} cm\n")

MOVES = {
    "+Z (up 5cm)":    current_pos + [0, 0, 0.05],
    "-Z (down 5cm)":  current_pos + [0, 0, -0.05],
    "+X (fwd 5cm)":   current_pos + [0.05, 0, 0],
    "-X (back 5cm)":  current_pos + [-0.05, 0, 0],
    "+Y (left 5cm)":  current_pos + [0, 0.05, 0],
    "-Y (right 5cm)": current_pos + [0, -0.05, 0],
    "HOME (return)":  current_pos,
}

for label, target in MOVES.items():
    print(f"--- {label} → target {np.round(target*100,1)} cm ---")
    best_j, best_e = None, np.inf
    for guess in [current[:5]] + [
        np.rad2deg([np.random.uniform(lo, hi) for lo, hi in JOINT_LIMITS_RAD])
        for _ in range(30)
    ]:
        j, _, e = inverse_kinematics_deg(target, initial_guess_deg=guess)
        if e < best_e:
            best_j, best_e = j, e
        if best_e < 0.002:
            break

    print(f"    IK error: {best_e*1000:.1f} mm")
    ans = input("    Move? [y/n/q]: ").strip().lower()
    if ans == 'q':
        break
    if ans in ('y', 'н', 'yes', 'да'):
        ctrl.move_smooth(best_j, duration=1.5)
        actual = ctrl.get_state()
        actual_pos, _ = forward_kinematics_deg(actual[:5])
        print(f"    Reached (FK): {np.round(actual_pos*100, 1)} cm")
        print(f"    Delta from start: {np.round((actual_pos - current_pos)*100, 1)} cm\n")

ctrl.disconnect()
