

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation




_BODY_CHAIN = [

    ("base",      [0.0388353, 0.0, 0.0624],  [1, 0, 0, 0]),

    ("shoulder",  [0.0388353, 0.0, 0.0624],
                  [3.56167e-16, 1.22818e-15, -1, -4.14635e-16]),

    ("upper_arm", [-0.0303992, -0.0182778, -0.0542],
                  [0.5, -0.5, -0.5, -0.5]),

    ("lower_arm", [-0.11257, -0.028, 0.0],
                  [0.707107, 0.0, 0.0, 0.707107]),

    ("wrist",     [-0.1349, 0.0052, 0.0],
                  [0.707107, 0.0, 0.0, -0.707107]),

    ("gripper",   [0.0, -0.0611, 0.0181],
                  [0.0172091, -0.0172091, 0.706897, 0.706897]),

    ("tcp",       [-0.0079, -0.000218121, -0.0981274],
                  [0.707107, 0.0, 0.707107, 0.0]),
]







_JOINT_BODY_IDX = [1, 2, 3, 4, 5]


JOINT_LIMITS_RAD = np.array([
    [-1.9199, 1.9199],
    [-1.7453, 1.7453],
    [-1.69,   1.69],
    [-1.6581, 1.6581],
    [-2.7438, 2.8412],
])

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


GRIPPER_MIN_RAD = -0.17453
GRIPPER_MAX_RAD = 1.74533
GRIPPER_MIN_REAL = 0.0
GRIPPER_MAX_REAL = 100.0


def _quat_to_matrix(q_wxyz):
    """Convert (w,x,y,z) quaternion to 3Г—3 rotation matrix."""
    w, x, y, z = q_wxyz
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def _make_transform(pos, quat_wxyz):
    """Build 4Г—4 homogeneous transform from position + quaternion."""
    T = np.eye(4)
    T[:3, :3] = _quat_to_matrix(quat_wxyz)
    T[:3, 3] = pos
    return T


def _rot_z(angle_rad):
    """4Г—4 rotation about local Z axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    T = np.eye(4)
    T[0, 0] = c;  T[0, 1] = -s
    T[1, 0] = s;  T[1, 1] = c
    return T


def forward_kinematics(joint_angles_rad):

    angles = np.asarray(joint_angles_rad, dtype=np.float64)
    assert len(angles) == 5

    T = np.eye(4)
    joint_idx = 0

    for i, (name, pos, quat) in enumerate(_BODY_CHAIN):

        if i in _JOINT_BODY_IDX and joint_idx < 5:
            T = T @ _rot_z(angles[joint_idx])
            joint_idx += 1

        T = T @ _make_transform(pos, quat)

    return T[:3, 3].copy(), T[:3, :3].copy()


def forward_kinematics_deg(joint_angles_deg):

    return forward_kinematics(np.deg2rad(joint_angles_deg))


def inverse_kinematics(
    target_pos,
    initial_guess_rad=None,
    pos_weight=1.0,
    max_iter=500,
):

    target = np.asarray(target_pos, dtype=np.float64)

    if initial_guess_rad is None:
        initial_guess_rad = np.zeros(5)
    x0 = np.asarray(initial_guess_rad, dtype=np.float64)

    bounds = [(lo, hi) for lo, hi in JOINT_LIMITS_RAD]

    def cost(angles):
        tcp_pos, _ = forward_kinematics(angles)
        pos_err = np.sum((tcp_pos - target) ** 2) * pos_weight
        return pos_err

    result = minimize(
        cost, x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-12},
    )

    final_pos, _ = forward_kinematics(result.x)
    error = np.linalg.norm(final_pos - target)
    return result.x, result.success and error < 0.01, error


def inverse_kinematics_deg(target_pos, initial_guess_deg=None):

    ig = np.deg2rad(initial_guess_deg) if initial_guess_deg is not None else None
    angles_rad, success, error = inverse_kinematics(target_pos, ig)
    return np.rad2deg(angles_rad), success, error


def gripper_real_to_rad(value_0_100):

    alpha = np.clip(value_0_100, 0, 100) / 100.0
    return GRIPPER_MIN_RAD + alpha * (GRIPPER_MAX_RAD - GRIPPER_MIN_RAD)


def gripper_rad_to_real(rad):

    rad = np.clip(rad, GRIPPER_MIN_RAD, GRIPPER_MAX_RAD)
    alpha = (rad - GRIPPER_MIN_RAD) / (GRIPPER_MAX_RAD - GRIPPER_MIN_RAD)
    return alpha * 100.0



if __name__ == "__main__":

    pos, rot = forward_kinematics(np.zeros(5))
    print(f"Home TCP pos: {pos}")
    print(f"Home TCP pos (mm): {pos * 1000}")


    test_angles = np.array([0.3, -0.5, 0.8, -0.3, 0.0])
    target_pos, _ = forward_kinematics(test_angles)
    recovered, success, err = inverse_kinematics(target_pos)
    print(f"\nIK test:")
    print(f"  Target pos: {target_pos}")
    print(f"  Recovered angles: {np.rad2deg(recovered)}")
    print(f"  Original angles:  {np.rad2deg(test_angles)}")
    print(f"  Success: {success}, Error: {err:.6f} m")
