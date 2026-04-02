
import sys, json, time
import numpy as np
from pathlib import Path
sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")

from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig

FOLLOWER_PORT = "COM7"
LEADER_PORT   = "COM6"
CALIB_F = r"C:\Users\simal\.cache\huggingface\lerobot\calibration\robots\so_follower"
CALIB_L = r"C:\Users\simal\.cache\huggingface\lerobot\calibration\teleoperators\so_leader"
POSES_FILE = Path(__file__).parent / "poses.json"
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

f_cfg = SOFollowerRobotConfig(port=FOLLOWER_PORT, id="my_follower",
    disable_torque_on_disconnect=False, cameras={}, calibration_dir=Path(CALIB_F))
follower = SOFollower(f_cfg); follower.connect()

l_cfg = SOLeaderTeleopConfig(port=LEADER_PORT, id="my_leader", calibration_dir=Path(CALIB_L))
leader = SOLeader(l_cfg); leader.connect()

poses = {}
if POSES_FILE.exists():
    poses = json.loads(POSES_FILE.read_text())
    print(f"Loaded existing poses: {list(poses.keys())}")

print("\nTeleop active. SPACE = save pose, Q = quit\n")

import msvcrt

def kbhit():
    return msvcrt.kbhit()

def getch():
    return msvcrt.getch().decode(errors='ignore')

while True:
    action = leader.get_action()
    follower.send_action(action)

    if kbhit():
        ch = getch()
        if ch == ' ':
            obs = follower.get_observation()
            joints = [float(obs[f"{m}.pos"]) for m in MOTORS]
            label = input("\n  Label for this pose: ").strip()
            if label:
                poses[label] = joints
                POSES_FILE.write_text(json.dumps(poses, indent=2))
                print(f"  Saved '{label}': {[round(j,1) for j in joints]}\n")
        elif ch in ('q', 'Q'):
            break

    time.sleep(1/30)

follower.disconnect()
leader.disconnect()
print(f"\nAll poses saved to {POSES_FILE}:")
for k, v in poses.items():
    print(f"  {k}: {[round(j,1) for j in v]}")
