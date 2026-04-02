from __future__ import annotations

from .config import CollectDataConfig


def create_controller(config: CollectDataConfig):
    if not config.use_master_arm:
        print("Используется управление с клавиатуры")
        return None

    from master_arm_control import SO101MasterArmController
    controller = SO101MasterArmController(
        port=config.leader_port,
        leader_id=config.leader_id,
        use_degrees=False,
        motion_threshold=config.motion_threshold,
    )
    controller.connect()
    print("Мастер-рука подключена")
    return controller


def disconnect_controller(controller) -> None:
    if controller is not None:
        controller.disconnect()
