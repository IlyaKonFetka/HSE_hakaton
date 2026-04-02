from __future__ import annotations

from .config import default_config
from .controller import create_controller, disconnect_controller
from .env_runner import close_env, collect_demonstrations, create_env, create_or_load_dataset


def main() -> None:
    config = default_config()
    env = create_env(config)
    controller = create_controller(config)
    dataset = create_or_load_dataset(config)

    try:
        collect_demonstrations(config, env, dataset, controller)
    finally:
        disconnect_controller(controller)
        close_env(env)


if __name__ == "__main__":
    main()
