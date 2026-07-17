import cv2
import numpy as np


def main() -> None:
    from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
    from lerobot_robot_fr3.fr3 import FR3Robot

    print("[INFO] Connecting to Robot...")
    config = FR3RobotConfig(is_relative=False, rotation_type="quaternion", arm_action_dim=7)
    robot = FR3Robot(config)

    try:
        robot.connect()
    except Exception as exc:
        print(f"[ERROR] Connection failed: {exc}")
        return

    if not robot.is_connected:
        print("[ERROR] Failed to connect to robot.")
        return

    print("[INFO] Starting Control & Visualization Loop. Press 'q' to quit.")

    try:
        while True:
            obs_dict: dict[str, np.ndarray] = robot.get_observation()
            print("=========== observation ===========")
            for obs_key, obs_value in obs_dict.items():
                print(f"{obs_key}: {obs_value.shape}")

            vla_obs_dict: dict[str, np.ndarray] = robot.get_vla_observation()
            print("=========== vla observation ===========")
            for obs_key, obs_value in vla_obs_dict.items():
                print(f"{obs_key}: {obs_value.shape}")

            panels: list[np.ndarray] = []
            cam_keys: list[str] = ["wrist", "left", "right"]

            for key in cam_keys:
                obs_key = f"video.{key}_view"
                if obs_key in vla_obs_dict:
                    frame: np.ndarray = vla_obs_dict[obs_key]
                    if frame.ndim == 4:
                        frame = frame[0]

                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    label = key
                else:
                    camera_config = config.cameras[key]
                    bgr = np.zeros((camera_config.height, camera_config.width, 3), dtype=np.uint8)
                    label = f"{key} (no frame)"

                cv2.putText(
                    bgr,
                    label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                panels.append(bgr)

            if panels:
                combined: np.ndarray = np.hstack(panels)
                cv2.imshow("FR3 Robot Multi-Camera Viewer", combined)

            action_dict: dict[str, np.ndarray] = {
                "arm_actions": np.random.random((6,)),
                "gripper_actions": np.random.random((1,)),
            }
            robot.send_action(action_dict)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by User.")
    except Exception as exc:
        print(f"\n[ERROR] Runtime error: {exc}")
    finally:
        cv2.destroyAllWindows()
        robot.disconnect()
        print("[INFO] Robot Disconnected Safely.")


if __name__ == "__main__":
    main()
