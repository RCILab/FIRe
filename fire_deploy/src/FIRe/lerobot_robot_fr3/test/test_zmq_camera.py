"""
ZMQ camera visualization test using the LeRobot ZMQCamera.

Run:
    python src/FIRe/lerobot_robot_fr3/test/test_zmq_camera.py --host 192.168.x.x --port 5555
"""

import argparse
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from lerobot.cameras.zmq import ZMQCamera


CAM_NAMES: list[str] = ["wrist", "left", "right"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="163.180.132.241", help="Server IP")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port")
    parser.add_argument("--timeout", type=int, default=5000, help="Receive timeout in ms")
    return parser.parse_args()


def main() -> None:
    from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig

    args = parse_args()

    cameras: dict[str, ZMQCamera] = {}
    for name in CAM_NAMES:
        config = ZMQCameraConfig(
            server_address=args.host,
            port=args.port,
            camera_name=name,
            timeout_ms=args.timeout,
        )
        cameras[name] = ZMQCamera(config)

    print(f"[ZMQ Viewer] Connecting: tcp://{args.host}:{args.port}")
    for name, camera in cameras.items():
        camera.connect()
        print(f"  {name} connected ({camera.width}x{camera.height})")

    print("[ZMQ Viewer] Started. Press 'q' to quit.")

    fps_times: list[float] = []

    try:
        while True:
            t0 = time.perf_counter()

            panels: list[np.ndarray] = []
            for name, camera in cameras.items():
                try:
                    frame: np.ndarray = camera.read_latest(max_age_ms=1000)
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    label = name
                except (TimeoutError, RuntimeError):
                    bgr = np.zeros((256, 256, 3), dtype=np.uint8)
                    label = f"{name} (no frame)"

                cv2.putText(
                    bgr,
                    label,
                    (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                panels.append(bgr)

            fps_times.append(time.perf_counter() - t0)
            if len(fps_times) > 30:
                fps_times.pop(0)
            fps = len(fps_times) / sum(fps_times) if fps_times else 0.0

            combined: np.ndarray = np.hstack(panels)
            cv2.putText(
                combined,
                f"FPS: {fps:.1f}",
                (4, combined.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("ZMQ Viewer (lerobot)", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        for camera in cameras.values():
            camera.disconnect()
        cv2.destroyAllWindows()
        print("[ZMQ Viewer] Stopped.")


if __name__ == "__main__":
    main()
