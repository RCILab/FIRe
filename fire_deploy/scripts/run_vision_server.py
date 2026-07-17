from vision_server import VisionServer
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Vision Server (camera streams over ZMQ).")
    parser.add_argument("--width_size", type=int, default=256, help="Published image width.")
    parser.add_argument("--height_size", type=int, default=256, help="Published image height.")
    args = parser.parse_args()
    if args.width_size <= 0:
        parser.error("--width_size must be greater than 0")
    if args.height_size <= 0:
        parser.error("--height_size must be greater than 0")

    MY_CAMERAS = {
        "wrist": "844212070094",
        "left":  "2F1C99DF",
        "right": "16FB99DF"
    }

    server = VisionServer(
        camera_configs=MY_CAMERAS,
        port=5555,
        width_size=args.width_size,
        height_size=args.height_size,
    )
    server.start()
