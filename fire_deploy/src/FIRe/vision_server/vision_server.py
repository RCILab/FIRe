import os
import time
import json
import queue
import threading
import zmq
import base64
import cv2
import numpy as np
import logging
import contextlib
import argparse

# ---------------------------------------------------------------------------
# LeRobot camera imports
# ---------------------------------------------------------------------------
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode

# The lerobot camera imports set the root logger to WARNING first, so use
# force=True to forcibly reconfigure the INFO level/handler (otherwise INFO logs are suppressed).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger("VisionServer")
logger.setLevel(logging.INFO)

# ===========================================================================
# 1. Camera Factory
# ===========================================================================
def get_webcam_path_by_id(serial_id: str):
    base_dir = "/dev/v4l/by-id/"
    if os.path.exists(base_dir):
        for f in os.listdir(base_dir):
            if serial_id in f and "video-index0" in f:
                return os.path.join(base_dir, f)
    return 0

def create_camera(serial_id: str, width: int = 640, height: int = 480, fps: int = 30):
    if serial_id.isdigit() and len(serial_id) == 12:
        logger.info(f"Creating RealSenseCamera for serial: {serial_id}")
        config = RealSenseCameraConfig(
            serial_number_or_name=serial_id, fps=fps, width=width, height=height, color_mode=ColorMode.RGB
        )
        return RealSenseCamera(config)
    else:
        dev_path = get_webcam_path_by_id(serial_id)
        logger.info(f"Creating OpenCVCamera for ID: {serial_id} (Path: {dev_path})")
        config = OpenCVCameraConfig(
            index_or_path=dev_path, fps=fps, width=width, height=height, color_mode=ColorMode.RGB
        )
        return OpenCVCamera(config)


# ===========================================================================
# 2. Vision Server class
# ===========================================================================
class VisionServer:
    def __init__(
        self,
        camera_configs: dict,
        port: int = 5555,
        width_size: int = 256,
        height_size: int = 256,
    ) -> None:
        self.port = port
        self.width_size = width_size
        self.height_size = height_size
        self.cameras = {}

        # Initialize cameras using the factory
        for name, serial in camera_configs.items():
            self.cameras[name] = create_camera(serial)

        # Configure the ZMQ PUB socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")

        # Capture-publish pipeline queue (maxsize=1: always keep only the latest frame)
        self._raw_queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = False
        self._capture_thread = None
        self._publish_thread = None

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Publish images as contiguous uint8 RGB arrays with shape (height, width, 3)."""
        prepared = np.asarray(image)
        if prepared.ndim == 2:
            prepared = np.repeat(prepared[:, :, None], 3, axis=2)
        if prepared.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape {prepared.shape}.")
        if prepared.shape[2] == 4:
            prepared = prepared[:, :, :3]
        if prepared.shape[2] != 3:
            raise ValueError(f"Expected image with 3 channels, got shape {prepared.shape}.")
        if prepared.shape[:2] != (self.height_size, self.width_size):
            prepared = cv2.resize(
                prepared,
                (self.width_size, self.height_size),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(prepared, dtype=np.uint8)

    @staticmethod
    def _encode_image(image: np.ndarray, quality: int = 85) -> str:
        """Encode an RGB image as a Base64 JPEG string (no color conversion, LeRobot-compatible)."""
        _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return base64.b64encode(buffer).decode("utf-8")

    def _wait_for_frames(self, name, cam, timeout: float = 5.0) -> bool:
        """Verify that actual frames (data) arrive after the camera connects."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                frame = cam.read_latest(max_age_ms=1000)
                if frame is not None:
                    h, w = np.asarray(frame).shape[:2]
                    logger.info(
                        f"[DATA OK] Camera '{name}' streaming — frame received ({w}x{h})."
                    )
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        logger.error(
            f"[NO DATA] Camera '{name}' connected but no frame received within {timeout:.0f}s."
        )
        return False

    def start(self):
        logger.info("Connecting to all cameras...")
        connected = []
        failed = []
        for name, cam in self.cameras.items():
            try:
                cam.connect()
                logger.info(f"[OK] Camera '{name}' connected. Verifying data stream...")
                if self._wait_for_frames(name, cam):
                    connected.append(name)
                else:
                    failed.append(name)
                    with contextlib.suppress(Exception):
                        cam.disconnect()
            except Exception as e:
                failed.append(name)
                logger.error(f"[FAIL] Camera '{name}' failed to connect: {e}")

        # Keep only cameras confirmed to be streaming (failed cameras are excluded from the capture loop)
        for name in failed:
            self.cameras.pop(name, None)

        if not connected:
            logger.error("No cameras streaming data. Aborting start.")
            self.stop()
            return

        logger.info(
            f"{len(connected)}/{len(self.cameras)} camera(s) connected & streaming: {connected}"
        )

        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._capture_thread.start()
        self._publish_thread.start()

        logger.info("=" * 60)
        logger.info(f"VisionServer READY — publishing on tcp://*:{self.port}")
        logger.info(f"  cameras : {list(self.cameras.keys())}")
        logger.info(f"  size    : {self.width_size}x{self.height_size}")
        logger.info("  press Ctrl+C to stop")
        logger.info("=" * 60)
        try:
            while self._running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.stop()

    def _capture_loop(self):
        """Thread that continuously captures camera frames and puts them into the queue."""
        while self._running:
            try:
                raw_frames = {}
                for name, cam in self.cameras.items():
                    try:
                        raw_frames[name] = cam.read_latest(max_age_ms=1000)
                    except Exception as e:
                        logger.warning(f"Could not get frame from {name}: {e}")

                if not raw_frames:
                    time.sleep(0.01)
                    continue

                # If the queue is full, drop the stale frame and replace it with the latest one
                if self._raw_queue.full():
                    try:
                        self._raw_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._raw_queue.put_nowait(raw_frames)
            except Exception as e:
                logger.error(f"Capture error: {e}")

    def _publish_loop(self):
        """Thread that pulls frames from the queue and sends them over ZMQ."""
        while self._running:
            try:
                raw_frames = self._raw_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                t0 = time.perf_counter()

                message = {"timestamps": {}, "images": {}}
                for name, frame in raw_frames.items():
                    frame = self._prepare_image(frame)
                    message["timestamps"][name] = time.time()
                    message["images"][name] = self._encode_image(frame)

                if message["images"]:
                    with contextlib.suppress(zmq.Again):
                        self.socket.send_string(json.dumps(message), zmq.NOBLOCK)

                # Limit the publish rate (max 30Hz)
                elapsed = time.perf_counter() - t0
                sleep_time = max(0, (1.0 / 30.0) - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Publish error: {e}")

    def stop(self):
        logger.info("Shutting down VisionServer...")
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
        if self._publish_thread is not None:
            self._publish_thread.join(timeout=5.0)
        for cam in self.cameras.values():
            cam.disconnect()
        self.socket.close()
        self.context.term()
        logger.info("Disconnected safely.")
