import time
from rclpy.node import Node
from typing import Dict, Tuple, List
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras import CameraConfig


class CameraSensorManager:
    def __init__(self, node: Node, config: Dict[str, CameraConfig], fps: float = 30.0) -> None:
        self.node = node
        self.config = config
        self._cameras = make_cameras_from_configs(self.config)
        self.bridge = CvBridge()

        # data
        self._raw_data: Dict[str, np.ndarray | None] = {cam_name: None for cam_name in self._cameras.keys()}
        self._last_frame_time: Dict[str, float | None] = {cam_name: None for cam_name in self._cameras.keys()}
        self._stale_after_s: float = 2.0
        
        self._is_connected = False
        self.is_initialized = False

        self._pubs = {}
        for cam_name in self.names:
            topic_name = f"/camera/{cam_name}/image_raw"
            self._pubs[cam_name] = self.node.create_publisher(Image, topic_name, 10)

        self._update_timer = self.node.create_timer(
            1.0 / fps, self._update_timer_callback
        )
    
    def connect(self) -> None:
        if not self._is_connected:
            self._is_connected = True
            for cam in self._cameras.values():
                cam.connect()
    
    def disconnect(self) -> None:
        if self._is_connected:
            self._is_connected = False
            for cam in self._cameras.values():
                cam.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def missing_frames(self) -> List[str]:
        now = time.monotonic()
        missing: List[str] = []
        for cam_name, frame in self._raw_data.items():
            last_frame_time = self._last_frame_time[cam_name]
            if frame is None or last_frame_time is None:
                missing.append(cam_name)
            elif now - last_frame_time > self._stale_after_s:
                missing.append(cam_name)
        return missing

    @property
    def is_ready(self) -> bool:
        return self._is_connected and self.is_initialized and not self.missing_frames
    
    @property
    def names(self) -> List[str]:
        return list(self._cameras.keys())
    
    @property
    def shapes(self) -> Dict[str, Tuple[int, int, int]]:
        return {
            cam_name: (self._cameras[cam_name].height, self._cameras[cam_name].width, 3) for cam_name in self.names
        }
    
    @property
    def data(self) -> Dict[str, np.ndarray]:
        if self.is_initialized:
            return self._raw_data
        else:
            return {
                cam_name: np.zeros(self.shapes[cam_name], dtype=np.uint8) for cam_name in self.names
            }
    
    def _update_timer_callback(self) -> None:
        if not self._is_connected:
            return

        try:
            for cam_name, cam in self._cameras.items():
                frame = cam.async_read(timeout_ms=1000)
                if frame is not None:
                    self._raw_data[cam_name] = frame
                    self._last_frame_time[cam_name] = time.monotonic()
            
            if not self.is_initialized:
                if all(frame is not None for frame in self._raw_data.values()):
                    self.is_initialized = True
                    self.node.get_logger().info("All cameras initialized successfully. Starting to publish.")

            if self.is_initialized:
                self._publish_data()
                
        except Exception as e:
            # Ignore read errors caused by the camera being shut down first during disconnect, to prevent a segfault
            pass

    def _publish_data(self) -> None:
        now_msg = self.node.get_clock().now().to_msg()

        for cam_name, frame in self._raw_data.items():
            # Even if a frame is lost this loop and only a previous frame (or None) remains,
            # process it while is_initialized is True (skip it when it is None)
            if frame is None:
                continue 

            try:
                img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
                img_msg.header.stamp = now_msg
                img_msg.header.frame_id = f"{cam_name}_camera_link"

                self._pubs[cam_name].publish(img_msg)
            except Exception as e:
                self.node.get_logger().warn(f"Failed to publish {cam_name} image: {e}")
