from __future__ import annotations


def main() -> None:
    import rclpy
    from lerobot_ft_sensor import FTSensor, FTSensorConfig
    from rclpy.node import Node
    from rclpy.timer import Timer

    rclpy.init()

    node = Node("test_ft_sensor_node")

    config = FTSensorConfig()

    node.get_logger().info("Initializing FT Sensor...")
    ft_sensor = FTSensor(node=node, config=config)
    ft_sensor.connect()
    last_stamp: float | None = None

    def print_data_callback() -> None:
        nonlocal last_stamp
        if ft_sensor.is_initialized:
            force = ft_sensor.force
            torque = ft_sensor.torque
            stamp = ft_sensor.timestamp
            now_msg = node.get_clock().now().to_msg()
            now = float(now_msg.sec) + float(now_msg.nanosec) * 1e-9
            age_ms = (now - stamp) * 1000.0
            step_ms = 0.0 if last_stamp is None else (stamp - last_stamp) * 1000.0
            last_stamp = stamp

            node.get_logger().info(
                f"Force: [Fx: {force[0]:>7.3f}, Fy: {force[1]:>7.3f}, Fz: {force[2]:>7.3f}] N | "
                f"Torque: [Tx: {torque[0]:>7.3f}, Ty: {torque[1]:>7.3f}, Tz: {torque[2]:>7.3f}] Nm | "
                f"stamp: {stamp:.9f} | age: {age_ms:>7.2f} ms | dt: {step_ms:>7.2f} ms"
            )
        else:
            node.get_logger().warn("Waiting for sensor initialization...", throttle_duration_sec=1.0)

    _print_timer: Timer = node.create_timer(0.5, print_data_callback)

    try:
        node.get_logger().info("Spinning node. Press Ctrl+C to stop.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected. Shutting down...")
    finally:
        ft_sensor.disconnect()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
