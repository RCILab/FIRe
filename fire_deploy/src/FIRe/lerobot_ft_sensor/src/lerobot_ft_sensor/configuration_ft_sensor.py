from dataclasses import dataclass


@dataclass
class FTSensorConfig:
    timeout_ms: int = 200
    wrench_topic: str = "/bota_ft_sensor/wrench"
    queue_size: int = 1

    def __post_init__(self) -> None:
        if not self.wrench_topic:
            raise ValueError("FTSensorConfig.wrench_topic is required.")
        if self.queue_size <= 0:
            raise ValueError("FTSensorConfig.queue_size must be positive.")
