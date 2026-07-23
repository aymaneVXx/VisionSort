import threading
import time

import numpy as np

from visionsort.acquisition.worker import LatestFrameBuffer
from visionsort.core.types import Frame


class _FastSource:
    def __init__(self, count: int = 30):
        self.count = count
        self.index = 0
        self.closed = False

    def open(self) -> None:
        self.closed = False

    def read(self):
        if self.index >= self.count or self.closed:
            return None
        index = self.index
        self.index += 1
        time.sleep(0.001)
        return Frame(
            session_id="s",
            camera_id="c",
            camera_role="C1",
            frame_index=index,
            timestamp_local=index * 0.01,
            timestamp_global=100.0 + index * 0.01,
            image=np.zeros((8, 8, 3), dtype=np.uint8),
        )

    def close(self) -> None:
        self.closed = True


def test_latest_frame_buffer_acquires_while_consumer_is_blocked():
    source = _FastSource()
    buffer = LatestFrameBuffer(source, capacity=3)
    buffer.start()

    first = buffer.take_latest(timeout=1.0)
    assert first is not None
    original_timestamp = first.timestamp_global
    consumer_released = threading.Event()

    def slow_inference():
        time.sleep(0.04)
        consumer_released.set()

    thread = threading.Thread(target=slow_inference)
    thread.start()
    thread.join()
    latest = buffer.take_latest(timeout=1.0)
    buffer.mark_processed()
    buffer.stop()

    assert consumer_released.is_set()
    assert latest is not None
    assert latest.frame_index > first.frame_index
    assert latest.timestamp_global > original_timestamp
    metrics = buffer.metrics()
    assert metrics["frames_received"] > metrics["frames_processed"]
    assert metrics["frames_dropped"] > 0
    assert metrics["buffered_frames"] <= 3
