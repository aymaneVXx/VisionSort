import queue
import time
import uuid

import numpy as np

from visionsort.acquisition.worker import build_inference_request
from visionsort.core.types import Frame
from visionsort.runtime.supervisor import RuntimeSupervisor
from visionsort.sources import frame_sources
from visionsort.sources.frame_sources import ReplaySource, RTSPSource, SourceSettings


class _FakeCapture:
    def __init__(self):
        self.cursor = 0
        self.opened = True

    def isOpened(self):
        return self.opened

    def read(self):
        if self.cursor > 0:
            return False, None
        self.cursor += 1
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def get(self, prop):
        if prop == frame_sources.cv2.CAP_PROP_FPS:
            return 8.0
        if prop == frame_sources.cv2.CAP_PROP_POS_MSEC:
            return self.cursor * 125.0
        return 0.0

    def set(self, prop, value):
        if prop == frame_sources.cv2.CAP_PROP_POS_FRAMES and value == 0:
            self.cursor = 0
        return True

    def release(self):
        self.opened = False


def _frame(session_id: str, stream_epoch: int) -> Frame:
    return Frame(
        session_id=session_id,
        camera_id="camera-1",
        camera_role="C1",
        frame_index=0,
        timestamp_local=0.0,
        timestamp_global=100.0,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        stream_epoch=stream_epoch,
    )


def test_request_ids_never_collide_between_sessions_or_stream_epochs():
    requests = [
        build_inference_request(
            frame=_frame("session-a", 0), source_id="source-1", ttl_seconds=2.0
        ),
        build_inference_request(
            frame=_frame("session-b", 0), source_id="source-1", ttl_seconds=2.0
        ),
        build_inference_request(
            frame=_frame("session-b", 1), source_id="source-1", ttl_seconds=2.0
        ),
    ]

    assert len({request["request_id"] for request in requests}) == 3
    assert all(uuid.UUID(request["request_id"]) for request in requests)
    assert requests[0]["session_id"] != requests[1]["session_id"]
    assert requests[1]["stream_epoch"] != requests[2]["stream_epoch"]


def test_replay_is_non_looping_by_default_and_explicit_loop_increments_epoch(
    monkeypatch,
):
    monkeypatch.setattr(frame_sources.cv2, "VideoCapture", lambda _: _FakeCapture())
    base = {
        "session_id": "session-a",
        "camera_id": "camera-1",
        "camera_role": "C1",
        "uri": "fake.avi",
        "session_start_global": 100.0,
        "replay_fps": 0.0,
    }
    non_looping = ReplaySource(SourceSettings(**base))
    non_looping.open()
    assert non_looping.read() is not None
    assert non_looping.read() is None

    looping = ReplaySource(SourceSettings(**base, loop=True))
    looping.open()
    first = looping.read()
    second = looping.read()

    assert first is not None and second is not None
    assert first.frame_index == second.frame_index == 0
    assert first.stream_epoch == 0
    assert second.stream_epoch == 1
    assert second.timestamp_global >= first.timestamp_global


def test_rtsp_reconnection_keeps_monotone_frame_index_and_increments_epoch(
    monkeypatch,
):
    monkeypatch.setattr(frame_sources.cv2, "VideoCapture", lambda _: _FakeCapture())
    source = RTSPSource(
        SourceSettings(
            session_id="session-rtsp",
            camera_id="camera-rtsp",
            camera_role="C1",
            uri="rtsp://fake",
            session_start_global=100.0,
            reconnect_delay_s=0.0,
        )
    )
    source.open()
    first = source.read()
    second = source.read()

    assert first is not None and second is not None
    assert (first.frame_index, second.frame_index) == (0, 1)
    assert (first.stream_epoch, second.stream_epoch) == (0, 1)


def test_supervisor_rejects_stale_late_and_expired_results():
    supervisor = RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.inference_result_queue = queue.Queue()
    supervisor.inference_result_store = {}
    supervisor.active_source_sessions = {"source-1": "session-current"}
    supervisor.latest_stream_epoch_by_source = {"source-1": 2}

    def result(*, session_id="session-current", epoch=2, expires_at=None):
        return {
            "kind": "INFER_RESULT",
            "request_id": str(uuid.uuid4()),
            "session_id": session_id,
            "source_id": "source-1",
            "camera_id": "camera-1",
            "stream_epoch": epoch,
            "frame_index": 1,
            "expires_at": expires_at or time.time() + 5.0,
            "observations": [],
        }

    stale_session = result(session_id="session-old")
    stale_epoch = result(epoch=1)
    late = result(expires_at=time.time() - 1.0)
    valid = result()
    for message in (stale_session, stale_epoch, late, valid):
        supervisor.inference_result_queue.put(message)

    supervisor.drain_inference_results()

    assert valid["request_id"] in supervisor.inference_result_store
    assert stale_session["request_id"] not in supervisor.inference_result_store
    assert stale_epoch["request_id"] not in supervisor.inference_result_store
    assert late["request_id"] not in supervisor.inference_result_store
    metrics = supervisor.inference_result_store[
        "__inference_metrics__:source-1"
    ]
    assert metrics["ignored"] == 2
    assert metrics["late"] == 1

    supervisor.inference_result_store[valid["request_id"]]["expires_at"] = (
        time.time() - 1.0
    )
    supervisor._cleanup_expired_inference_results()
    assert valid["request_id"] not in supervisor.inference_result_store
    assert supervisor.inference_result_store[
        "__inference_metrics__:source-1"
    ]["expired"] == 1
