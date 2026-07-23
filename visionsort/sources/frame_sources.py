from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from visionsort.core.enums import SourceType
from visionsort.core.types import Frame


@dataclass(slots=True)
class SourceSettings:
    session_id: str
    camera_id: str
    camera_role: str
    uri: str
    session_start_global: float
    replay_fps: float = 8.0
    replay_offset_ms: float = 0.0
    loop: bool = False
    reconnect_delay_s: float = 2.0


class OpenCVSourceBase:
    def __init__(self, settings: SourceSettings):
        self.settings = settings
        self.capture: cv2.VideoCapture | None = None
        self.frame_index = 0
        self.stream_epoch = 0
        self._opened_once = False
        self._replay_epoch_offset = 0.0
        self._video_fps = 0.0
        self._video_t0_local = 0.0

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.settings.uri)
        if not self.capture or not self.capture.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source {self.settings.uri}")
        if self._opened_once:
            self.stream_epoch += 1
        else:
            self.frame_index = 0
            self._opened_once = True
        try:
            self._video_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            self._video_fps = 0.0
        self._video_t0_local = self._current_local_ts()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _read_raw(self) -> tuple[bool, any]:
        if self.capture is None:
            raise RuntimeError("Source non ouverte")
        return self.capture.read()

    def _current_local_ts(self) -> float:
        if self.capture is None:
            return 0.0
        pos_msec = float(self.capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if pos_msec > 0.0:
            return pos_msec / 1000.0
        if self._video_fps > 0.0:
            return float(self.frame_index) / self._video_fps
        return float(self.frame_index) / max(self.settings.replay_fps, 1.0)

    def _build_frame(self, image) -> Frame:
        timestamp_local = self._current_local_ts()
        offset_s = float(self.settings.replay_offset_ms) / 1000.0
        timestamp_global = (
            float(self.settings.session_start_global)
            + self._replay_epoch_offset
            + (timestamp_local - self._video_t0_local)
            + offset_s
        )
        frame = Frame(
            session_id=self.settings.session_id,
            camera_id=self.settings.camera_id,
            camera_role=self.settings.camera_role,
            frame_index=self.frame_index,
            timestamp_local=timestamp_local,
            timestamp_global=timestamp_global,
            image=image,
            source_fps=self._video_fps or self.settings.replay_fps,
            stream_epoch=self.stream_epoch,
        )
        self.frame_index += 1
        return frame


class VideoFileSource(OpenCVSourceBase):
    source_type = SourceType.VIDEO_FILE

    def read(self) -> Frame | None:
        ok, image = self._read_raw()
        if not ok:
            return None
        return self._build_frame(image)


class ReplaySource(OpenCVSourceBase):
    source_type = SourceType.REPLAY

    def read(self) -> Frame | None:
        ok, image = self._read_raw()
        if not ok:
            if not self.settings.loop:
                return None
            if self.capture is None:
                return None
            self._replay_epoch_offset += max(
                self._current_local_ts(),
                float(self.frame_index)
                / max(self._video_fps or self.settings.replay_fps, 1.0),
            )
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_index = 0
            self.stream_epoch += 1
            ok, image = self._read_raw()
            if not ok:
                return None
        if self.settings.replay_fps > 0:
            time.sleep(max(0.0, 1.0 / self.settings.replay_fps))
        return self._build_frame(image)


class RTSPSource(OpenCVSourceBase):
    source_type = SourceType.RTSP

    def read(self) -> Frame | None:
        ok, image = self._read_raw()
        if ok:
            timestamp_global = time.time()
            frame = Frame(
                session_id=self.settings.session_id,
                camera_id=self.settings.camera_id,
                camera_role=self.settings.camera_role,
                frame_index=self.frame_index,
                timestamp_local=timestamp_global,
                timestamp_global=timestamp_global,
                image=image,
                source_fps=self._video_fps or 0.0,
                stream_epoch=self.stream_epoch,
            )
            self.frame_index += 1
            return frame
        self.close()
        time.sleep(self.settings.reconnect_delay_s)
        self.open()
        ok, image = self._read_raw()
        if not ok:
            return None
        timestamp_global = time.time()
        frame = Frame(
            session_id=self.settings.session_id,
            camera_id=self.settings.camera_id,
            camera_role=self.settings.camera_role,
            frame_index=self.frame_index,
            timestamp_local=timestamp_global,
            timestamp_global=timestamp_global,
            image=image,
            source_fps=self._video_fps or 0.0,
            stream_epoch=self.stream_epoch,
        )
        self.frame_index += 1
        return frame


def build_source(
    *,
    source_type: str,
    session_id: str,
    camera_id: str,
    camera_role: str,
    uri: str,
    session_start_global: float,
    replay_fps: float = 8.0,
    replay_offset_ms: float = 0.0,
    loop: bool = False,
) -> OpenCVSourceBase:
    settings = SourceSettings(
        session_id=session_id,
        camera_id=camera_id,
        camera_role=camera_role,
        uri=uri,
        session_start_global=session_start_global,
        replay_fps=replay_fps,
        replay_offset_ms=replay_offset_ms,
        loop=loop,
    )
    normalized = source_type.upper()
    if normalized == SourceType.REPLAY.value:
        return ReplaySource(settings)
    if normalized == SourceType.VIDEO_FILE.value:
        return VideoFileSource(settings)
    if normalized == SourceType.RTSP.value:
        return RTSPSource(settings)
    raise ValueError(f"Type de source non supporté: {source_type}")


def can_open_uri(uri: str) -> tuple[bool, str]:
    if uri.startswith("rtsp://"):
        capture = cv2.VideoCapture(uri)
    else:
        path = Path(uri)
        if not path.exists():
            return False, "Le chemin vidéo n'existe pas."
        capture = cv2.VideoCapture(str(path))
    ok = bool(capture.isOpened())
    capture.release()
    return ok, "" if ok else "Échec d'ouverture OpenCV."
