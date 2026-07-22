from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from visionsort.core.enums import SourceType
from visionsort.core.types import Frame


@dataclass(slots=True)
class SourceSettings:
    camera_id: str
    uri: str
    replay_fps: float = 8.0
    loop: bool = True
    reconnect_delay_s: float = 2.0


class OpenCVSourceBase:
    def __init__(self, settings: SourceSettings):
        self.settings = settings
        self.capture: cv2.VideoCapture | None = None
        self.frame_index = 0

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.settings.uri)
        self.frame_index = 0
        if not self.capture or not self.capture.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source {self.settings.uri}")

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _read_raw(self) -> tuple[bool, any]:
        if self.capture is None:
            raise RuntimeError("Source non ouverte")
        return self.capture.read()

    def _build_frame(self, image) -> Frame:
        frame = Frame(
            camera_id=self.settings.camera_id,
            frame_index=self.frame_index,
            timestamp=time.time(),
            image=image,
            source_fps=self.settings.replay_fps,
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
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_index = 0
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
            return self._build_frame(image)
        self.close()
        time.sleep(self.settings.reconnect_delay_s)
        self.open()
        ok, image = self._read_raw()
        if not ok:
            return None
        return self._build_frame(image)


def build_source(source_type: str, camera_id: str, uri: str, replay_fps: float = 8.0) -> OpenCVSourceBase:
    settings = SourceSettings(camera_id=camera_id, uri=uri, replay_fps=replay_fps)
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
