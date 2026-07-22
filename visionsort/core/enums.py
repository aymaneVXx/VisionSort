from __future__ import annotations

from enum import Enum


class SourceType(str, Enum):
    REPLAY = "REPLAY"
    VIDEO_FILE = "VIDEO_FILE"
    RTSP = "RTSP"


class CameraRole(str, Enum):
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"


class SourceStatus(str, Enum):
    OFFLINE = "OFFLINE"
    CONNECTING = "CONNECTING"
    REPLAY = "REPLAY"
    LIVE = "LIVE"
    DEGRADED = "DEGRADED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


# Alias for backward compatibility if needed
CameraStatus = SourceStatus


class CommandStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ModelTask(str, Enum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    POSE = "pose"


class ModelStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    CHAMPION = "CHAMPION"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"


class AnnotationStatus(str, Enum):
    AUTO_ACCEPTED = "AUTO_ACCEPTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    HUMAN_VALIDATED = "HUMAN_VALIDATED"
    REJECTED = "REJECTED"


class ParcelState(str, Enum):
    ON_CONVEYOR = "ON_CONVEYOR"
    PICK_CANDIDATE = "PICK_CANDIDATE"
    PICKED = "PICKED"
    CARRIED = "CARRIED"
    DROP_CANDIDATE = "DROP_CANDIDATE"
    DROPPED = "DROPPED"


class MatchResult(str, Enum):
    MATCHED = "MATCHED"
    AMBIGUOUS = "AMBIGUOUS"
    UNMATCHED = "UNMATCHED"


class JobType(str, Enum):
    SUPERVISOR = "SUPERVISOR"
    CAMERA = "CAMERA"
    GPU_INFERENCE = "GPU_INFERENCE"
    TRAINING = "TRAINING"
    DATASET = "DATASET"


class CommandType(str, Enum):
    START_SOURCE = "START_SOURCE"
    STOP_SOURCE = "STOP_SOURCE"
    START_RECORDING = "START_RECORDING"
    STOP_RECORDING = "STOP_RECORDING"
    TEST_SOURCE = "TEST_SOURCE"
    CREATE_DATASET = "CREATE_DATASET"
    PSEUDO_ANNOTATE = "PSEUDO_ANNOTATE"
    START_TRAINING = "START_TRAINING"
    ACTIVATE_MODEL = "ACTIVATE_MODEL"
    ROLLBACK_MODEL = "ROLLBACK_MODEL"
    REGISTER_SOURCE = "REGISTER_SOURCE"
    UPSERT_SITE_CONFIG = "UPSERT_SITE_CONFIG"
    BOOTSTRAP_DEMO = "BOOTSTRAP_DEMO"
