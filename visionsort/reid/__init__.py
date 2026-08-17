"""Local, open-set parcel ReID and conservative multicamera association."""

from visionsort.reid.encoder import ParcelReIDEncoder, ProjectionHead
from visionsort.reid.handoff import HandoffCandidateGenerator, HandoffScorer
from visionsort.reid.keyframes import HandoffKeyframeSelector

__all__ = [
    "HandoffCandidateGenerator",
    "HandoffKeyframeSelector",
    "HandoffScorer",
    "ParcelReIDEncoder",
    "ProjectionHead",
]
