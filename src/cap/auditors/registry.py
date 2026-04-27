"""Build auditor instances from config."""
from __future__ import annotations

from typing import Any

from cap.auditors.base import Auditor
from cap.auditors.aws_rekognition import AWSRekognitionAuditor
from cap.auditors.azure_face import AzureFaceAuditor
from cap.auditors.google_vision import GoogleVisionAuditor
from cap.auditors.face_plus_plus import FacePlusPlusAuditor
from cap.auditors.deepface_local import DeepFaceAuditor
from cap.auditors.insightface_local import InsightFaceAuditor
from cap.auditors.arcface_local import ArcFaceAuditor

_REGISTRY = {
    "aws_rekognition": AWSRekognitionAuditor,
    "azure_face": AzureFaceAuditor,
    "google_vision": GoogleVisionAuditor,
    "face_plus_plus": FacePlusPlusAuditor,
    "deepface": DeepFaceAuditor,
    "insightface": InsightFaceAuditor,
    "arcface": ArcFaceAuditor,
}


def build_auditors(specs: list[dict[str, Any]]) -> list[Auditor]:
    """Build a list of auditors from config specs.

    Each spec: {"name": "aws_rekognition", "kwargs": {...}}
    """
    auditors: list[Auditor] = []
    for spec in specs:
        name = spec["name"]
        kwargs = spec.get("kwargs", {}) or {}
        cls = _REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"Unknown auditor: {name}. Known: {list(_REGISTRY)}")
        auditors.append(cls(**kwargs))
    return auditors
