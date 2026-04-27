"""Abstract auditor interface — one client per auditee facial-analysis system.

Each auditor wraps a single facial-analysis system (commercial API or open-source
model) and exposes a uniform `predict(image, task)` interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class AuditTask(str, Enum):
    GENDER = "gender"
    AGE = "age"
    RACE = "race"
    EMOTION = "emotion"
    IDENTITY_VERIFICATION = "identity_verification"
    FACE_DETECTION = "face_detection"
    ATTRACTIVENESS = "attractiveness"


@dataclass
class AuditPrediction:
    auditor: str
    image_path: str
    task: AuditTask
    prediction: Any  # task-dependent: str for gender/race/emotion, int/float for age
    confidence: float | None
    raw_response: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class Auditor(ABC):
    name: str = "base"

    @abstractmethod
    def supported_tasks(self) -> list[AuditTask]:
        """Tasks this auditor can perform."""

    @abstractmethod
    def predict(self, image_path: str | Path, task: AuditTask) -> AuditPrediction:
        """Run a single task on a single image. Should NOT raise — return prediction with error field."""

    def predict_all(self, image_path: str | Path) -> list[AuditPrediction]:
        return [self.predict(image_path, task) for task in self.supported_tasks()]
