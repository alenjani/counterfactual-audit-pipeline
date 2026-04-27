"""Google Cloud Vision auditor — limited face attributes (no gender, age, race)."""
from __future__ import annotations

from pathlib import Path

from cap.auditors.base import Auditor, AuditPrediction, AuditTask


class GoogleVisionAuditor(Auditor):
    name = "google_vision"

    def supported_tasks(self) -> list[AuditTask]:
        return [AuditTask.FACE_DETECTION, AuditTask.EMOTION]

    def predict(self, image_path: str | Path, task: AuditTask) -> AuditPrediction:
        # TODO: implement google.cloud.vision client.
        return AuditPrediction(self.name, str(image_path), task, None, None, error="not_implemented")
