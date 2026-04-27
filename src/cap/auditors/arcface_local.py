"""ArcFace local auditor — identity verification only (1:1 face matching)."""
from __future__ import annotations

from pathlib import Path

from cap.auditors.base import Auditor, AuditPrediction, AuditTask


class ArcFaceAuditor(Auditor):
    name = "arcface"

    def supported_tasks(self) -> list[AuditTask]:
        return [AuditTask.IDENTITY_VERIFICATION]

    def predict(self, image_path: str | Path, task: AuditTask) -> AuditPrediction:
        # TODO: pair-based verification — needs a "claimed identity" reference image.
        # Wire from validator.identity for embedding compute.
        return AuditPrediction(self.name, str(image_path), task, None, None, error="needs_pair")
