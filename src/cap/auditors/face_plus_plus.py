"""Face++ (Megvii) auditor — REST API, full attribute set."""
from __future__ import annotations

import os
from pathlib import Path

from cap.auditors.base import Auditor, AuditPrediction, AuditTask


class FacePlusPlusAuditor(Auditor):
    name = "face_plus_plus"

    def __init__(self, api_key: str | None = None, api_secret: str | None = None):
        self.api_key = api_key or os.getenv("FACEPP_API_KEY")
        self.api_secret = api_secret or os.getenv("FACEPP_API_SECRET")

    def supported_tasks(self) -> list[AuditTask]:
        return [AuditTask.GENDER, AuditTask.AGE, AuditTask.EMOTION, AuditTask.RACE, AuditTask.FACE_DETECTION]

    def predict(self, image_path: str | Path, task: AuditTask) -> AuditPrediction:
        # TODO: implement requests POST to https://api-us.faceplusplus.com/facepp/v3/detect
        return AuditPrediction(self.name, str(image_path), task, None, None, error="not_implemented")
