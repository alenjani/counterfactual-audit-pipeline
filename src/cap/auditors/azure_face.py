"""Azure Face / Image Analysis auditor.

Note: Microsoft retired emotion detection and restricted gender classification in 2022.
We document the *availability* of each task as an empirical finding (a direct consequence
of Gender Shades' impact on industry practice).

Tasks supported (current): age, face detection.
Tasks restricted: gender (limited access program), emotion (retired).
"""
from __future__ import annotations

from pathlib import Path

from cap.auditors.base import Auditor, AuditPrediction, AuditTask
from cap.utils.logging import get_logger

logger = get_logger()


class AzureFaceAuditor(Auditor):
    name = "azure_face"

    def __init__(self, endpoint: str, key: str):
        self.endpoint = endpoint
        self.key = key
        self._client = None

    def _lazy_load(self) -> None:
        if self._client is not None:
            return
        from azure.ai.vision.imageanalysis import ImageAnalysisClient
        from azure.core.credentials import AzureKeyCredential

        self._client = ImageAnalysisClient(self.endpoint, AzureKeyCredential(self.key))

    def supported_tasks(self) -> list[AuditTask]:
        return [AuditTask.AGE, AuditTask.FACE_DETECTION]

    def predict(self, image_path: str | Path, task: AuditTask) -> AuditPrediction:
        # TODO: implement Azure Image Analysis call. Skeleton for now.
        return AuditPrediction(
            self.name, str(image_path), task, None, None,
            error="not_implemented",
        )
