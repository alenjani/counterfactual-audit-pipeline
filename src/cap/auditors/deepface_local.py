"""DeepFace local auditor — open-source, multi-backend face analysis."""
from __future__ import annotations

from pathlib import Path

from cap.auditors.base import Auditor, AuditPrediction, AuditTask
from cap.utils.logging import get_logger

logger = get_logger()


class DeepFaceAuditor(Auditor):
    name = "deepface"

    def __init__(self, detector_backend: str = "retinaface"):
        self.detector_backend = detector_backend

    def supported_tasks(self) -> list[AuditTask]:
        return [AuditTask.GENDER, AuditTask.AGE, AuditTask.EMOTION, AuditTask.RACE]

    def predict(self, image_path: str | Path, task: AuditTask) -> AuditPrediction:
        try:
            from deepface import DeepFace

            actions_map = {
                AuditTask.GENDER: "gender",
                AuditTask.AGE: "age",
                AuditTask.EMOTION: "emotion",
                AuditTask.RACE: "race",
            }
            action = actions_map.get(task)
            if action is None:
                return AuditPrediction(self.name, str(image_path), task, None, None, error="unsupported_task")
            result = DeepFace.analyze(
                img_path=str(image_path), actions=[action],
                detector_backend=self.detector_backend, enforce_detection=False, silent=True,
            )
            r = result[0] if isinstance(result, list) else result
            if task == AuditTask.GENDER:
                return AuditPrediction(self.name, str(image_path), task,
                                       r["dominant_gender"].lower(),
                                       r["gender"][r["dominant_gender"]] / 100.0, raw_response=r)
            if task == AuditTask.AGE:
                return AuditPrediction(self.name, str(image_path), task, float(r["age"]), None, raw_response=r)
            if task == AuditTask.EMOTION:
                return AuditPrediction(self.name, str(image_path), task,
                                       r["dominant_emotion"].lower(),
                                       r["emotion"][r["dominant_emotion"]] / 100.0, raw_response=r)
            if task == AuditTask.RACE:
                return AuditPrediction(self.name, str(image_path), task,
                                       r["dominant_race"].lower(),
                                       r["race"][r["dominant_race"]] / 100.0, raw_response=r)
        except Exception as e:
            logger.warning(f"DeepFace {task} failed for {image_path}: {e}")
            return AuditPrediction(self.name, str(image_path), task, None, None, error=str(e))
