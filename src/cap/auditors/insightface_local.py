"""InsightFace local auditor — strong face analysis (gender, age, detection)."""
from __future__ import annotations

from pathlib import Path

from cap.auditors.base import Auditor, AuditPrediction, AuditTask
from cap.utils.logging import get_logger

logger = get_logger()


class InsightFaceAuditor(Auditor):
    name = "insightface"

    def __init__(self, model_name: str = "buffalo_l", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._app = None

    def _lazy_load(self) -> None:
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(name=self.model_name,
                                 providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self._app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))

    def supported_tasks(self) -> list[AuditTask]:
        return [AuditTask.GENDER, AuditTask.AGE, AuditTask.FACE_DETECTION]

    def predict(self, image_path: str | Path, task: AuditTask) -> AuditPrediction:
        try:
            import cv2

            self._lazy_load()
            img = cv2.imread(str(image_path))
            faces = self._app.get(img)
            if not faces:
                return AuditPrediction(self.name, str(image_path), task, None, None, error="no_face_detected")
            face = faces[0]
            if task == AuditTask.GENDER:
                # InsightFace: 0 = female, 1 = male
                return AuditPrediction(self.name, str(image_path), task,
                                       "male" if face.gender == 1 else "female", None,
                                       raw_response={"gender": int(face.gender)})
            if task == AuditTask.AGE:
                return AuditPrediction(self.name, str(image_path), task, float(face.age), None,
                                       raw_response={"age": float(face.age)})
            if task == AuditTask.FACE_DETECTION:
                return AuditPrediction(self.name, str(image_path), task, True, float(face.det_score),
                                       raw_response={"bbox": face.bbox.tolist()})
        except Exception as e:
            logger.warning(f"InsightFace {task} failed for {image_path}: {e}")
            return AuditPrediction(self.name, str(image_path), task, None, None, error=str(e))
