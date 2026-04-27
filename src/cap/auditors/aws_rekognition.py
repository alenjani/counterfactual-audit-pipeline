"""AWS Rekognition auditor.

Tasks supported: gender, age (range), emotion, face detection.
Cost: ~$0.001 per image (DetectFaces).
Auth: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars or default profile.
"""
from __future__ import annotations

from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from cap.auditors.base import Auditor, AuditPrediction, AuditTask
from cap.utils.logging import get_logger

logger = get_logger()


class AWSRekognitionAuditor(Auditor):
    name = "aws_rekognition"

    def __init__(self, region_name: str = "us-east-1"):
        import boto3

        self.client = boto3.client("rekognition", region_name=region_name)

    def supported_tasks(self) -> list[AuditTask]:
        return [AuditTask.GENDER, AuditTask.AGE, AuditTask.EMOTION, AuditTask.FACE_DETECTION]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    def _detect_faces(self, image_path: str | Path) -> dict:
        with open(image_path, "rb") as f:
            return self.client.detect_faces(Image={"Bytes": f.read()}, Attributes=["ALL"])

    def predict(self, image_path: str | Path, task: AuditTask) -> AuditPrediction:
        try:
            resp = self._detect_faces(image_path)
            details = resp.get("FaceDetails", [])
            if not details:
                return AuditPrediction(
                    auditor=self.name, image_path=str(image_path), task=task,
                    prediction=None, confidence=None, error="no_face_detected",
                )
            face = details[0]
            if task == AuditTask.GENDER:
                return AuditPrediction(
                    self.name, str(image_path), task,
                    face["Gender"]["Value"].lower(), face["Gender"]["Confidence"] / 100.0, raw_response=face,
                )
            if task == AuditTask.AGE:
                low, high = face["AgeRange"]["Low"], face["AgeRange"]["High"]
                return AuditPrediction(
                    self.name, str(image_path), task,
                    (low + high) / 2.0, None, raw_response={"low": low, "high": high},
                )
            if task == AuditTask.EMOTION:
                top = max(face["Emotions"], key=lambda e: e["Confidence"])
                return AuditPrediction(
                    self.name, str(image_path), task,
                    top["Type"].lower(), top["Confidence"] / 100.0, raw_response={"emotions": face["Emotions"]},
                )
            if task == AuditTask.FACE_DETECTION:
                return AuditPrediction(
                    self.name, str(image_path), task,
                    True, face.get("Confidence", 0) / 100.0, raw_response={"bbox": face["BoundingBox"]},
                )
            return AuditPrediction(
                self.name, str(image_path), task, None, None, error="unsupported_task",
            )
        except Exception as e:
            logger.warning(f"Rekognition {task} failed for {image_path}: {e}")
            return AuditPrediction(
                self.name, str(image_path), task, None, None, error=str(e),
            )
