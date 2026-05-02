"""DeepFace local auditor — open-source, multi-backend face analysis."""
from __future__ import annotations

import os
from pathlib import Path

from cap.auditors.base import Auditor, AuditPrediction, AuditTask
from cap.utils.logging import get_logger

logger = get_logger()


def _force_tf_cpu() -> None:
    """Pin TensorFlow to CPU and Keras 2 for DeepFace.

    Two issues to handle:
    1. GPU contention: PyTorch holds the L4 CUDA context; TF + PyTorch
       sharing a context fails at the graph executor (`FusedBatchNormV3`).
       Force TF to CPU.
    2. Keras 3 incompatibility: DeepFace's models use raw TF ops on
       KerasTensors, which Keras 3 (default in TF 2.16+) forbids with
       "A KerasTensor cannot be used as input to a TensorFlow function".
       TF_USE_LEGACY_KERAS=1 routes TF to Keras 2 (legacy), which DeepFace
       supports. Requires the tf-keras package; we try and tolerate.

    Both env vars must be set BEFORE the first TF import. We also call
    set_visible_devices as a defensive belt-and-suspenders.
    """
    # Force CPU (idempotent — overwrite even if already set).
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


class DeepFaceAuditor(Auditor):
    name = "deepface"

    def __init__(self, detector_backend: str = "retinaface"):
        self.detector_backend = detector_backend
        _force_tf_cpu()

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
