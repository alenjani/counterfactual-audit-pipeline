"""ArcFace-based identity preservation validator.

Computes cosine similarity between ArcFace embeddings of seed and counterfactual.
Identity is considered preserved if similarity >= threshold (default 0.5 — stricter
than face-recognition default of 0.4).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cap.utils.logging import get_logger

logger = get_logger()


@dataclass
class IdentityScore:
    seed_image: str
    counterfactual_image: str
    cosine_similarity: float
    is_preserved: bool


class ArcFaceIdentityValidator:
    def __init__(
        self,
        model_name: str = "antelopev2",
        threshold: float = 0.5,
        device: str = "cuda",
        root: str | None = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        # Where InsightFace looks for / downloads the model pack. If not given,
        # falls back to ~/.insightface (often unwritable on Databricks workers).
        self.root = root
        self._app = None

    def _lazy_load(self) -> None:
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        # Pre-stage antelopev2 from a HF mirror if needed (auto-download from
        # the InsightFace GitHub release URL is blocked on some clusters).
        if self.root and self.model_name == "antelopev2":
            from cap.generator.flux_pulid import _ensure_antelopev2

            _ensure_antelopev2(self.root)

        kwargs = {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
        if self.root:
            kwargs["root"] = self.root
        self._app = FaceAnalysis(name=self.model_name, **kwargs)
        self._app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))

    def embed(self, image_path: str | Path) -> np.ndarray | None:
        self._lazy_load()
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            return None
        faces = self._app.get(img)
        if not faces:
            return None
        return faces[0].normed_embedding

    def score_pair(self, seed_path: str | Path, cf_path: str | Path) -> IdentityScore:
        seed_emb = self.embed(seed_path)
        cf_emb = self.embed(cf_path)
        if seed_emb is None or cf_emb is None:
            sim = float("nan")
        else:
            sim = float(np.dot(seed_emb, cf_emb))
        return IdentityScore(
            seed_image=str(seed_path),
            counterfactual_image=str(cf_path),
            cosine_similarity=sim,
            is_preserved=sim >= self.threshold,
        )
