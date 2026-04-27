"""Structural conditioning preprocessors for ControlNet.

Extracts pose / depth / canny maps from a seed face image. The control image
locks structural attributes (head angle, framing, lighting orientation) across
all counterfactuals of a seed identity, so demographics become the only varying
factor.
"""
from __future__ import annotations

from typing import Callable, Protocol

from PIL import Image


class ControlProcessor(Protocol):
    def __call__(self, image: Image.Image) -> Image.Image: ...


def build_control_processor(mode: str, device: str = "cuda") -> ControlProcessor:
    """Build a callable that turns a seed image into a ControlNet conditioning image.

    Modes:
      - "pose":    OpenPose face/body keypoints (controlnet_aux.OpenposeDetector)
      - "depth":   Midas depth (controlnet_aux.MidasDetector)
      - "canny":   Canny edges (cv2)
      - "identity": passthrough (no transformation)
    """
    if mode == "pose":
        from controlnet_aux import OpenposeDetector

        det = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

        def _run(img: Image.Image) -> Image.Image:
            return det(img, hand_and_face=True)

        return _run

    if mode == "depth":
        from controlnet_aux import MidasDetector

        det = MidasDetector.from_pretrained("lllyasviel/Annotators")

        def _run(img: Image.Image) -> Image.Image:
            return det(img)

        return _run

    if mode == "canny":
        import cv2
        import numpy as np

        def _run(img: Image.Image) -> Image.Image:
            arr = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return Image.fromarray(edges).convert("RGB")

        return _run

    if mode == "identity":
        return lambda img: img

    raise ValueError(f"Unknown control mode: {mode}. Use pose|depth|canny|identity.")
