"""Attribute classifier validator.

Validates that the *intended* demographic shift actually occurred in the image
(e.g., a "skin_tone=5" counterfactual should be classified as Fitzpatrick V by
an independent classifier). Detects generator failures.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AttributeReading:
    image_path: str
    predicted_attributes: dict[str, str | int]
    confidences: dict[str, float]


class AttributeClassifierValidator:
    """Wraps DeepFace / FairFace classifier for independent attribute readout."""

    def __init__(self, backend: str = "deepface"):
        self.backend = backend

    def classify(self, image_path: str | Path) -> AttributeReading:
        # TODO: implement using DeepFace.analyze(...) for race/gender/age/skin
        raise NotImplementedError("Attribute classification pending — implement after generation")
