"""Abstract counterfactual generator interface.

A generator takes a seed identity (image or prompt) and produces N counterfactual
variations along specified demographic axes (skin tone, gender, age, etc.) while
preserving identity and structural attributes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GenerationRequest:
    seed_identity_id: str
    seed_image_path: str | None  # path to FairFace image, or None for text-only
    seed_prompt: str | None
    counterfactual_axes: dict[str, list[str | int | float]]  # {"skin_tone": [1..6], "gender": ["male", "female"], ...}
    fixed_attributes: dict[str, str] = field(default_factory=dict)  # e.g. {"pose": "frontal", "expression": "neutral"}
    seed: int = 0
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    width: int = 1024
    height: int = 1024


@dataclass
class GenerationResult:
    seed_identity_id: str
    counterfactual_id: str  # e.g. "id042_skin3_female_age30"
    image_path: str
    prompt_used: str
    axis_values: dict[str, Any]  # which axis values this image represents
    metadata: dict[str, Any] = field(default_factory=dict)  # generation time, model versions, etc.


class CounterfactualGenerator(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest, output_dir: str | Path) -> list[GenerationResult]:
        """Generate the full factorial of counterfactuals for the given request."""

    @abstractmethod
    def model_versions(self) -> dict[str, str]:
        """Return version/hash of every loaded model component for reproducibility."""
