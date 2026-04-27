"""FID-based image quality validator (Fréchet Inception Distance)."""
from __future__ import annotations

from pathlib import Path


class FIDValidator:
    """Compute FID between a set of generated images and a reference distribution.

    Reference: torch-fidelity or cleanfid — standard implementations.
    Status: skeleton; implementation deferred until generation works.
    """

    def __init__(self, reference_set_path: str | Path | None = None):
        self.reference_set_path = reference_set_path

    def score(self, generated_dir: str | Path) -> float:
        # TODO: use cleanfid.compute_fid(...) once data exists
        raise NotImplementedError("FID scoring pending — implement after first generation run")
