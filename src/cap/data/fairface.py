"""FairFace seed identity loader with stratified sampling.

FairFace (Karkkainen & Joo, 2021) — 108K Flickr faces with race × gender × age
labels, designed for balanced bias measurement. We use it as our seed identity
pool because:

  - It is demographically balanced (no over/underrepresented groups)
  - It is real (not synthetic) — counterfactuals from real seeds carry stronger
    causal claims than text-only generation
  - It is openly licensed (CC-BY 4.0)

We pull from the HuggingFace mirror `HuggingFaceM4/FairFace` and stratify-sample
N identities balanced across the (race, gender, age) joint distribution.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cap.utils.logging import get_logger

logger = get_logger()

FAIRFACE_HF_REPO = "HuggingFaceM4/FairFace"
FAIRFACE_RACE_GROUPS = [
    "White", "Black", "East Asian", "Southeast Asian",
    "Indian", "Middle Eastern", "Latino_Hispanic",
]
FAIRFACE_GENDER_GROUPS = ["Male", "Female"]
FAIRFACE_AGE_GROUPS = [
    "0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70",
]


@dataclass
class SeedIdentity:
    id: str
    image_path: str
    race: str
    gender: str
    age: str
    source_index: int


class FairFaceLoader:
    """Load FairFace from HuggingFace and materialize selected images to disk."""

    def __init__(self, output_dir: str | Path, split: str = "1.25", subset: str = "train"):
        self.output_dir = Path(output_dir)
        self.split = split  # FairFace has "0.25" and "1.25" padding splits — 1.25 has more context
        self.subset = subset

    def load_index(self) -> pd.DataFrame:
        """Load the full FairFace metadata index without downloading images."""
        from datasets import load_dataset

        logger.info(f"Loading FairFace index ({self.split} / {self.subset}) from {FAIRFACE_HF_REPO}")
        ds = load_dataset(FAIRFACE_HF_REPO, self.split, split=self.subset)
        df = pd.DataFrame({
            "race": ds["race"],
            "gender": ds["gender"],
            "age": ds["age"],
            "service_test": ds.get("service_test", [False] * len(ds)),
            "source_index": list(range(len(ds))),
        })
        # FairFace stores labels as ints; convert to strings if needed
        if df["race"].dtype != object:
            df["race"] = df["race"].map(lambda i: FAIRFACE_RACE_GROUPS[i])
            df["gender"] = df["gender"].map(lambda i: FAIRFACE_GENDER_GROUPS[i])
            df["age"] = df["age"].map(lambda i: FAIRFACE_AGE_GROUPS[i])
        return df, ds

    def stratified_sample(
        self,
        df: pd.DataFrame,
        n: int,
        stratify_by: list[str],
        seed: int = 42,
    ) -> pd.DataFrame:
        """Stratified sample of size n, balanced across `stratify_by` groups.

        We compute per-cell quotas and sample from each cell. If a cell has fewer
        rows than its quota, we take all and redistribute the slack proportionally.
        """
        rng = random.Random(seed)
        cells = df.groupby(stratify_by, dropna=False).indices  # dict: cell_key -> [row indices]
        n_cells = len(cells)
        per_cell = max(1, n // n_cells)

        sampled_indices: list[int] = []
        deficit = 0
        for cell_key, idxs in cells.items():
            quota = per_cell
            idxs = list(idxs)
            rng.shuffle(idxs)
            if len(idxs) < quota:
                deficit += quota - len(idxs)
                sampled_indices.extend(idxs)
            else:
                sampled_indices.extend(idxs[:quota])

        # Distribute deficit across cells that still have slack
        if deficit > 0 or len(sampled_indices) < n:
            need = n - len(sampled_indices)
            remaining = []
            for cell_key, idxs in cells.items():
                taken = set(sampled_indices)
                extras = [i for i in idxs if i not in taken]
                rng.shuffle(extras)
                remaining.extend(extras)
            rng.shuffle(remaining)
            sampled_indices.extend(remaining[:need])

        return df.iloc[sampled_indices[:n]].copy().reset_index(drop=True)

    def materialize(self, sampled: pd.DataFrame, ds) -> list[SeedIdentity]:
        """Save the selected images from the HF dataset to local disk and build SeedIdentity records."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        seeds: list[SeedIdentity] = []
        for row_idx, row in sampled.iterrows():
            src_idx = int(row["source_index"])
            image = ds[src_idx]["image"]
            seed_id = f"ff{src_idx:06d}"
            image_path = self.output_dir / f"{seed_id}.jpg"
            if not image_path.exists():
                image.convert("RGB").save(image_path, format="JPEG", quality=95)
            seeds.append(
                SeedIdentity(
                    id=seed_id,
                    image_path=str(image_path),
                    race=row["race"],
                    gender=row["gender"],
                    age=row["age"],
                    source_index=src_idx,
                )
            )
        return seeds


def save_seed_manifest(seeds: list[SeedIdentity], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(s) for s in seeds], indent=2))


def load_seed_manifest(path: str | Path) -> list[SeedIdentity]:
    data = json.loads(Path(path).read_text())
    return [SeedIdentity(**d) for d in data]


def load_or_sample_seeds(
    output_dir: str | Path,
    n: int,
    stratify_by: list[str],
    seed: int = 42,
    manifest_name: str = "seed_manifest.json",
    split: str = "1.25",
) -> list[SeedIdentity]:
    """High-level entry: load from disk if manifest exists, else sample fresh."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / manifest_name
    if manifest_path.exists():
        seeds = load_seed_manifest(manifest_path)
        if len(seeds) >= n:
            logger.info(f"Loaded {len(seeds)} seeds from existing manifest {manifest_path}")
            return seeds[:n]
        logger.info(f"Existing manifest has {len(seeds)} seeds; need {n} — resampling")

    loader = FairFaceLoader(output_dir=output_dir, split=split)
    df, ds = loader.load_index()
    sampled = loader.stratified_sample(df, n=n, stratify_by=stratify_by, seed=seed)
    seeds = loader.materialize(sampled, ds)
    save_seed_manifest(seeds, manifest_path)
    logger.info(f"Sampled {len(seeds)} seeds, manifest at {manifest_path}")
    return seeds
