"""CLI: surgical rebalance — fill specific underfilled (race × gender) cells.

Replaces the wasteful uniform-stratified iteration in cap-balance-seeds with a
targeted approach that ONLY loads candidates from the specific cells that are
short of the per-cell target.

Algorithm:
  1. Read existing confirmed_seeds.json
  2. Look up race/gender for each via FairFace's full metadata index
  3. Compute per-cell deficit
  4. For each underfilled cell, sample N more candidates from FairFace's full
     index (NOT a stratified sample of multiple cells — just IDs in this exact
     cell), excluding any already-confirmed IDs and any IDs we've already failed
  5. Run prefilter Steps 1+2 on those targeted candidates ONLY
  6. Append survivors; write confirmed_seeds_balanced.json

Usage:
  cap-targeted-rebalance \
    --config configs/full.yaml \
    --confirmed-seeds-file <path> \
    --output-dir <path> \
    --target-per-cell 14 \
    --max-attempts-per-cell 30   # how many candidates to try per cell before giving up
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

from cap.utils import get_logger, load_config

logger = get_logger()


@click.command()
@click.option("--config", "config_path", required=True)
@click.option("--confirmed-seeds-file", required=True, type=click.Path(),
              help="Existing confirmed_seeds.json from prior steps")
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--target-per-cell", default=14, type=int)
@click.option("--max-attempts-per-cell", default=30, type=int,
              help="Max candidates to try per underfilled cell before moving on")
@click.option("--cache-dir", default="/local_disk0/hf_cache")
@click.option("--pulid-src", default="/Volumes/ds_work/alenj00/cap_cache/pulid_src")
@click.option("--hf-token", default=None)
def main(
    config_path: str,
    confirmed_seeds_file: str,
    output_dir: str,
    target_per_cell: int,
    max_attempts_per_cell: int,
    cache_dir: str,
    pulid_src: str,
    hf_token: str | None,
) -> None:
    cfg = load_config(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: read existing confirmed list -------------------------------
    confirmed_path = Path(confirmed_seeds_file)
    confirmed_ids = set(json.loads(confirmed_path.read_text()))
    logger.info(f"Existing confirmed: {len(confirmed_ids)} IDs")

    # --- Step 2: full FairFace metadata via direct lookup -------------------
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from cap.data.fairface import FairFaceLoader
    ldr = FairFaceLoader(output_dir=cfg["paths.seed_dataset"])
    df_full, _hf_ds = ldr.load_index()
    df_full["id"] = df_full["source_index"].apply(lambda i: f"ff{int(i):06d}")
    logger.info(f"FairFace metadata loaded: {len(df_full)} rows")

    confirmed_meta = df_full[df_full["id"].isin(confirmed_ids)].copy()
    counts = confirmed_meta.groupby(["race", "gender"]).size().rename("n").reset_index()
    logger.info("Initial balance:")
    logger.info("\n" + counts.to_string(index=False))

    underfilled = counts[counts["n"] < target_per_cell].copy()
    logger.info(f"\nUnderfilled cells: {len(underfilled)} / {len(counts)}")
    logger.info("\n" + underfilled.to_string(index=False))

    # --- Step 3: identify deficit per cell -----------------------------------
    deficits = {
        (row["race"], row["gender"]): target_per_cell - int(row["n"])
        for _, row in underfilled.iterrows()
    }
    total_deficit = sum(deficits.values())
    logger.info(f"\nTotal deficit: {total_deficit} IDs across {len(deficits)} cells")

    if total_deficit == 0:
        logger.info("No deficit — writing confirmed_seeds_balanced.json unchanged.")
        (out_dir / "confirmed_seeds_balanced.json").write_text(json.dumps(sorted(confirmed_ids)))
        return

    # --- Step 4: sample candidates from underfilled cells only -----------
    # For each underfilled cell, take up to max_attempts_per_cell candidates
    # from FairFace, excluding those already confirmed.
    candidates_to_try: list[dict] = []
    for (race, gender), deficit in deficits.items():
        pool = df_full[
            (df_full["race"] == race)
            & (df_full["gender"] == gender)
            & (~df_full["id"].isin(confirmed_ids))
        ]
        # deterministic shuffle (so reruns hit the same candidates)
        rng = np.random.default_rng(42 + hash((race, gender)) % 1000)
        idx = rng.permutation(len(pool))[:max_attempts_per_cell]
        cand = pool.iloc[idx].copy()
        logger.info(
            f"  ({race}, {gender}): deficit={deficit}, will try "
            f"{len(cand)} candidates"
        )
        for _, r in cand.iterrows():
            candidates_to_try.append({
                "id": r["id"],
                "race": r["race"], "gender": r["gender"], "age": r["age"],
                "image_path": str(Path(cfg["paths.seed_dataset"]) / f"{r['id']}.jpg"),
                "deficit_remaining": deficit,
            })
    logger.info(f"\nTotal candidates to try: {len(candidates_to_try)}")

    # --- Step 5: materialize images for new candidates ---------------------
    # FairFace's HF dataset has the actual image bytes. We need to save them
    # to the seed_dataset dir under the conventional ff{src_idx:06d}.jpg name
    # so the prefilter face_filter and round_trip can read them.
    materialized = _materialize_images(
        [c["id"] for c in candidates_to_try],
        df_full, _hf_ds, cfg["paths.seed_dataset"]
    )
    logger.info(f"Materialized {materialized} new seed images.")

    # --- Step 6: face filter (Step 1) on these candidates ----------------
    from cap.cli.prefilter_seeds import _step1_face_filter, _step2_round_trip
    from cap.data.fairface import SeedIdentity

    cand_seeds = [
        SeedIdentity(
            id=c["id"], image_path=c["image_path"],
            race=c["race"], gender=c["gender"], age=c["age"], source_index=int(c["id"][2:]),
        )
        for c in candidates_to_try
    ]
    face_passed = _step1_face_filter(cand_seeds, cache_dir, hf_token)
    survivors = [s for s in cand_seeds if s.id in face_passed]
    logger.info(f"Step 1 face filter: {len(survivors)}/{len(cand_seeds)} passed")

    # --- Step 7: round-trip (Step 2) on survivors ------------------------
    new_confirmed, scores = _step2_round_trip(
        survivors, cfg, cache_dir, hf_token, pulid_src,
        threshold=0.5, out_dir=out_dir / "targeted_round_trip",
    )
    logger.info(f"Step 2 round-trip: {len(new_confirmed)}/{len(survivors)} passed")

    # --- Step 8: build per-cell new-confirmed; greedy fill deficit ----------
    # For each cell, take up to its deficit from the new survivors. If a cell
    # doesn't have enough new survivors, log it but proceed.
    by_cell_new: dict[tuple, list[str]] = {k: [] for k in deficits}
    for sid in new_confirmed:
        meta = df_full[df_full["id"] == sid].iloc[0]
        key = (meta["race"], meta["gender"])
        if key in by_cell_new:
            by_cell_new[key].append(sid)

    final_added = set()
    for cell, deficit in deficits.items():
        chosen = by_cell_new[cell][:deficit]
        final_added.update(chosen)
        if len(chosen) < deficit:
            logger.warning(
                f"  ({cell[0]}, {cell[1]}): wanted {deficit}, got {len(chosen)} "
                f"— pool may be exhausted at this filter rigor"
            )

    final_confirmed = confirmed_ids | final_added
    logger.info(
        f"\nFinal: {len(confirmed_ids)} existing + {len(final_added)} new "
        f"= {len(final_confirmed)} total confirmed"
    )

    # Final balance report
    final_meta = df_full[df_full["id"].isin(final_confirmed)]
    final_counts = final_meta.groupby(["race", "gender"]).size().rename("n").reset_index()
    final_counts.to_csv(out_dir / "balance_report_targeted.csv", index=False)
    logger.info("\nFinal balance:")
    logger.info("\n" + final_counts.to_string(index=False))

    out_path = out_dir / "confirmed_seeds_balanced.json"
    out_path.write_text(json.dumps(sorted(final_confirmed)))
    logger.info(f"\nWrote {out_path}")

    pd.DataFrame(scores).to_parquet(out_dir / "targeted_round_trip_scores.parquet")


def _materialize_images(ids: list[str], df_full, hf_ds, seed_dataset_dir: str) -> int:
    """Save FairFace images to disk for any IDs we haven't materialized yet."""
    from PIL import Image as PILImage
    seed_dir = Path(seed_dataset_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    n_materialized = 0
    for sid in ids:
        out = seed_dir / f"{sid}.jpg"
        if out.exists():
            continue
        src_idx = int(sid[2:])
        img = hf_ds[src_idx]["image"]
        img.convert("RGB").save(out, format="JPEG", quality=95)
        n_materialized += 1
    return n_materialized


if __name__ == "__main__":
    main()
