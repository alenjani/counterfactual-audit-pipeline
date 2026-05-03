"""CLI: check + adaptively rebalance the confirmed-seeds list across source demographics.

After `cap-prefilter-seeds` finishes, the surviving 200-ish IDs may not be
balanced across source-demographic cells (race × gender) because filters
drop non-uniformly (e.g., multi-face is more common in family photos →
correlates with race/age).

This CLI:
  1. Reads the existing confirmed_seeds.json
  2. Joins with FairFace metadata to get per-seed (race, gender, age)
  3. Computes per-cell counts on the configured stratification grid
  4. Reports the imbalance
  5. If --rebalance: iteratively load more candidates from underfilled cells,
     run cap-prefilter-seeds Steps 1+2 on those new candidates, append
     survivors, repeat up to --max-iterations.

Stratification grid: race × gender by default (14 cells × ~14 per cell for
n=200). Configurable via --grid {race_x_gender, race_x_gender_x_age}.

Usage:
  cap-balance-seeds --config configs/full.yaml \
                    --confirmed-seeds-file <path-to-confirmed_seeds.json> \
                    --output-dir <output-dir> \
                    [--rebalance --target-per-cell 14 --max-iterations 5 \
                     --extra-per-iteration 50]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import click
import pandas as pd

from cap.data import load_or_sample_seeds
from cap.utils import get_logger, load_config

logger = get_logger()


@click.command()
@click.option("--config", "config_path", required=True)
@click.option("--confirmed-seeds-file", required=True, type=click.Path(),
              help="Existing confirmed_seeds.json from cap-prefilter-seeds")
@click.option("--output-dir", required=True, type=click.Path(),
              help="Where to write the (possibly rebalanced) seeds + report")
@click.option("--grid", type=click.Choice(["race_x_gender", "race_x_gender_x_age"]),
              default="race_x_gender", help="Stratification grid for balance check")
@click.option("--target-per-cell", default=14, type=int,
              help="Min IDs per stratification cell (only relevant with --rebalance)")
@click.option("--rebalance/--no-rebalance", default=False,
              help="If imbalanced, iteratively oversample underfilled cells")
@click.option("--max-iterations", default=5, type=int)
@click.option("--extra-per-iteration", default=50, type=int,
              help="N extra FairFace candidates to load per rebalance iteration")
@click.option("--cache-dir", default="/local_disk0/hf_cache")
@click.option("--pulid-src", default="/Volumes/ds_work/alenj00/cap_cache/pulid_src")
@click.option("--hf-token", default=None)
def main(
    config_path: str,
    confirmed_seeds_file: str,
    output_dir: str,
    grid: str,
    target_per_cell: int,
    rebalance: bool,
    max_iterations: int,
    extra_per_iteration: int,
    cache_dir: str,
    pulid_src: str,
    hf_token: str | None,
) -> None:
    cfg = load_config(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    confirmed_path = Path(confirmed_seeds_file)
    if not confirmed_path.exists():
        raise click.ClickException(f"Missing: {confirmed_path}")
    confirmed_ids = set(json.loads(confirmed_path.read_text()))
    logger.info(f"Loaded {len(confirmed_ids)} confirmed IDs from {confirmed_path}")

    # Load enough FairFace candidates so the metadata join covers everything
    # (we use the same load function here as cap-prefilter; deterministic).
    pool_n = max(int(cfg["seed_identities"].get("count", 250)), 250)
    pool = load_or_sample_seeds(
        output_dir=cfg["paths.seed_dataset"],
        n=pool_n,
        stratify_by=list(cfg["seed_identities"].get("stratify_by", ["race", "gender", "age"])),
        seed=cfg.get("seed", 42),
    )
    pool_by_id = {s.id: s for s in pool}

    rows = [
        {"id": sid, "race": pool_by_id[sid].race, "gender": pool_by_id[sid].gender,
         "age": pool_by_id[sid].age}
        for sid in confirmed_ids if sid in pool_by_id
    ]
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise click.ClickException("No confirmed IDs found in FairFace pool — check seed_dataset path.")

    if grid == "race_x_gender":
        cell_cols = ["race", "gender"]
    else:
        cell_cols = ["race", "gender", "age"]

    counts = df.groupby(cell_cols).size().rename("n").reset_index()
    counts.to_csv(out_dir / "balance_report_initial.csv", index=False)

    underfilled = counts[counts["n"] < target_per_cell]
    logger.info(f"\nInitial balance ({grid}, n={len(df)}, target={target_per_cell}/cell):")
    logger.info("\n" + counts.to_string(index=False))
    logger.info(
        f"\nUnderfilled cells: {len(underfilled)}/{len(counts)}; "
        f"min count: {counts['n'].min()}, max count: {counts['n'].max()}"
    )

    if not rebalance:
        # Just write back the original list (so callers can chain on a single output path).
        (out_dir / "confirmed_seeds_balanced.json").write_text(json.dumps(sorted(confirmed_ids)))
        logger.info(f"--no-rebalance: wrote {len(confirmed_ids)} IDs unchanged → "
                    f"{out_dir / 'confirmed_seeds_balanced.json'}")
        return

    # ---- Adaptive oversampling loop ------------------------------------------
    iteration = 0
    while iteration < max_iterations and len(underfilled) > 0:
        iteration += 1
        logger.info(f"\n=== Rebalance iteration {iteration}/{max_iterations} ===")
        logger.info(f"Loading {extra_per_iteration} additional FairFace candidates...")

        # Load N more candidates stratified the same way; offset the seed so we
        # get DIFFERENT IDs than the prior pool.
        more_pool = load_or_sample_seeds(
            output_dir=cfg["paths.seed_dataset"],
            n=pool_n + extra_per_iteration * iteration,
            stratify_by=list(cfg["seed_identities"].get("stratify_by", ["race", "gender", "age"])),
            seed=cfg.get("seed", 42) + iteration,  # different seed → different sample
        )
        new_ids = [s.id for s in more_pool if s.id not in confirmed_ids and s.id not in pool_by_id]
        for s in more_pool:
            pool_by_id.setdefault(s.id, s)
        if not new_ids:
            logger.warning("No new candidate IDs — exhausted FairFace pool variations. Stopping.")
            break
        logger.info(f"Got {len(new_ids)} new candidates to filter.")

        # Run prefilter on just these new candidates. We need the cap-prefilter-seeds
        # CLI to accept a "ids only" mode. For simplicity here, we shell out to the
        # existing prefilter with a temporary config that includes only these IDs.
        # This avoids depending on internal Python imports + lets us reuse the
        # tested prefilter pipeline.
        new_confirmed = _prefilter_subset(
            new_ids, pool_by_id, cfg, cache_dir, pulid_src, hf_token, out_dir, iteration,
        )
        logger.info(f"Iteration {iteration}: {len(new_confirmed)} of {len(new_ids)} survived prefilter")
        confirmed_ids.update(new_confirmed)

        # Recompute balance
        rows = [
            {"id": sid, "race": pool_by_id[sid].race, "gender": pool_by_id[sid].gender,
             "age": pool_by_id[sid].age}
            for sid in confirmed_ids if sid in pool_by_id
        ]
        df = pd.DataFrame(rows)
        counts = df.groupby(cell_cols).size().rename("n").reset_index()
        underfilled = counts[counts["n"] < target_per_cell]
        logger.info(
            f"After iter {iteration}: total={len(df)}, "
            f"underfilled cells={len(underfilled)}/{len(counts)}, "
            f"min count={counts['n'].min()}"
        )

    # ---- Final outputs -------------------------------------------------------
    counts.to_csv(out_dir / "balance_report_final.csv", index=False)
    out_path = out_dir / "confirmed_seeds_balanced.json"
    out_path.write_text(json.dumps(sorted(confirmed_ids)))
    logger.info(f"\nFinal balanced seed list ({len(confirmed_ids)} IDs) → {out_path}")
    logger.info("\nFinal balance:")
    logger.info("\n" + counts.to_string(index=False))


def _prefilter_subset(
    candidate_ids: list[str],
    pool_by_id: dict,
    cfg,
    cache_dir: str,
    pulid_src: str,
    hf_token: str | None,
    out_dir: Path,
    iteration: int,
) -> set[str]:
    """Run prefilter Steps 1+2 on a subset of candidates. Returns confirmed IDs.

    Implementation note: this duplicates a slice of cap-prefilter-seeds rather
    than shelling out, because we need to operate on an explicit ID list — the
    existing CLI expects to load via stratified sampling, not from a pre-built
    list. A small refactor of prefilter_seeds.main to accept an `--ids-list`
    flag would let this share more code; for now, inline.
    """
    from cap.cli.prefilter_seeds import _step1_face_filter, _step2_round_trip

    # Step 1: face filter
    candidate_seeds = [pool_by_id[i] for i in candidate_ids if i in pool_by_id]
    face_passed_ids = _step1_face_filter(candidate_seeds, cache_dir, hf_token)
    survivors = [s for s in candidate_seeds if s.id in face_passed_ids]
    logger.info(f"  step 1 (face filter): {len(survivors)}/{len(candidate_seeds)} passed")
    if not survivors:
        return set()

    # Step 2: round-trip
    confirmed_ids, scores = _step2_round_trip(
        survivors, cfg, cache_dir, hf_token, pulid_src, threshold=0.5,
        out_dir=out_dir / f"rebalance_iter_{iteration}",
    )
    return set(confirmed_ids)


if __name__ == "__main__":
    main()
