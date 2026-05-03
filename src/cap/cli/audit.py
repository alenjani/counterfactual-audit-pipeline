"""CLI: run all configured auditors over images and store predictions.

Two modes (Phase 11b — bias-attribution control):

  --mode synthetic       Audit the GENERATED counterfactuals (default).
                         Uses generated_dir/manifest.parquet as the image
                         index. Image rows include axis_skin_tone,
                         axis_gender, axis_age, etc. — i.e., what we
                         ASKED the model to render, used as the asserted
                         ground truth in downstream analyze.py.

  --mode real_baseline   Audit the ORIGINAL FairFace seed images. The
                         "axis_*" values come from FairFace's gold-standard
                         metadata (not what we asked for, but what the
                         person actually is). Output: a parallel parquet
                         used to disambiguate auditor bias from generator
                         artifact: if auditors are wrong on real photos
                         the same way they're wrong on synthetic ones,
                         the bias is the auditor's. If they're right on
                         real and wrong on synthetic, generator introduced
                         a confound.

  --mode both            Run both passes; writes predictions.parquet
                         (synthetic) AND baseline_predictions.parquet
                         (real). Cheaper to run both at once because
                         auditor models stay loaded.
"""
from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from cap.auditors import AuditTask, build_auditors
from cap.utils import RunManifest, get_logger, load_config

logger = get_logger()


@click.command()
@click.option("--config", "config_path", required=True)
@click.option("--limit", default=None, type=int)
@click.option(
    "--mode",
    type=click.Choice(["synthetic", "real_baseline", "both"]),
    default="synthetic",
    help="Which image set to audit. See module docstring for the bias-attribution rationale.",
)
def main(config_path: str, limit: int | None, mode: str) -> None:
    cfg = load_config(config_path)
    audit_dir = Path(cfg["paths.audit_dir"])
    audit_dir.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest.create(
        run_id=cfg.get("run_id", "unknown") + f"_audit_{mode}",
        config_path=config_path,
        config_resolved=cfg.to_dict(),
    )

    auditors = build_auditors(cfg["auditors"])
    requested_tasks = [AuditTask(t) for t in cfg["tasks"]]

    if mode in ("synthetic", "both"):
        synth_index = _load_synthetic_index(cfg, limit)
        _audit_and_write(
            synth_index, auditors, requested_tasks,
            audit_dir / "predictions.parquet",
            label="synthetic",
        )

    if mode in ("real_baseline", "both"):
        real_index = _load_real_baseline_index(cfg, limit)
        _audit_and_write(
            real_index, auditors, requested_tasks,
            audit_dir / "baseline_predictions.parquet",
            label="real_baseline",
        )

    manifest.finish()
    manifest.write(audit_dir / f"run_manifest_{mode}.json")


def _load_synthetic_index(cfg, limit: int | None) -> pd.DataFrame:
    """Load the generated counterfactuals' manifest."""
    generated_dir = Path(cfg["paths.generated_dir"])
    df = pd.read_parquet(generated_dir / "manifest.parquet")
    if limit is not None:
        df = df.head(limit)
    return df


def _load_real_baseline_index(cfg, limit: int | None) -> pd.DataFrame:
    """Build an index of the ORIGINAL FairFace seed images.

    Each row carries the per-seed FairFace gold-standard demographic labels
    in the same `axis_*` columns the analyze.py code expects, so the same
    statistical pipeline runs unchanged on the baseline data.

    For a real seed, `axis_*` represents WHAT THE PERSON IS (FairFace
    metadata), not WHAT WE ASKED FOR (since this isn't a generated
    counterfactual). axis_gender comes from FairFace's gender label;
    axis_skin_tone is approximated from FairFace's race label.
    """
    from cap.data import load_or_sample_seeds

    si_cfg = cfg["seed_identities"]
    if si_cfg.get("source") != "fairface":
        raise click.ClickException(
            f"--mode real_baseline only supports FairFace seeds. Got: {si_cfg.get('source')}"
        )

    # Honor seed_identities.ids list (e.g., from confirmed_seeds.json) when
    # present so the baseline matches the seeds actually used in generation.
    full = load_or_sample_seeds(
        output_dir=cfg["paths.seed_dataset"],
        n=int(si_cfg.get("count", 200)),
        stratify_by=list(si_cfg.get("stratify_by", ["race", "gender", "age"])),
        seed=cfg.get("seed", 42),
    )
    by_id = {s.id: s for s in full}
    explicit = si_cfg.get("ids")
    seeds = [by_id[i] for i in explicit if i in by_id] if explicit else full

    # Map FairFace race → Fitzpatrick skin tone (rough). Used for analysis-side
    # joinability with synthetic axis_skin_tone (1-6). For the paper, document
    # this as an approximation; the more rigorous version uses Skin Color
    # Estimator on the seed image.
    _RACE_TO_FITZPATRICK = {
        "East Asian": 3, "Indian": 4, "Black": 6, "White": 1,
        "Middle Eastern": 3, "Latino_Hispanic": 3, "Southeast Asian": 4,
    }

    rows = []
    for s in seeds:
        rows.append({
            "seed_identity_id": s.id,
            "counterfactual_id": s.id,  # baseline = the seed itself
            "image_path": s.image_path,
            "prompt_used": "(real seed image — no prompt)",
            "axis_skin_tone": _RACE_TO_FITZPATRICK.get(s.race, 3),
            "axis_gender": (s.gender or "").lower(),
            "axis_age": s.age,
            "axis_race_fairface": s.race,
            "metadata": {"is_baseline": True, "skipped": False, "reason": "real_seed_image"},
        })
    df = pd.DataFrame(rows)
    if limit is not None:
        df = df.head(limit)
    logger.info(f"Loaded {len(df)} real baseline seeds for audit")
    return df


def _audit_and_write(
    image_index: pd.DataFrame,
    auditors: list,
    requested_tasks: list,
    out_path: Path,
    label: str,
) -> None:
    logger.info(f"[{label}] auditing {len(image_index)} images × {len(auditors)} auditors × {len(requested_tasks)} tasks")
    rows = []
    for _, row in tqdm(image_index.iterrows(), total=len(image_index), desc=label):
        for auditor in auditors:
            for task in requested_tasks:
                if task not in auditor.supported_tasks():
                    continue
                pred = auditor.predict(row["image_path"], task)
                rows.append({
                    **{c: row[c] for c in image_index.columns},
                    "auditor": auditor.name,
                    "task": task.value,
                    "prediction": pred.prediction,
                    "confidence": pred.confidence,
                    "error": pred.error,
                })

    df = pd.DataFrame(rows)
    df["prediction"] = df["prediction"].astype(str)
    df.to_parquet(out_path)
    logger.info(f"[{label}] wrote {len(rows)} audit predictions → {out_path}")


if __name__ == "__main__":
    main()
