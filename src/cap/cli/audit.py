"""CLI: run all configured auditors over generated images and store predictions."""
from __future__ import annotations

from dataclasses import asdict
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
def main(config_path: str, limit: int | None) -> None:
    cfg = load_config(config_path)
    audit_dir = Path(cfg["paths.audit_dir"])
    audit_dir.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest.create(
        run_id=cfg.get("run_id", "unknown") + "_audit",
        config_path=config_path,
        config_resolved=cfg.to_dict(),
    )

    generated_dir = Path(cfg["paths.generated_dir"])
    image_index = pd.read_parquet(generated_dir / "manifest.parquet")
    if limit is not None:
        image_index = image_index.head(limit)
    logger.info(f"Auditing {len(image_index)} images")

    auditors = build_auditors(cfg["auditors"])
    requested_tasks = [AuditTask(t) for t in cfg["tasks"]]

    rows = []
    for _, row in tqdm(image_index.iterrows(), total=len(image_index)):
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

    out_path = audit_dir / "predictions.parquet"
    pd.DataFrame(rows).to_parquet(out_path)
    manifest.finish()
    manifest.write(audit_dir / "run_manifest.json")
    logger.info(f"Wrote {len(rows)} audit predictions to {out_path}")


if __name__ == "__main__":
    main()
