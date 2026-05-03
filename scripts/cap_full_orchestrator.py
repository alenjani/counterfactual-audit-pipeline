#!/usr/bin/env python3
"""CAP full-instrument orchestrator: submit → poll → resubmit → HF-backup.

Drives the 21-day, 36,000-image full run as a series of Databricks job
submissions, each chunked under the 18-hr per-job timeout. The generator's
skip-if-exists logic handles per-image resumption; this script handles the
multi-day cluster orchestration around it.

Stages (cumulative; see plans/003-full-instrument-plan.md):
  A: 200 IDs × 12 (skin × gender) × seed 42, age=anchor       → 2,400 images   (~36 hr)
  B: + remaining 4 ages, seed 42                                → 12,000 cumul (~7 days)
  C: + seeds 137 + 2718                                         → 36,000 cumul (~21 days)

After each stage completes, snapshot to HuggingFace Datasets (private):
  alenjani/cap-counterfactuals-stage-a, -stage-b, -stage-c

Usage:
  python scripts/cap_full_orchestrator.py --stage a --cluster-id 6106-192556-lj0uddmy
  python scripts/cap_full_orchestrator.py --stage b --cluster-id ...
  python scripts/cap_full_orchestrator.py --stage c --cluster-id ...

Or `--stage all` runs A → B → C sequentially with HF backups between.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Targets per stage (cumulative PNG count expected on Volume after each stage).
STAGE_TARGETS = {
    "A": 2_400,
    "B": 12_000,
    "C": 36_000,
}

# Notebook job templates (paths on Databricks workspace).
PREFILTER_NOTEBOOK = "/Users/alenj00@safeway.com/counterfactual-audit-pipeline/notebooks/09_databricks_prefilter_seeds"
BALANCE_NOTEBOOK = "/Users/alenj00@safeway.com/counterfactual-audit-pipeline/notebooks/10_databricks_balance_seeds"
GEN_NOTEBOOK = "/Users/alenj00@safeway.com/counterfactual-audit-pipeline/notebooks/02_databricks_mvp_run"
HF_PUBLISH_NOTEBOOK = "/Users/alenj00@safeway.com/counterfactual-audit-pipeline/notebooks/08_databricks_stage_publish_to_hf"

# Volume paths (match configs/full.yaml).
GENERATED_DIR = "/Volumes/ds_work/alenj00/cap_cache/runs/full/generated"
SEED_FILTER_DIR = "/Volumes/ds_work/alenj00/cap_cache/runs/full/seed_filter"
CONFIRMED_SEEDS_FILE = f"{SEED_FILTER_DIR}/confirmed_seeds.json"
# Stage A consumes the BALANCED list (output of A0b) — falls back to the raw
# A0 list if balance step is skipped.
BALANCED_SEEDS_FILE = f"{SEED_FILTER_DIR}/confirmed_seeds_balanced.json"

# Per-submission caps.
JOB_TIMEOUT_S = 17 * 3600          # 17 hr (under Databricks's 18-hr cap)
MAX_RESUBMISSIONS_PER_STAGE = 60   # Each chunk is up to 17 hr → 60 × 17 hr = 1020 hr (overkill, but safe)
POLL_INTERVAL_S = 600              # 10 min between status pings


def databricks(*args: str, profile: str = "dev") -> str:
    """Run a databricks CLI command, return stdout."""
    cmd = ["databricks", *args, "--profile", profile]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"databricks CLI failed: {' '.join(cmd)}\nstderr: {result.stderr}")
    return result.stdout


def submit_generation(cluster_id: str, profile: str) -> int:
    """Submit a generation chunk; return run_id."""
    job_spec = {
        "run_name": f"cap_full_chunk_{int(time.time())}",
        "tasks": [{
            "task_key": "gen",
            "existing_cluster_id": cluster_id,
            "notebook_task": {
                "notebook_path": GEN_NOTEBOOK,
                "base_parameters": {
                    "config_path": "/Workspace/Users/alenj00@safeway.com/counterfactual-audit-pipeline/configs/full.yaml",
                    "priority_mode": "paper1_first",
                    "seed_ids_file": BALANCED_SEEDS_FILE,
                },
            },
            "timeout_seconds": JOB_TIMEOUT_S,
        }],
    }
    spec_path = "/tmp/cap_full_chunk_job.json"
    with open(spec_path, "w") as f:
        json.dump(job_spec, f)
    out = databricks("jobs", "submit", "--no-wait", "--json", f"@{spec_path}", profile=profile)
    return int(json.loads(out)["run_id"])


def poll_run(run_id: int, profile: str) -> tuple[str, str]:
    """Poll a job until terminal. Return (life_cycle_state, result_state)."""
    while True:
        out = databricks("jobs", "get-run", str(run_id), "--output", "JSON", profile=profile)
        state = json.loads(out).get("state", {})
        life = state.get("life_cycle_state", "?")
        result = state.get("result_state", "-")
        elapsed_min = int(time.time() - START_TIME) // 60
        print(f"[{time.strftime('%H:%M')}] run={run_id} life={life} result={result} elapsed={elapsed_min}m", flush=True)
        if life in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
            return life, result
        time.sleep(POLL_INTERVAL_S)


def count_pngs(profile: str) -> int:
    """Count PNGs currently on Volume (via a minimal Databricks notebook submission)."""
    # Reuse the existing /tmp/cap_progress_job.json template if present.
    spec_path = "/tmp/cap_progress_job.json"
    if not Path(spec_path).exists():
        raise RuntimeError("Missing /tmp/cap_progress_job.json — author it first")
    out = databricks("jobs", "submit", "--json", f"@{spec_path}", profile=profile)
    run_id = json.loads(out)["run_id"]
    # Wait until terminal (synchronous submit has no --no-wait, but be robust).
    while True:
        d = databricks("jobs", "get-run", str(run_id), "--output", "JSON", profile=profile)
        if json.loads(d).get("state", {}).get("life_cycle_state") == "TERMINATED":
            break
        time.sleep(15)
    task_run_id = json.loads(databricks(
        "jobs", "get-run", str(run_id), "--output", "JSON", profile=profile
    ))["tasks"][0]["run_id"]
    out = databricks("jobs", "get-run-output", str(task_run_id), "--output", "JSON", profile=profile)
    result = json.loads(out).get("notebook_output", {}).get("result", "")
    # Result format: "PNGs written so far: NN / 600"
    import re
    m = re.search(r"PNGs written so far:\s*(\d+)", result)
    return int(m.group(1)) if m else 0


def submit_hf_publish(stage: str, cluster_id: str, profile: str) -> int:
    """Submit the HF publish notebook for a given stage. Returns run_id."""
    job_spec = {
        "run_name": f"cap_hf_publish_stage_{stage.lower()}",
        "tasks": [{
            "task_key": "publish",
            "existing_cluster_id": cluster_id,
            "notebook_task": {
                "notebook_path": HF_PUBLISH_NOTEBOOK,
                "base_parameters": {
                    "stage": stage.lower(),
                    "source_dir": GENERATED_DIR,
                },
            },
            "timeout_seconds": 7200,
        }],
    }
    spec_path = "/tmp/cap_hf_publish_job.json"
    with open(spec_path, "w") as f:
        json.dump(job_spec, f)
    out = databricks("jobs", "submit", "--no-wait", "--json", f"@{spec_path}", profile=profile)
    return int(json.loads(out)["run_id"])


def run_stage(stage: str, cluster_id: str, profile: str) -> bool:
    """Run a single stage. Returns True on success, False if abandoned."""
    target = STAGE_TARGETS[stage]
    print(f"\n=== Stage {stage}: targeting {target} cumulative PNGs ===\n")

    for attempt in range(1, MAX_RESUBMISSIONS_PER_STAGE + 1):
        current = count_pngs(profile)
        print(f"[Stage {stage} attempt {attempt}] current PNG count: {current}/{target}")
        if current >= target:
            print(f"=== Stage {stage} target reached ({current}/{target}) ===")
            return True

        run_id = submit_generation(cluster_id, profile)
        print(f"[Stage {stage} attempt {attempt}] submitted run {run_id}")
        life, result = poll_run(run_id, profile)
        print(f"[Stage {stage} attempt {attempt}] terminal: {life}/{result}")
        # On INTERNAL_ERROR / FAILED, just resubmit — skip-if-exists handles partial work.

    print(f"=== Stage {stage} ABANDONED after {MAX_RESUBMISSIONS_PER_STAGE} attempts ===")
    return False


def run_hf_backup(stage: str, cluster_id: str, profile: str) -> bool:
    print(f"\n=== HF backup for Stage {stage} ===")
    run_id = submit_hf_publish(stage, cluster_id, profile)
    life, result = poll_run(run_id, profile)
    return result == "SUCCESS"


START_TIME = time.time()


def run_prefilter(cluster_id: str, profile: str) -> bool:
    """Phase A0: prefilter seeds (idempotent — skip if confirmed_seeds.json exists)."""
    # Idempotency check: if the confirmed-seeds list already exists, skip.
    # The orchestrator can't directly read the Volume from the laptop, so we
    # submit a tiny notebook to check. Cheap.
    print("\n=== Phase A0: seed prefilter ===")
    job_spec = {
        "run_name": "cap_prefilter_seeds",
        "tasks": [{
            "task_key": "prefilter",
            "existing_cluster_id": cluster_id,
            "notebook_task": {
                "notebook_path": PREFILTER_NOTEBOOK,
                "base_parameters": {
                    "config_path": "/Workspace/Users/alenj00@safeway.com/counterfactual-audit-pipeline/configs/full.yaml",
                    "candidates": "250",
                    "target_confirmed": "200",
                    "threshold": "0.5",
                    "output_dir": SEED_FILTER_DIR,
                },
            },
            "timeout_seconds": 5 * 3600,
        }],
    }
    spec_path = "/tmp/cap_prefilter_job.json"
    with open(spec_path, "w") as f:
        json.dump(job_spec, f)
    out = databricks("jobs", "submit", "--no-wait", "--json", f"@{spec_path}", profile=profile)
    run_id = int(json.loads(out)["run_id"])
    print(f"Phase A0 submitted: run {run_id}")
    life, result = poll_run(run_id, profile)
    return result == "SUCCESS"


def run_balance(cluster_id: str, profile: str) -> bool:
    """Phase A0b: check + adaptively rebalance the confirmed-seeds list.

    Idempotent: notebook 10 runs the balance check; if --rebalance is on
    and any cell is underfilled, it iteratively oversamples + re-prefilters
    until balanced (or max iterations). Always writes confirmed_seeds_balanced.json.
    """
    print("\n=== Phase A0b: balance check + adaptive rebalance ===")
    job_spec = {
        "run_name": "cap_balance_seeds",
        "tasks": [{
            "task_key": "balance",
            "existing_cluster_id": cluster_id,
            "notebook_task": {
                "notebook_path": BALANCE_NOTEBOOK,
                "base_parameters": {
                    "config_path": "/Workspace/Users/alenj00@safeway.com/counterfactual-audit-pipeline/configs/full.yaml",
                    "confirmed_seeds_file": CONFIRMED_SEEDS_FILE,
                    "output_dir": SEED_FILTER_DIR,
                    "target_per_cell": "14",
                    "max_iterations": "5",
                    "extra_per_iteration": "50",
                },
            },
            "timeout_seconds": 5 * 3600,
        }],
    }
    spec_path = "/tmp/cap_balance_job.json"
    with open(spec_path, "w") as f:
        json.dump(job_spec, f)
    out = databricks("jobs", "submit", "--no-wait", "--json", f"@{spec_path}", profile=profile)
    run_id = int(json.loads(out)["run_id"])
    print(f"Phase A0b submitted: run {run_id}")
    life, result = poll_run(run_id, profile)
    return result == "SUCCESS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["a", "b", "c", "all"], required=True)
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--skip-prefilter", action="store_true",
                        help="Skip the A0 prefilter step (assume confirmed_seeds.json already exists)")
    parser.add_argument("--skip-balance", action="store_true",
                        help="Skip the A0b balance/rebalance step")
    args = parser.parse_args()

    stages_to_run = ["A", "B", "C"] if args.stage == "all" else [args.stage.upper()]

    # Phase A0: prefilter (only relevant if Stage A is in scope)
    if "A" in stages_to_run and not args.skip_prefilter:
        ok = run_prefilter(args.cluster_id, args.profile)
        if not ok:
            print("Aborting: prefilter failed.", file=sys.stderr)
            return 1

    # Phase A0b: balance + adaptive rebalance — produces confirmed_seeds_balanced.json
    if "A" in stages_to_run and not args.skip_balance:
        ok = run_balance(args.cluster_id, args.profile)
        if not ok:
            print("Aborting: balance step failed.", file=sys.stderr)
            return 1

    for st in stages_to_run:
        ok = run_stage(st, args.cluster_id, args.profile)
        if not ok:
            print(f"Aborting after Stage {st} failure.", file=sys.stderr)
            return 1
        ok = run_hf_backup(st, args.cluster_id, args.profile)
        if not ok:
            print(f"WARNING: Stage {st} HF backup failed; continuing anyway.", file=sys.stderr)

    print("\n=== All stages complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
