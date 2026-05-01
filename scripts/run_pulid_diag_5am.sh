#!/usr/bin/env bash
# Sleep until 5am local time, then run the PuLID diagnostic end-to-end.
#
# Usage (run before bed, plug in laptop, leave terminal open):
#
#   cd ~/counterfactual-audit-pipeline
#   nohup caffeinate -is bash scripts/run_pulid_diag_5am.sh > /tmp/cap_pulid_diag_5am.log 2>&1 &
#   disown
#
# In the morning:
#
#   cat /tmp/cap_pulid_diag_5am.log         # full timeline
#   cat /tmp/cap_pulid_diag_5am.verdict     # one-line summary if it finished
#
# What it does (all autonomous, no input needed):
#   1. Sleep until 5:00 local time.
#   2. Start cluster 6106-192556-lj0uddmy. Retry up to 6× with 15min waits if
#      GCP returns INSUFFICIENT_CAPACITY (STOCKOUT) for the L4 zone.
#   3. Once the cluster is RUNNING, submit notebook 06_databricks_pulid_diagnostic
#      as a one-time job (the FluxPuLIDNativeGenerator architectural fix).
#   4. Poll the run until terminal.
#   5. Fetch the run's notebook output and extract the 5 cosine sims +
#      Pixel MSE + diagnostic verdict line.
#   6. Write a one-line summary to /tmp/cap_pulid_diag_5am.verdict.
#
# Exit code: 0 if diagnostic ran (regardless of verdict), 1 if cluster never
# came up after 6 retries.

set -uo pipefail

PROFILE="${DATABRICKS_PROFILE:-dev}"
CLUSTER_ID="6106-192556-lj0uddmy"
JOB_JSON="/tmp/cap_pulid_diag_job.json"
LOG="/tmp/cap_pulid_diag_5am.log"
VERDICT="/tmp/cap_pulid_diag_5am.verdict"
TARGET_HOUR=5

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ------- 0. Sleep until 5am local --------------------------------------------

now_secs=$(($(date +%H) * 3600 + 10#$(date +%M) * 60 + 10#$(date +%S)))
target_secs=$((TARGET_HOUR * 3600))
if [ "$now_secs" -lt "$target_secs" ]; then
  wait_s=$((target_secs - now_secs))
else
  wait_s=$((86400 - now_secs + target_secs))
fi

log "current time: $(date)"
log "waiting ${wait_s}s (~$((wait_s / 3600))h $((wait_s % 3600 / 60))m) until 5:00 local..."
sleep "$wait_s"
log "woke up at $(date) — starting diagnostic flow"

# ------- 1. Recreate job JSON if missing -------------------------------------

if [ ! -f "$JOB_JSON" ]; then
  log "recreating $JOB_JSON"
  cat > "$JOB_JSON" <<EOF
{
  "run_name": "cap_pulid_diag_5am",
  "tasks": [{
    "task_key": "diag",
    "existing_cluster_id": "$CLUSTER_ID",
    "notebook_task": {
      "notebook_path": "/Users/alenj00@safeway.com/counterfactual-audit-pipeline/notebooks/06_databricks_pulid_diagnostic"
    }
  }]
}
EOF
fi

# ------- 2. Start cluster (retry on STOCKOUT) --------------------------------

cluster_state() {
  databricks clusters get "$CLUSTER_ID" --profile "$PROFILE" --output JSON 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('state','?'))"
}

cluster_msg() {
  databricks clusters get "$CLUSTER_ID" --profile "$PROFILE" --output JSON 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('state_message','')[:160])"
}

ATTEMPTS=6
WAIT_BETWEEN=900   # 15 min
ATTEMPT=0
while [ "$ATTEMPT" -lt "$ATTEMPTS" ]; do
  ATTEMPT=$((ATTEMPT + 1))
  log "cluster start attempt ${ATTEMPT}/${ATTEMPTS}"
  databricks clusters start "$CLUSTER_ID" --profile "$PROFILE" 2>&1 | head -2 || true

  # Poll until terminal-for-this-attempt state (RUNNING / TERMINATED / ERROR)
  while :; do
    STATE=$(cluster_state)
    log "  cluster state=$STATE — $(cluster_msg)"
    case "$STATE" in
      RUNNING)        break ;;
      TERMINATED|ERROR) break ;;
      *)              sleep 30 ;;
    esac
  done

  if [ "$STATE" = "RUNNING" ]; then
    log "✓ cluster RUNNING"
    break
  fi

  if [ "$ATTEMPT" -lt "$ATTEMPTS" ]; then
    log "  retrying in ${WAIT_BETWEEN}s..."
    sleep "$WAIT_BETWEEN"
  fi
done

if [ "$STATE" != "RUNNING" ]; then
  log "✗ FAILED: cluster never reached RUNNING after $ATTEMPTS attempts. Final state: $STATE"
  echo "FAILED: cluster never RUNNING (likely GCP_INSUFFICIENT_CAPACITY)" > "$VERDICT"
  exit 1
fi

# ------- 3. Submit the diagnostic --------------------------------------------

log "submitting diagnostic job..."
SUBMIT_OUT=$(databricks jobs submit --no-wait --json @"$JOB_JSON" --profile "$PROFILE" 2>&1)
RUN_ID=$(echo "$SUBMIT_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('run_id',''))" 2>/dev/null || true)

if [ -z "$RUN_ID" ]; then
  log "✗ job submit failed: $SUBMIT_OUT"
  echo "FAILED: submit failed — $SUBMIT_OUT" > "$VERDICT"
  exit 1
fi
log "✓ submitted run_id=$RUN_ID"

# ------- 4. Poll the run until terminal --------------------------------------

while :; do
  RUN_STATE=$(databricks jobs get-run "$RUN_ID" --profile "$PROFILE" --output JSON 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('state',{}).get('life_cycle_state','?'))")
  RESULT=$(databricks jobs get-run "$RUN_ID" --profile "$PROFILE" --output JSON 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('state',{}).get('result_state','-'))")
  ELAPSED=$(databricks jobs get-run "$RUN_ID" --profile "$PROFILE" --output JSON 2>/dev/null \
    | python3 -c "import sys,json,time; d=json.load(sys.stdin); s=d.get('start_time',0); print(int(time.time()*1000-s)//1000 if s else -1)")
  log "  run state=$RUN_STATE result=$RESULT elapsed=${ELAPSED}s"
  case "$RUN_STATE" in
    TERMINATED|INTERNAL_ERROR|SKIPPED) break ;;
    *) sleep 90 ;;
  esac
done

log "run terminal: state=$RUN_STATE result=$RESULT"

# ------- 5. Fetch verdict ----------------------------------------------------

TASK_RUN_ID=$(databricks jobs get-run "$RUN_ID" --profile "$PROFILE" --output JSON 2>/dev/null \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['tasks'][0]['run_id'])")

log "============== diagnostic output =============="
RUN_OUT=$(databricks jobs get-run-output "$TASK_RUN_ID" --profile "$PROFILE" --output JSON 2>/dev/null)

echo "$RUN_OUT" | python3 - <<'PY'
import sys, json, re

raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception as e:
    print("could not parse run output:", e)
    sys.exit(0)

err = d.get("error", "") or ""
if err:
    print("=== ERROR ===")
    print(err[:2500])

nb = d.get("notebook_output", {}).get("result", "") or ""
if nb:
    print("=== notebook output (last 6000 chars) ===")
    print(nb[-6000:])
PY

# Try to summarize: extract the verdict line + cosine sim values
SUMMARY=$(echo "$RUN_OUT" | python3 - <<'PY'
import sys, json, re
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception:
    print("(unable to parse run output)")
    sys.exit(0)
text = (d.get("notebook_output", {}).get("result", "") or "") + "\n" + (d.get("error", "") or "")
lines = text.splitlines()
keep = []
for L in lines:
    if any(k in L for k in (
        "Pixel MSE",
        "ArcFace self-sim",
        "ArcFace seed A vs seed B",
        "ArcFace seed A vs image A",
        "ArcFace seed B vs image B",
        "ArcFace image A vs image B",
        "Diagnostic verdict",
        "PuLID HOOKS NOT WIRED",
        "VALIDATOR ALIGNMENT BROKEN",
        "PuLID IS WORKING",
        "AMBIGUOUS",
    )):
        keep.append(L.strip())
print("\n".join(keep) if keep else "(verdict markers not found in run output)")
PY
)

log "============== SUMMARY =============="
echo "$SUMMARY" | tee "$VERDICT"
log "run page: https://2546847502462311.1.gcp.databricks.com/?o=2546847502462311#job/$(echo "$RUN_OUT" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("job_id",""))' 2>/dev/null)/run/$RUN_ID"
log "DIAGNOSTIC COMPLETE — full log: $LOG, verdict: $VERDICT"
