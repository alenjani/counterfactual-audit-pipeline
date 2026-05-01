#!/usr/bin/env bash
# Fire the PuLID diagnostic NOW, with patient retries on GCP STOCKOUT.
#
# Use when GCP is currently out of L4 capacity but you want the diagnostic
# to run as soon as capacity returns — script will retry every 15 min for
# up to 6 hours.
#
# Usage (run, then go about your day):
#
#   cd ~/counterfactual-audit-pipeline
#   nohup caffeinate -is bash scripts/run_pulid_diag_now.sh > /tmp/cap_pulid_diag_now.log 2>&1 &
#   disown
#
# Check progress:
#
#   tail -f /tmp/cap_pulid_diag_now.log
#   cat /tmp/cap_pulid_diag_now.verdict   # one-line summary once finished
#
# Difference from run_pulid_diag_5am.sh: no initial sleep-until-5am, more
# patient retry budget (24 attempts × 15min = 6 hours instead of 6 attempts).

set -uo pipefail

PROFILE="${DATABRICKS_PROFILE:-dev}"
CLUSTER_ID="6106-192556-lj0uddmy"
JOB_JSON="/tmp/cap_pulid_diag_job.json"
LOG="/tmp/cap_pulid_diag_now.log"
VERDICT="/tmp/cap_pulid_diag_now.verdict"

ATTEMPTS=24
WAIT_BETWEEN=900   # 15 min between retries

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ------- 1. Recreate job JSON if missing -------------------------------------

if [ ! -f "$JOB_JSON" ]; then
  log "recreating $JOB_JSON"
  cat > "$JOB_JSON" <<EOF
{
  "run_name": "cap_pulid_diag_now",
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

# ------- 2. Start cluster (retry on STOCKOUT, up to ATTEMPTS times) ----------

cluster_state() {
  databricks clusters get "$CLUSTER_ID" --profile "$PROFILE" --output JSON 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('state','?'))"
}

cluster_msg() {
  databricks clusters get "$CLUSTER_ID" --profile "$PROFILE" --output JSON 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('state_message','')[:160])"
}

ATTEMPT=0
while [ "$ATTEMPT" -lt "$ATTEMPTS" ]; do
  ATTEMPT=$((ATTEMPT + 1))
  log "cluster start attempt ${ATTEMPT}/${ATTEMPTS}"
  databricks clusters start "$CLUSTER_ID" --profile "$PROFILE" 2>&1 | head -2 || true

  # Poll until terminal-for-this-attempt
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
    log "✓ cluster RUNNING (attempt $ATTEMPT)"
    break
  fi

  if [ "$ATTEMPT" -lt "$ATTEMPTS" ]; then
    log "  STOCKOUT or other failure — retrying in ${WAIT_BETWEEN}s..."
    sleep "$WAIT_BETWEEN"
  fi
done

if [ "$STATE" != "RUNNING" ]; then
  log "✗ FAILED: cluster never reached RUNNING after $ATTEMPTS attempts (~$((ATTEMPTS * WAIT_BETWEEN / 3600))h). Final: $STATE"
  echo "FAILED: cluster never RUNNING after $ATTEMPTS retries (~$((ATTEMPTS * WAIT_BETWEEN / 3600))h). GCP zone likely sustained STOCKOUT — retry tomorrow." > "$VERDICT"
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

# ------- 4. Poll the run -----------------------------------------------------

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

# ------- 5. Fetch the verdict ------------------------------------------------

TASK_RUN_ID=$(databricks jobs get-run "$RUN_ID" --profile "$PROFILE" --output JSON 2>/dev/null \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['tasks'][0]['run_id'])")

log "============== diagnostic output =============="
RUN_OUT=$(databricks jobs get-run-output "$TASK_RUN_ID" --profile "$PROFILE" --output JSON 2>/dev/null)

echo "$RUN_OUT" | python3 - <<'PY'
import sys, json
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception as e:
    print("could not parse run output:", e); sys.exit(0)
err = d.get("error", "") or ""
if err:
    print("=== ERROR ===")
    print(err[:2500])
nb = d.get("notebook_output", {}).get("result", "") or ""
if nb:
    print("=== notebook output (last 6000 chars) ===")
    print(nb[-6000:])
PY

SUMMARY=$(echo "$RUN_OUT" | python3 - <<'PY'
import sys, json
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception:
    print("(unable to parse run output)"); sys.exit(0)
text = (d.get("notebook_output", {}).get("result", "") or "") + "\n" + (d.get("error", "") or "")
keep = []
for L in text.splitlines():
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
print("\n".join(keep) if keep else "(verdict markers not found)")
PY
)

log "============== SUMMARY =============="
echo "$SUMMARY" | tee "$VERDICT"
log "DIAGNOSTIC COMPLETE — full log: $LOG, verdict: $VERDICT"
