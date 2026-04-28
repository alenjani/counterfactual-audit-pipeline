#!/usr/bin/env bash
# Sync this repo's tracked files to the Databricks workspace via per-file
# `workspace import`. We use this instead of `databricks repos update` because
# the repo update path requires a workspace-wide GitHub credential which would
# conflict with the user's work GitHub identity in this Databricks workspace.
#
# Files synced (anything cluster runs need to see):
#   - src/**/*.py      — for `pip install` of cap
#   - pyproject.toml   — for `pip install`
#   - notebooks/*.ipynb — entry points for `databricks jobs submit`
#   - configs/*.yaml   — runtime configs
#
# Usage:
#   ./scripts/sync_databricks.sh                # default profile=dev
#   DATABRICKS_PROFILE=prod ./scripts/sync_databricks.sh
#
# Requires: databricks CLI authed to the target workspace.

set -euo pipefail

WS="/Users/alenj00@safeway.com/counterfactual-audit-pipeline"
PROFILE="${DATABRICKS_PROFILE:-dev}"

cd "$(git rev-parse --show-toplevel)"

echo "Syncing tracked files → ${WS} (profile=${PROFILE}) ..."

count=0
fail=0

while IFS= read -r file; do
  [ -z "$file" ] && continue
  [ ! -f "$file" ] && continue

  if [[ "$file" == *.ipynb ]]; then
    target="${WS}/${file%.ipynb}"
    out=$(databricks workspace import "$target" \
      --file "$file" --format JUPYTER --language PYTHON \
      --overwrite --profile "$PROFILE" 2>&1) || { fail=$((fail+1)); echo "  ✗ ${file}: ${out}"; continue; }
  else
    target="${WS}/${file}"
    out=$(databricks workspace import "$target" \
      --file "$file" --format AUTO \
      --overwrite --profile "$PROFILE" 2>&1) || { fail=$((fail+1)); echo "  ✗ ${file}: ${out}"; continue; }
  fi

  echo "  ✓ ${file}"
  count=$((count+1))
done < <(git ls-files | grep -E '^(src/.+\.py|notebooks/.+\.ipynb|configs/.+\.ya?ml|pyproject\.toml|CLAUDE\.md|README\.md)$')

echo
echo "Synced ${count} file(s)."
if [ "$fail" -gt 0 ]; then
  echo "Failed:  ${fail} file(s) — see above."
  exit 1
fi
