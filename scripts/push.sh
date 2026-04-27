#!/usr/bin/env bash
# Push to personal GitHub using a fine-grained PAT stored in .env.
#
# Why this exists: this machine has work GitHub credentials cached in macOS
# Keychain that win over a personal PAT during normal `git push`. This script
# bypasses the credential helper chain and authenticates directly via a Bearer
# token read from .env (which is gitignored).
#
# Usage:
#   ./scripts/push.sh                # equivalent to: git push origin main
#   ./scripts/push.sh origin <branch>
#
# .env format (no quotes, no spaces around =):
#   fine_grained_github_token=ghp_xxx

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
ENV_FILE="$REPO_ROOT/.env"

if [ ! -f "$ENV_FILE" ] || [ ! -s "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE is missing or empty." >&2
  echo "Add a line like:  fine_grained_github_token=ghp_xxx" >&2
  exit 1
fi

TOKEN=$(grep -E '^fine_grained_github_token=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'" | tr -d '[:space:]')
if [ -z "$TOKEN" ]; then
  echo "ERROR: 'fine_grained_github_token' not found in $ENV_FILE" >&2
  echo "Expected format (no quotes, no spaces around =):" >&2
  echo "  fine_grained_github_token=ghp_xxx" >&2
  exit 1
fi

if [ $# -eq 0 ]; then
  set -- origin main
fi

BASIC=$(printf 'alenjani:%s' "$TOKEN" | base64 | tr -d '\n')
exec git -c credential.helper= -c http.extraheader="Authorization: Basic $BASIC" push "$@"
