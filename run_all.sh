#!/usr/bin/env bash
# Convenience wrapper for local replication (mirrors the Code Ocean `code/run`).
# Usage:
#   1. Place the ESS11 integrated file at data/ESS11.csv  (see data/README.md)
#   2. pip install -r requirements.txt
#   3. bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
exec bash code/run
