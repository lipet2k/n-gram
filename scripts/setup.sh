#!/usr/bin/env bash
# Install the environment. Data, tables and query sets are built by scripts/run.sh as needed.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync
