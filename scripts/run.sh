#!/usr/bin/env bash
# Run the configuration in configs/config.json: build the run corpus if needed, then its experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m src.main
