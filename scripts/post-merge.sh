#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# The Replit-managed Python environment is marked externally managed by PEP
# 668. The project interpreter stores packages in .pythonlibs, so explicitly
# allow the non-interactive install there rather than attempting to mutate the
# immutable Nix store.
python -m pip install \
  --disable-pip-version-check \
  --no-input \
  --break-system-packages \
  -r backend/requirements.txt
python backend/aquag-training.py