#!/bin/bash
# Installs TimesFM into .venv so the forecast skill works in fresh sessions.
set -euo pipefail
cd "$CLAUDE_PROJECT_DIR"
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c "import timesfm, pandas, matplotlib" 2>/dev/null; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r timesfm/requirements.txt
fi
