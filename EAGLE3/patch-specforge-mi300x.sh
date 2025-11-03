#!/bin/bash

# Patch SpecForge's Triton log-softmax kernel so it respects MI300X thread limits.

set -euo pipefail

SCRIPT_DIR=$(
  cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd
)

REPO_LOSS_FILE="$SCRIPT_DIR/SpecForge/specforge/core/loss.py"

if [[ ! -f "$REPO_LOSS_FILE" ]]; then
  echo "error: expected SpecForge/specforge/core/loss.py at $REPO_LOSS_FILE" >&2
  exit 1
fi

SITE_LOSS_FILE=$(
  python - <<'PY'
import importlib
import inspect
import os
import sys

try:
    module = importlib.import_module("specforge.core.loss")
except ModuleNotFoundError as exc:
    sys.stderr.write("error: specforge is not installed in this environment\n")
    sys.exit(1)

loss_path = inspect.getfile(module)
if not os.path.exists(loss_path):
    sys.stderr.write(f"error: resolved loss.py path does not exist: {loss_path}\n")
    sys.exit(1)

print(loss_path)
PY
)

if [[ -z "$SITE_LOSS_FILE" ]]; then
  echo "error: failed to locate installed specforge.core.loss" >&2
  exit 1
fi

if ! cmp -s "$REPO_LOSS_FILE" "$SITE_LOSS_FILE"; then
  echo "Patching $(realpath "$SITE_LOSS_FILE") from repo copy..."
  cp "$REPO_LOSS_FILE" "$SITE_LOSS_FILE"
else
  echo "Installed specforge.core.loss already matches repo copy; nothing to do."
fi

echo "Done. Ensure your environment points at SpecForge via 'pip install -e SpecForge'."
