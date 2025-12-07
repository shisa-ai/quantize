#!/bin/bash

# Patch SpecForge's Triton log-softmax kernel so it respects MI300X thread limits.
# NOTE: As of SpecForge 0.1.1+, this fix is included upstream. This script
# is kept for verification and in case of custom site-packages installs.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_LOSS_FILE="$SCRIPT_DIR/SpecForge/specforge/core/loss.py"

# Conda/Mamba environment
ENV=${ENV:-quantize}

if [[ ! -f "$REPO_LOSS_FILE" ]]; then
  echo "error: expected SpecForge/specforge/core/loss.py at $REPO_LOSS_FILE" >&2
  exit 1
fi

# Check if the MI300X fix exists in the repo file
if ! grep -q "AMD GPU (ROCm)" "$REPO_LOSS_FILE"; then
  echo "warning: MI300X fix not found in repo loss.py - may need manual update" >&2
fi

# Find the installed specforge location - handle both editable and regular installs
INSTALL_INFO=$(mamba run -n "$ENV" pip show specforge 2>/dev/null)
EDITABLE_LOC=$(echo "$INSTALL_INFO" | grep -i "Editable project location:" | cut -d: -f2- | tr -d ' ')
SITE_LOC=$(echo "$INSTALL_INFO" | grep -i "^Location:" | cut -d: -f2- | tr -d ' ')

if [[ -n "$EDITABLE_LOC" ]]; then
  SITE_LOSS_FILE="$EDITABLE_LOC/specforge/core/loss.py"
  echo "Editable install detected at: $EDITABLE_LOC"
elif [[ -n "$SITE_LOC" ]]; then
  SITE_LOSS_FILE="$SITE_LOC/specforge/core/loss.py"
  echo "Site-packages install at: $SITE_LOC"
else
  echo "error: could not locate installed specforge" >&2
  echo "Is specforge installed? Try: mamba run -n $ENV pip show specforge" >&2
  exit 1
fi

if [[ ! -f "$SITE_LOSS_FILE" ]]; then
  echo "error: loss.py not found at $SITE_LOSS_FILE" >&2
  exit 1
fi

echo "Repo file:      $REPO_LOSS_FILE"
echo "Installed file: $SITE_LOSS_FILE"

# For editable installs, they're the same file
if [[ "$(realpath "$REPO_LOSS_FILE")" == "$(realpath "$SITE_LOSS_FILE")" ]]; then
  echo "Editable install: repo and installed files are the same."
elif ! cmp -s "$REPO_LOSS_FILE" "$SITE_LOSS_FILE"; then
  echo "Files differ. Patching $SITE_LOSS_FILE from repo copy..."
  cp "$REPO_LOSS_FILE" "$SITE_LOSS_FILE"
  echo "Done."
else
  echo "Installed specforge.core.loss already matches repo copy; nothing to do."
fi

# Verify the fix is in place
if grep -q "AMD GPU (ROCm)" "$SITE_LOSS_FILE"; then
  echo "MI300X num_warps fix is present."
else
  echo "warning: MI300X fix not found in installed loss.py!" >&2
  exit 1
fi

echo "Patch verification complete."
