#!/usr/bin/env bash
set -euo pipefail
C78_PACKAGE_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
exec "${CART78_LAUNCH_PYTHON:-python3}" "$C78_PACKAGE_DIR/setup_env.py" "$@"
