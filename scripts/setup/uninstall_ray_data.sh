#!/usr/bin/env bash
# Restore the original Ray installation, undoing scripts/setup/install_ray_data.sh.
#

set -euo pipefail

SITE_RAY=$(python -c 'import ray, os; print(os.path.dirname(ray.__file__))' 2>/dev/null) \
  || { echo "Ray is not installed; nothing to undo." >&2; exit 0; }

BACKUP=$(find "$SITE_RAY" -maxdepth 1 -name 'data.orig-*' -type d | head -1)

if [ -z "$BACKUP" ]; then
  echo "No backup of the original ray/data found in $SITE_RAY."
  echo "Ray Data was probably never installed here. To get a clean Ray:"
  echo "  pip install --force-reinstall 'ray[data]==2.40.0'"
  exit 0
fi

echo "Restoring $(basename "$BACKUP") -> ray/data"
rm -rf "$SITE_RAY/data"
cp -a "$BACKUP" "$SITE_RAY/data"
rm -rf "$BACKUP"

python - <<'PY'
from ray.data import DataContext
ctx = DataContext.get_current()
leftover = [k for k in ("scheduling_policy", "microbatch_size", "llf_latency_target")
            if hasattr(ctx, k)]
print("  Ray Data settings still present:", leftover or "none - original Ray restored")
PY
