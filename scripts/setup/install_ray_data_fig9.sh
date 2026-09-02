#!/usr/bin/env bash
# Install the Ray Data build used for Figure 9 (memory-aware pipelining).
#
# Usage:
#   scripts/setup/install_ray_data_fig9.sh [path-to-ray-checkout]
#
# Branch nsdi27-fig9 of ray-data-eval/ray is based on Ray master, not 2.40.0,
# so use a separate conda environment.

set -euo pipefail

RAY_REPO="${RAY_REPO:-https://github.com/ray-data-eval/ray.git}"
RAY_BRANCH="${RAY_BRANCH:-nsdi27-fig9}"
RAY_DATA_SRC="${1:-$HOME/ray-data-eval-ray-fig9}"
# The upstream commit the branch forked from.
UPSTREAM_BASE=05067f4955e6e79e3cbf05fe6875677de5663924
WHEEL="https://s3-us-west-2.amazonaws.com/ray-wheels/master/${UPSTREAM_BASE}/ray-3.0.0.dev0-cp311-cp311-manylinux2014_x86_64.whl"
REQUIRED_RAY_VERSION="3.0.0.dev0"

log()  { printf '  %s\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

echo
echo "Ray Data installer ($RAY_BRANCH)"
echo "================================"

if [ ! -d "$RAY_DATA_SRC" ]; then
  log "Cloning $RAY_REPO ($RAY_BRANCH) -> $RAY_DATA_SRC"
  git clone --quiet --depth 1 --filter=blob:none --sparse \
      -b "$RAY_BRANCH" "$RAY_REPO" "$RAY_DATA_SRC"
  git -C "$RAY_DATA_SRC" sparse-checkout set python/ray/data >/dev/null
fi

[ -d "$RAY_DATA_SRC/python/ray/data" ] || fail "not a Ray checkout: $RAY_DATA_SRC"

log "Ray Data source: $RAY_DATA_SRC"
if git -C "$RAY_DATA_SRC" rev-parse --git-dir >/dev/null 2>&1; then
  log "Branch:       $(git -C "$RAY_DATA_SRC" rev-parse --abbrev-ref HEAD) ($(git -C "$RAY_DATA_SRC" rev-parse --short HEAD))"
fi

installed=$(cd /tmp && python -c 'import ray; print(ray.__version__)' 2>/dev/null || echo none)
if [ "$installed" != "$REQUIRED_RAY_VERSION" ]; then
  log "Installing Ray $REQUIRED_RAY_VERSION from the matching nightly build"
  pip install --quiet "$WHEEL"
  pip install --quiet "numpy==1.26.4" "pandas==2.2.2" "pyarrow==14.0.2" psutil humanize grpcio
else
  log "Found Ray $REQUIRED_RAY_VERSION"
fi

SITE_RAY=$(cd /tmp && python -c 'import ray, os; print(os.path.dirname(ray.__file__))')
log "Target:       $SITE_RAY"
[ -d "$SITE_RAY/data" ] || fail "no ray/data in $SITE_RAY"

BACKUP="$SITE_RAY/data.orig-$REQUIRED_RAY_VERSION"
if [ ! -d "$BACKUP" ]; then
  log "Backing up the original ray/data -> $(basename "$BACKUP")"
  cp -a "$SITE_RAY/data" "$BACKUP"
else
  log "Restoring the original before reinstalling"
  rm -rf "$SITE_RAY/data"
  cp -a "$BACKUP" "$SITE_RAY/data"
fi

log "Copying ray/data into the installed Ray"
cp -a "$RAY_DATA_SRC/python/ray/data/." "$SITE_RAY/data/"
find "$SITE_RAY/data" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo
log "Verifying..."
cd /tmp && python - <<'PY'
import sys

try:
    import ray
    from ray.data import DataContext
except Exception as exc:
    sys.exit(f"FAILED: cannot import ray.data after install: {exc}")

required = {
    "is_budget_policy": "memory-aware admission control (Figure 9)",
    "is_conservative_policy": "anti-spill variant",
}

ctx = DataContext.get_current()
print(f"    ray {ray.__version__}")
missing = [k for k in required if not hasattr(ctx, k)]
for key, why in required.items():
    print(f"    [{'ok' if hasattr(ctx, key) else 'missing':>7}] {key:<24} {why}")

# Without these the benchmark runs unmodified Ray Data with no error.
if missing:
    sys.exit(f"\nFAILED: {len(missing)} setting(s) missing; install did not apply.")
print("\n    All settings present.")
PY

echo
echo "Ready.  Next:  bash scripts/run_fig9.sh"
echo
