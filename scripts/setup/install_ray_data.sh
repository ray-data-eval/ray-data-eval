#!/usr/bin/env bash
# Install Ray Data by replacing ray/data inside an installed Ray 2.40.0.
#
# Usage:
#   scripts/setup/install_ray_data.sh [path-to-ray-checkout]
#
# If no path is supplied, the source is cloned from ray-data-eval/ray, branch
# nsdi27-artifact.


set -euo pipefail

RAY_REPO="${RAY_REPO:-https://github.com/ray-data-eval/ray.git}"
RAY_BRANCH="${RAY_BRANCH:-nsdi27-artifact}"
RAY_DATA_SRC="${1:-$HOME/ray-data-eval-ray}"
REQUIRED_RAY_VERSION="2.40.0"

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
  branch=$(git -C "$RAY_DATA_SRC" rev-parse --abbrev-ref HEAD)
  log "Branch:       $branch ($(git -C "$RAY_DATA_SRC" rev-parse --short HEAD))"
  if [ "$branch" != "$RAY_BRANCH" ] && [ "$branch" != "HEAD" ]; then
    log "WARNING: expected '$RAY_BRANCH'; other branches carry a different"
    log "         subset of the policies and some experiments will not run."
  fi
fi

installed=$(cd /tmp && python -c 'import ray; print(ray.__version__)' 2>/dev/null || echo none)
if [ "$installed" != "$REQUIRED_RAY_VERSION" ]; then
  log "Installing ray==$REQUIRED_RAY_VERSION (found: $installed)"
  pip install --quiet "ray[data]==$REQUIRED_RAY_VERSION"
else
  log "Found ray==$REQUIRED_RAY_VERSION"
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
    "op_resource_reservation_enabled": "memory budget (Alg. 2)",
    "op_resource_reservation_ratio": "optimistic/pessimistic policy (4.3.1)",
    "scheduling_policy": "policy selector",
    "microbatch_size": "Spark Streaming / Drizzle emulation",
    "microbatch_group_size": "Drizzle group scheduling",
    "microbatch_stage_barrier": "Drizzle pre-scheduling",
    "llf_latency_target": "Cameo LLF emulation (2.2)",
    "llf_disable_admission_control": "Cameo admission-control ablation",
}

ctx = DataContext.get_current()
print(f"    ray {ray.__version__}")
missing = [k for k in required if not hasattr(ctx, k)]
for key, why in required.items():
    print(f"    [{'ok' if hasattr(ctx, key) else 'missing':>7}] {key:<32} {why}")

# DataContext silently ignores unknown settings.
if missing:
    sys.exit(f"\nFAILED: {len(missing)} setting(s) missing; install did not apply.")
print("\n    All settings present.")
PY

echo
echo "Ready."
echo
