#!/usr/bin/env bash
# Install Ray Data using a Ray 2.40.0 wheel.
#
# Usage:
#   scripts/install_ray_data.sh [path-to-ray-fork]
#
# The changes are confined to python/ray/data/, with none to C++ or the build
# system, so this is a source overlay on the released wheel rather than a full
# Ray build. Setup takes a few minutes instead of a few hours.
#
# The checkout must be on the `nsdi27-artifact` branch, which unifies the
# scheduling policies used by the baseline emulations.

set -euo pipefail

RAY_DATA_SRC="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../ray-fork" && pwd)}"
REQUIRED_RAY_VERSION="2.40.0"

log()  { printf '  %s\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

echo
echo "Ray Data installer"
echo "==============="

[ -d "$RAY_DATA_SRC/python/ray/data" ] \
  || fail "not a Ray checkout: $RAY_DATA_SRC
Pass the path explicitly:  scripts/install_ray_data.sh /path/to/ray"

log "Ray Data source: $RAY_DATA_SRC"

if command -v git >/dev/null && git -C "$RAY_DATA_SRC" rev-parse --git-dir >/dev/null 2>&1; then
  branch=$(git -C "$RAY_DATA_SRC" rev-parse --abbrev-ref HEAD)
  commit=$(git -C "$RAY_DATA_SRC" rev-parse --short HEAD)
  log "Branch:       $branch ($commit)"
  if [ "$branch" != "nsdi27-artifact" ]; then
    log "WARNING: expected branch 'nsdi27-artifact'. Other branches carry only"
    log "         a subset of the scheduling policies, so some baselines will"
    log "         refuse to run."
  fi
fi

# ---------------------------------------------------------------- stock wheel
installed=$(python -c 'import ray; print(ray.__version__)' 2>/dev/null || echo "none")
if [ "$installed" != "$REQUIRED_RAY_VERSION" ]; then
  log "Installing ray==$REQUIRED_RAY_VERSION (found: $installed)"
  pip install --quiet "ray[data]==$REQUIRED_RAY_VERSION"
else
  log "Found ray==$REQUIRED_RAY_VERSION"
fi

SITE_RAY=$(python -c 'import ray, os; print(os.path.dirname(ray.__file__))')
log "Target:       $SITE_RAY"

[ -d "$SITE_RAY/data" ] || fail "no ray/data in $SITE_RAY -- install ray[data] first"

# ------------------------------------------------------------------- back up
BACKUP="$SITE_RAY/data.stock-$REQUIRED_RAY_VERSION"
if [ ! -d "$BACKUP" ]; then
  log "Backing up stock ray/data -> $(basename "$BACKUP")"
  cp -a "$SITE_RAY/data" "$BACKUP"
else
  log "Stock backup already present; restoring before re-overlay"
  rm -rf "$SITE_RAY/data"
  cp -a "$BACKUP" "$SITE_RAY/data"
fi

# ------------------------------------------------------------------- overlay
log "Overlaying Ray Data's ray/data"
# --delete would remove files the wheel ships but the source tree does not
# (e.g. compiled artifacts), so copy over the top instead.
cp -a "$RAY_DATA_SRC/python/ray/data/." "$SITE_RAY/data/"
find "$SITE_RAY/data" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true


# ------------------------------------------------------- read_images(transform=)
# ray_data_e2e_training.py (Figure 8a) calls read_images(transform=...), which
# upstream Ray does not have. See patches/README.md.
PATCH="$(dirname "$0")/../patches/ray-image-transform.patch"
if [ -f "$PATCH" ]; then
  if python -c "import inspect,ray.data,sys; sys.exit(0 if 'transform' in inspect.signature(ray.data.read_images).parameters else 1)" 2>/dev/null; then
    log "read_images already accepts transform="
  else
    log "Applying $(basename "$PATCH")"
    ( cd "$SITE_RAY/.." && patch -p3 --forward --silent < "$PATCH" ) \
      && log "  applied" || log "  WARNING: patch did not apply; Figure 8a will not run"
  fi
fi

# ------------------------------------------------------------------- verify
echo
log "Verifying installation..."
python - <<'PY'
import sys

try:
    import ray
    from ray.data import DataContext
except Exception as exc:
    sys.exit(f"FAILED: cannot import ray.data after overlay: {exc}")

ctx = DataContext.get_current()

# Every knob the artifact's experiments depend on. A missing knob means the
# overlay landed on the wrong branch or did not land at all; DataContext is a
# plain dataclass, so setting an absent attribute would silently do nothing.
required = {
    "op_resource_reservation_enabled": "memory budget (Alg. 2)",
    "op_resource_reservation_ratio":   "optimistic/pessimistic policy (§4.3.1)",
    "scheduling_policy":               "policy selector",
    "microbatch_size":                 "Spark Streaming / Drizzle emulation",
    "microbatch_group_size":           "Drizzle group scheduling",
    "microbatch_stage_barrier":        "Drizzle pre-scheduling",
    "llf_latency_target":              "Cameo LLF emulation (§2.2)",
    "llf_disable_admission_control":   "Cameo admission-control ablation",
}

missing = [(k, why) for k, why in required.items() if not hasattr(ctx, k)]
print(f"    ray version: {ray.__version__}")
for key, why in required.items():
    mark = "missing" if not hasattr(ctx, key) else "ok"
    print(f"    [{mark:>7}] {key:<32} {why}")

if missing:
    sys.exit(
        "\nFAILED: %d scheduling knob(s) missing. The overlay did not apply, "
        "or the checkout is not on the nsdi27-artifact branch." % len(missing)
    )
print("\n    All scheduling knobs present.")
PY

echo
echo "Ray Data installed. Next:"
echo "  python -m ray_data_ae.doctor                       # environment check"
echo "  python -m ray_data_ae.run fig09_memory --scale smoke   # ~2 min"
echo
