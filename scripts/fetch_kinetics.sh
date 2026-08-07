#!/usr/bin/env bash
# Fetch the Kinetics-700-2020 test split for Figures 7b and 7c.
#
# The dataset is 64,535 videos, 137.3 GB. Kinetics is distributed as a list of
# YouTube URLs rather than as video. We have therefore provided a copy of the 
# video files in an S3 bucket, along with the video-ID manifest and checksums.
#
#   scripts/fetch_kinetics.sh                 # frozen copy (requester pays)
#   scripts/fetch_kinetics.sh --verify-only   # just check an existing copy
#   scripts/fetch_kinetics.sh --subset 6500   # 10%, enough for --scale small
#
# S3 egress is free if you run in us-west-2. Otherwise, it costs roughly $12.
# You can either stream straight from S3 with no local copy, or use this script
# to download the data to local disk.

set -euo pipefail

BUCKET="s3://ray-data-eval-us-west-2/kinetics/k700-2020/test"
DEST="${KINETICS_DIR:-$HOME/kinetics/k700-2020/test}"
MANIFEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/kinetics-test-manifest.txt"
SUBSET=""
VERIFY_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --verify-only) VERIFY_ONLY=1; shift ;;
    --subset) SUBSET="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v aws >/dev/null || { echo "ERROR: aws CLI not found." >&2; exit 1; }

echo
echo "Kinetics-700-2020 test split"
echo "  source: $BUCKET (requester pays)"
echo "  dest:   $DEST"

if [ "$VERIFY_ONLY" -eq 0 ]; then
  mkdir -p "$DEST"
  if [ -n "$SUBSET" ]; then
    echo "  subset: first $SUBSET videos"
    [ -f "$MANIFEST" ] || { echo "ERROR: manifest missing: $MANIFEST" >&2; exit 1; }
    head -n "$SUBSET" "$MANIFEST" | while read -r key; do
      aws s3 cp --request-payer requester "$BUCKET/$key" "$DEST/$key" --quiet
    done
  else
    echo "  full split: ~137 GB, expect 1-3 hours"
    aws s3 sync --request-payer requester "$BUCKET" "$DEST"
  fi
fi

# ------------------------------------------------------------------ verify
if [ -f "$MANIFEST" ]; then
  expected=$(wc -l < "$MANIFEST" | tr -d ' ')
  actual=$(find "$DEST" -name '*.mp4' | wc -l | tr -d ' ')
  echo
  echo "  manifest: $expected videos"
  echo "  on disk:  $actual videos"
  if [ "$actual" -lt "$expected" ]; then
    echo
    echo "  NOTE: a partial split changes absolute throughput but not the"
    echo "  relative ordering of systems. Record the count you used -- the"
    echo "  experiment scripts write it into every result file."
  fi
else
  echo "  (no manifest at $MANIFEST; skipping verification)"
fi

echo
echo "Point the experiments at it with:  export KINETICS_DIR=$DEST"
echo "Or stream from S3 directly, which is the default -- no download needed."
echo
