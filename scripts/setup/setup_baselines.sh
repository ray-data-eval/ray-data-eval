#!/usr/bin/env bash
# Install the Spark and Flink baselines, used by Figures 7b and 9.
#
# Installed into third_party/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
SPARK_VERSION=3.5.1
FLINK_VERSION=1.19.0

mkdir -p "$THIRD_PARTY"
cd "$THIRD_PARTY"

echo
echo "Baseline setup"
echo "=============="

if ! command -v java >/dev/null; then
  echo "ERROR: no JVM found. Install one first, e.g." >&2
  echo "  sudo apt-get install -y openjdk-11-jdk" >&2
  exit 1
fi
echo "  java: $(java -version 2>&1 | head -1)"

# ------------------------------------------------------------------- Spark
if [ ! -d "spark-${SPARK_VERSION}-bin-hadoop3" ]; then
  echo "  downloading Spark ${SPARK_VERSION}"
  curl -fsSL "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz" \
    | tar xz
else
  echo "  Spark ${SPARK_VERSION} already present"
fi

# ------------------------------------------------------------------- Flink
if [ ! -d "flink-${FLINK_VERSION}" ]; then
  echo "  downloading Flink ${FLINK_VERSION}"
  curl -fsSL "https://archive.apache.org/dist/flink/flink-${FLINK_VERSION}/flink-${FLINK_VERSION}-bin-scala_2.12.tgz" \
    | tar xz
else
  echo "  Flink ${FLINK_VERSION} already present"
fi

# ------------------------------------------------------- Python baselines
# Installed into the current environment. If pip conflicts with Ray, put
# TensorFlow in its own venv.
echo "  installing pyspark, apache-flink, tensorflow"
pip install --quiet "pyspark==${SPARK_VERSION}" "apache-flink==${FLINK_VERSION}" \
  "tensorflow==2.16.1" || {
    echo
    echo "  pip hit a conflict. The baselines run as separate processes, so a" >&2
    echo "  separate environment is fine:" >&2
    echo "    python -m venv .venv-baselines && . .venv-baselines/bin/activate" >&2
    echo "    pip install pyspark==${SPARK_VERSION} apache-flink==${FLINK_VERSION} tensorflow==2.16.1" >&2
    exit 1
  }

cat > "$ROOT/third_party/env.sh" <<ENVEOF
# Source this before running the Spark or Flink baselines.
export SPARK_HOME="$THIRD_PARTY/spark-${SPARK_VERSION}-bin-hadoop3"
export FLINK_HOME="$THIRD_PARTY/flink-${FLINK_VERSION}"
export PYSPARK_PYTHON="\$(command -v python)"
ENVEOF

echo
echo "Done. Before running a JVM baseline:"
echo "  source third_party/env.sh"
echo
echo "Then, from experiments/ray_data_eval/microbenchmarks/memory_pipelining:"
echo "  bash spark/launch.sh"
echo "  bash spark_streaming/launch.sh"
echo "  bash flink/launch.sh"
echo "  bash tfdata/launch.sh"
echo
