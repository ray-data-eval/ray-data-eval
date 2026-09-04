#!/usr/bin/env bash
# Install the Spark and Flink baselines, used by Figures 7b and 9.
#
# Installed into third_party/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
SPARK_VERSION=3.5.1

mkdir -p "$THIRD_PARTY"
cd "$THIRD_PARTY"

echo
echo "Baseline setup"
echo "=============="

JAVA_MAJOR=$(java -version 2>&1 | head -1 | sed -E 's/.*"(1\.)?([0-9]+).*/\2/')
case "${JAVA_MAJOR:-none}" in
  8|11|17) echo "  java: $(java -version 2>&1 | head -1)" ;;
  *)
    echo "ERROR: need Java 8, 11 or 17 (found: $(java -version 2>&1 | head -1)). Install one, e.g." >&2
    echo "  sudo apt-get install -y openjdk-11-jdk" >&2
    echo "  export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64  PATH=\$JAVA_HOME/bin:\$PATH" >&2
    exit 1 ;;
esac

# ------------------------------------------------------------------- Spark
if [ ! -d "spark-${SPARK_VERSION}-bin-hadoop3" ]; then
  echo "  downloading Spark ${SPARK_VERSION}"
  curl -fsSL "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz" \
    | tar xz
else
  echo "  Spark ${SPARK_VERSION} already present"
fi

# ------------------------------------------------------------------- Flink
FLINK_VERSION=1.20.0
if ! command -v mvn >/dev/null; then
  echo "ERROR: no Maven found. Install it first, e.g." >&2
  echo "  sudo apt-get install -y maven" >&2
  exit 1
fi
if [ ! -d "flink-${FLINK_VERSION}" ]; then
  echo "  downloading Flink ${FLINK_VERSION}"
  curl -fsSL "https://archive.apache.org/dist/flink/flink-${FLINK_VERSION}/flink-${FLINK_VERSION}-bin-scala_2.12.tgz" \
    | tar xz
else
  echo "  Flink ${FLINK_VERSION} already present"
fi
(cd "$ROOT/experiments/ray_data_eval/microbenchmarks/memory_pipelining/flink/java" && mvn -q clean package -DskipTests) \
  && echo "  built the Flink job"

# ------------------------------------------------------- Python baselines
# Installed into the current environment. If pip conflicts with Ray, put
# TensorFlow in its own venv.
echo "  installing pyspark, tensorflow"
pip install --quiet "pyspark==${SPARK_VERSION}" \
  "tensorflow==2.16.1" "numpy<2" "setuptools<80" psutil || {
    echo
    echo "  pip hit a conflict. The baselines run as separate processes, so a" >&2
    echo "  separate environment is fine:" >&2
    echo "    python -m venv .venv-baselines && . .venv-baselines/bin/activate" >&2
    echo "    pip install pyspark==${SPARK_VERSION} tensorflow==2.16.1 numpy<2 setuptools<80 psutil" >&2
    exit 1
  }

cat > "$ROOT/third_party/env.sh" <<ENVEOF
# Source this before running the Spark or Flink baselines.
export SPARK_HOME="$THIRD_PARTY/spark-${SPARK_VERSION}-bin-hadoop3"
export FLINK_HOME="$THIRD_PARTY/flink-1.20.0"
export PYSPARK_PYTHON="\$(command -v python)"
ENVEOF

echo
echo "Done. Before running a JVM baseline:"
echo "  source third_party/env.sh"
echo
echo "Then:"
echo "  bash scripts/run_fig9_baselines.sh"
echo
