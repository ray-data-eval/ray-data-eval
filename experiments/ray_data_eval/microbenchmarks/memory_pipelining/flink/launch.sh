#!/usr/bin/env bash
# Flink baseline for Figure 9: the Java job in java/ on a standalone Flink
# 1.20.0 cluster, TaskManager memory = limit - 2 GB.
#
#   bash flink/launch.sh <mem-limit-gb> [log-file]
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
cd "$(dirname "${BASH_SOURCE[0]}")"
[ -f "$ROOT/third_party/env.sh" ] && . "$ROOT/third_party/env.sh"
FLINK_HOME="$ROOT/third_party/flink-1.20.0"
mem_limit=${1:?mem limit in GB}
log_file=${2:-mem-limit-$mem_limit.log}
JAR=java/target/your-project-1.0-SNAPSHOT.jar

[ -f "$JAR" ] || (cd java && mvn -q clean package -DskipTests)
tm=$((mem_limit - 2)); [ "$tm" -lt 1 ] && tm=1
[ -f "$FLINK_HOME/conf/config.yaml.orig" ] || cp "$FLINK_HOME/conf/config.yaml" "$FLINK_HOME/conf/config.yaml.orig"
cat > "$FLINK_HOME/conf/config.yaml" <<CONF
jobmanager.rpc.address: localhost
jobmanager.bind-host: localhost
jobmanager.memory.process.size: 1600m
taskmanager.bind-host: localhost
taskmanager.host: localhost
taskmanager.memory.process.size: ${tm}g
taskmanager.numberOfTaskSlots: 8
parallelism.default: 1
rest.address: localhost
rest.bind-address: localhost
CONF
opts=$(grep -E '^\s+all: --add-exports' "$FLINK_HOME/conf/config.yaml.orig" | sed -E 's/^\s+all: //')
[ -n "$opts" ] && echo "env.java.opts.all: $opts" >> "$FLINK_HOME/conf/config.yaml"

echo "Running $mem_limit GB (TaskManager ${tm} GB) -> $log_file"
"$FLINK_HOME/bin/stop-cluster.sh" >/dev/null 2>&1
"$FLINK_HOME/bin/start-cluster.sh" > "$log_file" 2>&1
sleep 5
timeout 2400 "$FLINK_HOME/bin/flink" run "$JAR" >> "$log_file" 2>&1
"$FLINK_HOME/bin/stop-cluster.sh" >> "$log_file" 2>&1
ms=$(grep -aoE 'Flink job execution time: [0-9]+' "$log_file" | grep -oE '[0-9]+' | tail -1)
[ -n "$ms" ] && echo "Run time: $(python3 -c "print(round($ms/1000, 4))")s" | tee -a "$log_file"
