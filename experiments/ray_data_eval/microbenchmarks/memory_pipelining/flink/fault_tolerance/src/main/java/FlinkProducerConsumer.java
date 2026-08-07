package com.example;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.io.FileWriter;
import java.io.IOException;
import java.time.Instant;

public class FlinkProducerConsumer {

    private static final int NUM_CPUS = 8;
    private static final int PRODUCER_PARALLELISM = 1;
    private static final int CONSUMER_PARALLELISM = 1;
    private static final int NUM_TASKS = 16 * 5 * 10;
    private static final int TIME_UNIT = 100; // in milliseconds

    private static void appendDictToFile(String data, String filePath) {
        try (FileWriter fileWriter = new FileWriter(filePath, true)) {
            fileWriter.write(data + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Busy loop simulation
    private static void busyLoop(long durationInMillis) {
        long endTime = System.currentTimeMillis() + durationInMillis;
        while (System.currentTimeMillis() < endTime) {
            // Busy waiting
        }
    }

    public class Producer extends RichFlatMapFunction<Long, Long> {
        private transient ValueState<Integer> count;
        private transient Integer attemptNumber;

        @Override
        public void open(org.apache.flink.configuration.Configuration parameters) throws Exception {
            // Initialize state
            ValueStateDescriptor<Integer> stateDescriptor = new ValueStateDescriptor<>("count_state", Integer.class);
            count = getRuntimeContext().getState(stateDescriptor);

            ValueStateDescriptor<Integer> attemptNumberStateDescriptor = new ValueStateDescriptor<>("attempt_number_state", Integer.class);
            attemptNumber = getRuntimeContext().getAttemptNumber();
        }

        @Override
        public void flatMap(Long value, Collector<Long> out) throws Exception {
            count.update(count.value() == null ? 1 : count.value() + 1);

            long producerStart = System.currentTimeMillis();
            busyLoop(TIME_UNIT * 3); // Simulate work
            long producerEnd = System.currentTimeMillis();

            System.out.println("Current count value: " + (count.value() != null ? count.value() : "null"));
            System.out.println("Incoming value: " + value);

            // Log processing details
            String log = String.format("{\"cat\": \"producer:%d\", \"name\": \"producer:%d\", \"ts\": \"%d\", \"dur\": \"%d\", \"ph\": \"X\", \"args\": {}}",
                    getRuntimeContext().getIndexOfThisSubtask(), getRuntimeContext().getIndexOfThisSubtask(), producerStart, (producerEnd - producerStart));
            appendDictToFile(log, "flink_logs.log");

            // Emit values
            for (int i = 0; i < NUM_TASKS / 20; i++) {
                out.collect(i + value * NUM_TASKS / 20);
            }

            if (count.value() != null && count.value() == NUM_TASKS / 20 && attemptNumber == 0) {
                throw new RuntimeException("Simulated failure in Producer");
            }
        }
    }

    public class ConsumerActor extends ProcessFunction<Long, Long> {
        private List<Long> currentBatch = new ArrayList<>();
        private int idx = 0;
        private long consumerStart;

        @Override
        public void open(org.apache.flink.configuration.Configuration parameters) {
            // Initialization of consumer
        }

        @Override
        public void processElement(Long value, Context ctx, Collector<Long> out) throws Exception {
            if (currentBatch.isEmpty()) {
                consumerStart = System.currentTimeMillis();
            }

            busyLoop(TIME_UNIT); // Simulate work
            currentBatch.add(value);
            idx++;

            if (currentBatch.size() < NUM_TASKS / 20) {
                return;
            }

            long consumerEnd = System.currentTimeMillis();
            String log = String.format("{\"cat\": \"consumer:%d\", \"name\": \"consumer:%d\", \"ts\": \"%d\", \"dur\": \"%d\", \"ph\": \"X\", \"args\": {}}",
                    getRuntimeContext().getIndexOfThisSubtask(), getRuntimeContext().getIndexOfThisSubtask(), consumerStart, (consumerEnd - consumerStart));
            appendDictToFile(log, "flink_logs.log");

            // Output the batch
            out.collect(value);
            currentBatch.clear();
        }
    }

    public void runFlink(StreamExecutionEnvironment env) throws Exception {
        // Configure source: using a sequence of numbers as input
        DataStream<Long> numberSource = env.fromSequence(0, NUM_TASKS);

        // Set parallelism for source
        numberSource.keyBy(value -> value % PRODUCER_PARALLELISM)
                    .flatMap(new Producer()) // Producer function
                    .process(new ConsumerActor()) // Consumer function
                    .setParallelism(CONSUMER_PARALLELISM)
                    .print();

        // Execute the Flink job
        env.execute("Flink Producer and Consumer with Checkpointing");
    }

    public void runExperiment() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        env.enableCheckpointing(10000, CheckpointingMode.EXACTLY_ONCE);
        env.getCheckpointConfig().setCheckpointTimeout(120000); // 120 seconds
        env.getCheckpointConfig().setMaxConcurrentCheckpoints(1);

        env.setParallelism(NUM_CPUS);
        env.setRestartStrategy(org.apache.flink.api.common.restartstrategy.RestartStrategies.fixedDelayRestart(3, org.apache.flink.api.common.time.Time.seconds(3)));

        env.getCheckpointConfig().setCheckpointStorage("file:///home/ubuntu/ray-data-eval/ray_data_eval/microbenchmarks/flink/flink-checkpoints");

        runFlink(env);
    }

    public static void main(String[] args) throws Exception {
        FlinkProducerConsumer flinkProducerConsumer = new FlinkProducerConsumer();
        flinkProducerConsumer.runExperiment();
    }
}
