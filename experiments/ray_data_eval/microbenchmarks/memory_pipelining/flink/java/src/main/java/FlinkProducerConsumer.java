package com.example;

import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.util.Collector;

public class FlinkProducerConsumer {

    private static final int NUM_CPUS = 8;
    private static final int NUM_GPUS = 4;

    private static final int PRODUCER_PARALLELISM = 4;
    private static final int CONSUMER_PARALLELISM = 4;
    private static final int NUM_VIDEOS = 160;
    private static final long TIME_UNIT = 500; // in milliseconds
    private static final int FRAMES_PER_VIDEO = 5;
    private static final int FRAME_SIZE_B = 100 * 1024 * 1024; // Adjusted to 100 MB for memory management

    // Busy loop simulation
    private static void busyLoop(long durationInMillis) {
        long endTime = System.currentTimeMillis() + durationInMillis;
        while (System.currentTimeMillis() < endTime) {
            // Busy waiting
        }
    }

    public static class Producer extends RichFlatMapFunction<Long, byte[]> {
        @Override
        public void flatMap(Long value, Collector<byte[]> out) throws Exception {
            System.out.println("Producing video " + value); // Log video production
            busyLoop(TIME_UNIT * 10); // Simulate work

            // Emit frames for a video
            for (int i = 0; i < FRAMES_PER_VIDEO; i++) {
                System.out.println("Emitting frame " + (i + 1) + " for video " + value); // Log frame emission
                out.collect(new byte[FRAME_SIZE_B]); // Emit frame
            }
        }
    }

    public static class ConsumerActor extends ProcessFunction<byte[], byte[]> {
        @Override
        public void processElement(byte[] value, Context ctx, Collector<byte[]> out) throws Exception {
            System.out.println("Processing frame of size " + value.length + " bytes"); // Log frame processing
            busyLoop(TIME_UNIT); // Simulate processing
            out.collect(value); // Forward processed frame
        }
    }

    public static class Inference extends ProcessFunction<byte[], Long> {
        @Override
        public void processElement(byte[] value, Context ctx, Collector<Long> out) throws Exception {
            System.out.println("Performing inference on frame of size " + value.length + " bytes"); // Log inference
            busyLoop(TIME_UNIT); // Simulate inference
            out.collect(1L); // Emit inference result (e.g., 1 for successful processing)
        }
    }

    public void runFlink(StreamExecutionEnvironment env) throws Exception {
        // Configure source: using a sequence of numbers as input

        long startTime = System.currentTimeMillis();

        DataStream<Long> numberSource = env.fromSequence(0, NUM_VIDEOS);

        // Set parallelism and chain transformations
        numberSource
                .flatMap(new Producer())
                .setParallelism(6)
                .process(new ConsumerActor())
                .setParallelism(6)
                .process(new Inference())
                .setParallelism(4)
                .print(); // Printing the results

        // Execute the Flink job
        env.execute("Flink Producer and Consumer with Checkpointing");

        // Capture the end time
        long endTime = System.currentTimeMillis();

        // Calculate and print the time taken for the job
        long timeTaken = endTime - startTime;

        System.out.println("Flink job execution time: " + timeTaken + " milliseconds");
    }

    public void runExperiment() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        runFlink(env);
    }

    public static void main(String[] args) throws Exception {
        FlinkProducerConsumer flinkProducerConsumer = new FlinkProducerConsumer();
        flinkProducerConsumer.runExperiment();
    }
}
