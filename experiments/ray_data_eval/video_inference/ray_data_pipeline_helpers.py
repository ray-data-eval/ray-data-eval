import boto3
import csv
import json
import os
import ray
import re
import time


class ChromeTracer:
    """
    A simple custom profiler that records event start and end time, to replace Ray's profiler due to observed issues with profiling gpu workload.
    https://github.com/ray-project/ray/blob/master/python/ray/_private/profiling.py#L84

    """

    def __init__(self, log_file, device_name="NVIDIA_A10G"):
        self.log_file = log_file
        self.device_name = device_name
        self.events = []

    def _add_event(self, name, phase, timestamp, cname="rail_load", extra_data=None):
        event = {
            "name": name,
            "ph": phase,
            "ts": timestamp,
            "pid": ray._private.services.get_node_ip_address(),
            "tid": "gpu:" + "NVIDIA_A10G",
            "cname": cname,
            "args": extra_data or {},
        }
        self.events.append(event)

    def __enter__(self):
        self.start_time = time.time() * 1000000
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time() * 1000000
        self._add_event(self.name, "B", self.start_time, self.extra_data)
        self._add_event(self.name, "E", self.end_time)

    def profile(self, name, extra_data=None):
        self.name = name
        self.extra_data = extra_data
        return self

    def save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.events, f)


def append_gpu_timeline(timeline_filename, gpu_timeline_filename):
    """
    Append GPU events log to the main log.

    """
    try:
        with open(timeline_filename, "r") as file:
            timeline = json.load(file)
            assert isinstance(timeline, list)

        with open(gpu_timeline_filename, "r") as gpu_file:
            gpu_timeline = json.load(gpu_file)
            assert isinstance(gpu_timeline, list)

        timeline += gpu_timeline

        with open(timeline_filename, "w") as file:
            json.dump(timeline, file)
    except Exception as e:
        print(f"Error occurred when appending GPU timeline: {e}")


def get_prefixes(bucket_name, prefix):
    """
    Each bucket_name, prefix combination creates a path that leads to a folder,
    which contains training data of the same label.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    response_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter="/")

    prefixes = []
    for response in response_iterator:
        if "CommonPrefixes" in response:
            prefixes.extend(
                [f"s3://{bucket_name}/" + cp["Prefix"] for cp in response["CommonPrefixes"]]
            )

    return prefixes, len(prefixes)


def postprocess(logging_file):
    if not os.path.exists(logging_file):
        print(f"File {logging_file} does not exist.")
        return
    start_time_pattern = r"\[Start Time\] (\d+\.\d+)"
    batch_pattern = r"\[Completed Batch\] (\d+\.\d+) (\d+)"

    start_time = None
    batch_finish_times = []

    with open(logging_file, "r") as f:
        for line in f:
            if not start_time:
                match = re.search(start_time_pattern, line)
                if match:
                    start_time = float(match.group(1))
                    print(f"Found start time: {start_time}")
            else:
                match = re.search(batch_pattern, line)
                if match:
                    timestamp = float(match.group(1))
                    batch_size = int(match.group(2))

                    elapsed_time = timestamp - start_time
                    if elapsed_time > 0:
                        batch_finish_times.append((elapsed_time, batch_size))
                    print(f"Found batch completion: Completed {batch_size} at {elapsed_time}")

    batch_finish_times = sorted(batch_finish_times, key=lambda x: x[0])

    accumulated_size = 0
    results = []
    for elapsed_time, batch_size in batch_finish_times:
        accumulated_size += batch_size
        results.append((elapsed_time, accumulated_size))

    csv_filename = logging_file.replace(".out", ".csv")
    csv_ref = open(csv_filename, "w")
    print(f"Created csv file: {csv_filename}")
    writer = csv.writer(csv_ref)
    writer.writerow(["time_from_start", "number_of_rows_finished"])
    writer.writerow([0, 0])
    for result in results:
        writer.writerow(result)
    csv_ref.close()
    return


def download_train_directories(
    bucket_name,
    prefix,
    percentage=1,
    output_file="kinetics-train-1-percent.txt",
):
    directories, count = get_prefixes(bucket_name, prefix)
    num_samples = len(directories) * percentage // 100
    directories = directories[:num_samples]

    with open(output_file, "w") as f:
        f.write(repr(directories))
    print(f"Downloaded {num_samples} directories to {output_file}")
    return directories


if __name__ == "__main__":
    bucket_name = "ray-data-eval-us-west-2"
    prefix = "kinetics/k700-2020/train/"
    print(download_train_directories(bucket_name, prefix)[0])
