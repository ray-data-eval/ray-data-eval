import json
import os
import resource
import time

import psutil


# Overridable from the environment.
def _env_int(name, default):
    return int(os.environ.get(name, default))


MB = 1024 * 1024
GB = 1024 * MB
TIME_UNIT = float(os.environ.get("TIME_UNIT", 0.5))
NUM_CPUS = _env_int("NUM_CPUS", 8)
NUM_GPUS = _env_int("NUM_GPUS", 4)
FRAMES_PER_VIDEO = _env_int("FRAMES_PER_VIDEO", 5)
NUM_VIDEOS = _env_int("NUM_VIDEOS", 20)
NUM_FRAMES_TOTAL = FRAMES_PER_VIDEO * NUM_VIDEOS
FRAME_SIZE_B = _env_int("FRAME_SIZE_MB", 100) * MB
EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "process")


def busy_loop(time_in_s):
    end_time = time.time() + time_in_s
    while time.time() < end_time:
        pass


def limit_cpu_memory(mem_limit):
    mem_limit_bytes = mem_limit * GB
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))


def log_memory_usage(mem_limit):
    process_mem = psutil.Process(os.getpid()).memory_info().rss / MB
    vm = psutil.virtual_memory()
    print(f"  Process Memory Usage: {process_mem:.2f} MB")
    print(f"  Total Memory: {vm.total / MB:.2f} MB")
    print(f"  Used Memory: {vm.used / MB:.2f} MB ({vm.percent:.2f}%)")
    print(f"  Available Memory: {vm.available / MB:.2f} MB\n")
    if vm.used / MB > mem_limit * 1024:
        print("Memory exceeded!")


def log_memory_usage_process(interval=2, mem_limit=10):
    try:
        while True:
            log_memory_usage(mem_limit)
            time.sleep(interval)
    except KeyboardInterrupt:
        return


def append_dict_to_file(data: dict, file_path: str):
    with open(file_path, "a") as f:
        f.write(json.dumps(data) + "\n")
