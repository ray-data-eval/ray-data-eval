import os
import time

# Overridable from the environment.
#
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
