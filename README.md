# Ray Data — NSDI '27 Artifact (paper #193)

This repo contains the code and experiment artifacts for the paper "Ray Data: A dynamic and scalable system for heterogeneous data processing" submitted to NSDI 2027.
If you have any questions, please reach out to us on HotCRP or raise an issue here on GitHub!

## Quick start: Regenerate figures from archived results

Because many experiments require GPU or cluster resources, we have
provided results from our own runs in `results-archive/`.

To regenerate the figures from these archived results, run:

```bash
pip install -r env/requirements.txt
python plots/all.py
```

You can also append `--results results` to the command to plot your own runs instead.

## Set up your node for running experiments

To set up the environment, run the following commands on each node:

```bash
conda create -n raydata python=3.11 -y && conda activate raydata
pip install -r env/requirements-video.txt
bash scripts/setup/install_ray_data.sh
```

Figures 8a and 7a need their own environments: `env/requirements-training.txt`
and `env/requirements-rag.txt`. TensorFlow 2.16.1 pins `numpy<2`, which the
video experiments cannot use, so they cannot share one.

**Note:** Retrieval-augmented generation needs separate dependencies and is set up in its own environment; please see its section below.

For the experiments that span multiple nodes, you will need to start a Ray cluster. To do this, run the commands below:

```bash
ray start --head --disable-usage-stats      # head node
ray start --address=<head-ip>:6379          # each worker
```

---

## Retrieval-augmented generation (Figure 7a)

This experiment can be completed on **a single node with 8x H200 GPUs and 256 vCPUs**. Because it requires vLLM and corresponding dependencies, we set up a separate environment:

```bash
conda create -n raydata-rag python=3.11 -y && conda activate raydata-rag
pip install -r env/requirements-rag.txt
pip install "ray[data]==2.44.1"
```

After the environment is set up, run the following commands to download TriviaQA, build the knowledge base, and cache the model:

```bash
curl -LO http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz
tar xzf triviaqa-rc.tar.gz                       # gives qa/web-train.json

pip install faiss-gpu-cu12==1.8.0.2              # for the build only
python experiments/ray_data_eval/rag/build_kb_triviaqa_v2.py \
    --dataset qa/web-train.json --output-prefix kb --nlist 8192
pip uninstall -y faiss-gpu-cu12
pip install faiss-cpu==1.8.0                     # for the benchmark

huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
```

Then run the experiment:

```bash
bash scripts/run_fig7a.sh
```

This runs Ray Data-Dynamic at 1, 2, 4 and 8 GPUs and the staged baseline at 1.
Results are saved to
`results/rag/<timestamp>-<mode>-dp<N>-nprobe<N>-<prompts>/log.log`. `DATASET`,
`KB` and `MODEL` override the paths.

## Video classification (Figure 7b)

This experiment needs **4 GPU nodes** (1x g5.4xlarge and 3x g5.2xlarge). The
dataset is read from S3, so every node needs AWS credentials, and the model must
be cached on every node first:

```bash
python scripts/setup/warmup_models.py             # on every node
export RAY_DEDUP_LOGS=0
```

Then run all four systems. They are the same benchmark and differ only in
environment variables:

```bash
bash scripts/run_fig7b.sh
```

The script writes `results/video_classification/<system>.csv`. Each run takes
about an hour, so allow half a day.

## Fault tolerance (Figure 7c)

This experiment uses the same workload as before. However, here we inject
executor and node failures during the run. It needs **2 nodes**: 1 g5.xlarge
(4 vCPU, 1 GPU) and 1 m7i.2xlarge (8 vCPU, no GPU). Cache the model on both
first, then run the commands below on the GPU node; `<cpu-node>` is an ssh
target for the CPU-only node, which the GPU node must be able to reach.

```bash
# Executor failure: kills one worker process at t=15 min
bash scripts/run_fig7c.sh executor <cpu-node> <head-ip>

# Node failure: disconnects the CPU-only node at t=15 min, rejoins it at t=30 min
bash scripts/run_fig7c.sh node <cpu-node> <head-ip>
```

Each run takes about an hour. Set `CONDA_ENV` if the environment on the CPU
node is not called `raydata`.

The dashed curves in the figure are Ray Data modified to emulate the global
checkpointing used by systems with static task assignment, taking an empty
checkpoint every 6 minutes. There is no checkpointing code here; the curves come
from restarting the job at each rollback and recording the restarts as separate
segments, which the plot stitches together:
`node_failure_ckpt_seg{0,1,2}.csv` and `executor_failure_ckpt_seg{0,1}.csv`.

The archived results were produced before we fixed two bugs in this benchmark
that held throughput down, so a fresh run is faster than the figure in the
submitted paper. Use the regenerated results in `results-archive/fault_tolerance/`.

## ResNet-50 training (Figure 8a)

This experiment needs **1 GPU node**. ImageNet cannot be redistributed;
`scripts/setup/fetch_imagenet.sh` has download instructions, or you can generate a
substitute dataset:

```bash
python scripts/setup/make_synthetic_images.py --out /tmp/imagenet-synth --count 2000 --classes 10
```

Then run both systems:

```bash
bash scripts/run_fig8a.sh /tmp/imagenet-synth
```

The figure has four series: each system reading from a local path and from S3.
The command above produces the `_local` pair; pass an `s3://` path for the `_s3`
pair. The script names the CSVs `ray_data_<series>.csv` and
`tfdata_<series>.csv`, which is what `plots/resnet_training.py` reads.

## Stable Diffusion (Figure 8b)

This experiment is vendored in `experiments/ray-data-diffusion/`, with its own
dependencies, config format and instructions. Nothing above applies to it. Its
`requirements.txt` has five packages appended, without which it does not
install: `setuptools`, a numpy pin, `torchvision`, `ray` and `wandb`. It also
needs a third torch version (2.1.0), so give it its own environment:

```bash
cd experiments/ray-data-diffusion && cat README.md
```

## Memory-aware pipelining (Figure 9)

This experiment needs **1 m6i.2xlarge** (8 vCPU, 32 GB), with no GPU and no
cluster. It uses a different build of Ray Data from every other experiment, so
set up a separate environment:

```bash
conda create -n raydata-fig9 python=3.11 -y && conda activate raydata-fig9
pip install -r env/requirements.txt
bash scripts/setup/install_ray_data_fig9.sh
```

Then run Ray Data and the baselines:

```bash
# Ray Data: sweeps the memory limit, about 35 minutes
bash scripts/run_fig9.sh

# Baselines
cd experiments/ray_data_eval/microbenchmarks/memory_pipelining
bash spark/launch.sh
bash spark_streaming/launch.sh
bash flink/launch.sh
bash tfdata/launch.sh
```

`run_fig9.sh` writes `results/memory_pipelining/ray_data.csv`. Expect about
199-200 s at 8-16 GB and about 945 s at 6 GB.

Two details of the published configuration: the object store is capped at 12 GB,
so the 14 and 16 GB points run with a 12 GB store; and below 8 GB the benchmark
drops to 2 CPUs and 2 GPUs, which is why 6 GB takes five times as long rather
than modestly longer.

### Why this figure needs its own environment

The memory-aware scheduling policy this figure measures and the baseline
emulations used by Figures 7b and 7c live on two branches of
`ray-data-eval/ray`, built on different Ray versions, so one environment cannot
hold both:

| branch | provides | Ray version |
|---|---|---|
| `nsdi27-artifact` | `scheduling_policy`, `microbatch_*`, `llf_*` | 2.40.0 release |
| `nsdi27-fig9` | `is_budget_policy`, `is_conservative_policy` | master, Jul 2024 |

Running this figure against the main environment does not fail. It reports about
265 s at every memory limit instead of 199 s, because `is_budget_policy` does
not exist there and `DataContext` silently accepts an attribute it does not
define. The installer verifies the settings to catch exactly this.

## Partition size (Figure 10a)

This experiment needs **8 CPU cores** and **15 GB free in `/dev/shm`**, with
no GPU and no cluster. A Ray left over from an earlier experiment holds its
object store there; `ray stop --force && rm -rf /tmp/ray/session_*` frees it.
One invocation sweeps every partition size and takes about 20 minutes:

```bash
bash scripts/run_fig10a.sh
```

`PARTITION_NUM_ROWS=2048 PARTITION_SIZES=1,64,1024` runs a short version.

## Scalability (Figure 10b)

This experiment needs **a cluster**, and throughput is recorded against node
count, so run it, add nodes, and run it again. The dataset is
`128 x size x num_nodes` items, so `--num_nodes` must match the cluster or the
points are not comparable:

```bash
bash scripts/run_fig10b.sh <N>
```

The script runs Ray Data, Ray Core and Ray Core streaming, and appends a row per
node count to each CSV, so re-running it after adding nodes builds up the
series.

---

## Plotting your own results

The `scripts/run_fig*.sh` scripts write into `results/<experiment>/` already;
pass `--results results` to the matching script in `plots/`. The filenames each
figure expects are:

| figure | directory | files |
|---|---|---|
| 7a | `rag` | `ray_data_dynamic.csv`, `ray_data_staged.csv` |
| 7b | `video_classification` | `ray_data_dynamic.csv`, `ray_data_static.csv`, `ray_data_staged.csv`, `ray_data_microbatch.csv`, `cameo_llf.csv`, `flink.csv`, `spark.csv` |
| 7c | `fault_tolerance` | `node_failure.csv`, `executor_failure.csv`, and the `_ckpt_seg*` variants |
| 8a | `resnet_training` | `ray_data_local.csv`, `ray_data_s3.csv`, `tfdata_local.csv`, `tfdata_s3.csv`, `gpu_busy_time.csv` |
| 9 | `memory_pipelining` | `ray_data.csv`, `ray_data_no_part.csv`, `ray_data_no_adapt.csv`, `spark.csv`, `flink.csv`, `tfdata.csv` |
| 10a | `partitioning` | `ray_data.csv` |
| 10b | `scalability` | `ray_data.csv`, `ray.csv`, `ray_streaming.csv` |

A figure regenerates from whichever of these files are present, so a partial
set still plots.

## Common failures

| Symptom | Cause |
|---|---|
| `DataContext has no attribute ...` | Ray Data not installed on that node |
| `read_images() got an unexpected keyword argument 'transform'` | `patches/ray-image-transform.patch` not applied |
| `Repo id must be in the form ...` | model cache missing on the node the actor landed on; run `scripts/setup/warmup_models.py` there |
| `configured object store size exceeds /dev/shm` | `sudo mount -o remount,size=24G /dev/shm` |
| `File ... .out does not exist` | benchmark output was not redirected to the expected filename |
| Run succeeds but no CSV | `RAY_DEDUP_LOGS=0` was not set |
| `Timed out while starting actors` (Figure 7a) | too few vCPU; the encoder and retriever pools need 48 before vLLM asks for any |
