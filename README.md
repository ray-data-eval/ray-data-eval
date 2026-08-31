# Ray Data: A dynamic and scalable system for heterogeneous data processing

This repo contains the artifacts for our paper "Ray Data: A dynamic and scalable system for heterogeneous data processing" accepted to NSDI 2027.

If you have any questions, please feel free to reach out to us on HotCRP or raise an issue here on GitHub!

## Quick start: Regenerate figures from archived results

Because many experiments require GPU or cluster resources, we have provided results from our own runs in `results-archive/`.

To regenerate the figures from these archived results, run:

```bash
pip install -r env/requirements.txt
python plots/all.py
```

Here is a checklist of the figures and the corresponding archived results:

| figure | experiment | archived results | plot script |
|---|---|---|---|
| 7a | Retrieval-augmented generation | `results-archive/rag/` | `plots/rag.py` |
| 7b | Video classification | `results-archive/video_classification/` | `plots/video_classification.py` |
| 7c | Fault tolerance | `results-archive/fault_tolerance/` | `plots/fault_tolerance.py` |
| 8a | ResNet-50 training | `results-archive/resnet_training/` | `plots/resnet_training.py` |
| 9 | Memory-aware pipelining | `results-archive/memory_pipelining/` | `plots/memory_pipelining.py` |
| 10a | Partition size | `results-archive/partitioning/` | `plots/partitioning.py` |
| 10b | Scalability | `results-archive/scalability/` | `plots/scalability.py` |

Figure 8b (Stable Diffusion) instructions can be found separately in `experiments/ray-data-diffusion/`.

To plot your own runs, use `--results <out-dir>`. By default, experiment scripts write their results to `results/<experiment>/`. For example, the fault tolerance experiment script writes `results/fault_tolerance/node_failure.csv`. You are free to change the output directory. To do this, set the `RESULTS_DIR` environment variable to the desired output directory.

```bash
bash scripts/run_fig7c.sh node <cpu-node> <head-ip>   # writes results/fault_tolerance/
python plots/fault_tolerance.py --results results     # reads  results/fault_tolerance/

# or anywhere you like:
RESULTS_DIR=/data/myrun bash scripts/run_fig7c.sh node <cpu-node> <head-ip>
python plots/fault_tolerance.py --results /data/myrun --outdir /data/myfigs
```

## Set up your node for running experiments

### Setting up using public AMI
We have provided a public AMI in `us-west-2` (`ami-0ba8b0aff59c56f24`). This AMI contains the environments needed for reproducing the experiments. To launch an instance, run the following command:

```bash
aws ec2 run-instances --region us-west-2 --image-id ami-0ba8b0aff59c56f24 \
    --instance-type <instance-type> --key-name <your-key> --associate-public-ip-address
```

You can log in as `ubuntu`. For each instance, you also need to run the following commands to clone the repo and set up the environment:

```bash
cd ~/ray-data-eval
git remote set-url origin https://github.com/ray-data-eval/ray-data-eval.git
git pull
conda activate raydata
python scripts/setup/warmup_models.py       # model cache; rerun after any reboot
```

### Setting up from scratch

Here is a checklist of all figures and their corresponding environments:

| figures | environment | dependencies |
|---|---|---|
| 7a | `raydata-rag` | vLLM 0.7.3, torch 2.5.1+cu124, faiss 1.8.0, Ray 2.44.1 |
| 7b, 7c, 10a, 10b | `raydata` | torch 2.8.0+cu128, transformers 5.5.4, Ray 2.40.0 (artifact fork) |
| 8a | `raydata-training` | torch 2.4.0+cu121, TensorFlow 2.16.1, numpy 1.26.4, Ray 2.40.0 (artifact fork) |
| 9 | `raydata-fig9` | Ray fork branch `nsdi27-fig9` |

#### Figures 7b, 7c, 10a, 10b
```bash
conda create -n raydata python=3.11 -y && conda activate raydata
pip install -r env/requirements-video.txt
bash scripts/setup/install_ray_data.sh
python scripts/setup/warmup_models.py
```

#### Figure 8a
```bash
conda create -n raydata-training python=3.11 -y && conda activate raydata-training
pip install -r env/requirements-training.txt
bash scripts/setup/install_ray_data.sh
```

#### Figure 7a
```bash
conda create -n raydata-rag python=3.11 -y && conda activate raydata-rag
pip install -r env/requirements-rag.txt
pip install "ray[data]==2.44.1"
```

#### Figure 9
```bash
conda create -n raydata-fig9 python=3.11 -y && conda activate raydata-fig9
pip install -r env/requirements.txt
bash scripts/setup/install_ray_data_fig9.sh
```

### Starting a Ray cluster

For experiments that span multiple nodes, you need a Ray cluster, whichever way the nodes were set up (AMI or from scratch). Set up the same environment on every node, then:

```bash
conda activate raydata
ray start --head --disable-usage-stats      # head node
ray start --address=<head-ip>:6379          # each worker
```

Every node must run the same environment: Ray refuses to join nodes whose Python or Ray versions differ. Each experiment section names which node to use as the head.

---

## Retrieval-augmented generation (Figure 7a)

This experiment uses **a single node with 8x H200 GPUs and 256 vCPUs**. To fully reproduce the experiment, you can use an AWS p5e.48xlarge instance (8x H200, 192 vCPUs). The experiment should also run on other instances, such as g5.48xlarge (8x A10G, 192 vCPUs).

**Environment.** Check that you have the `raydata-rag` environment. If not, set up the environment by following the instructions in the "Setting up from scratch" section.

**Data and model.** With `raydata-rag` activated, run the following commands to download TriviaQA, build the knowledge base, and cache the model.

Note that to download the model, you need a Hugging Face account with access to [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). You can run `huggingface-cli login` to login to your Hugging Face account.

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

This runs Ray Data-Dynamic at 1, 2, 4 and 8 GPUs and the staged baseline at 1. If you only wish to test on a node with fewer GPUs, set `GPU_COUNT`; for example, `GPU_COUNT=2 bash scripts/run_fig7a.sh` runs only the 1- and 2-GPU points (and the staged baseline).

Results are saved to `results/rag/`.

## Video classification (Figure 7b)

This experiment needs **4 GPU nodes** (1x g5.4xlarge head node and 3x g5.2xlarge worker nodes).

**Environment.** Check that you have the `raydata` environment on every node. If not, set up the environment by following the instructions in the "Setting up from scratch" section.

**Data and model.** The dataset is read from S3, so every node needs AWS credentials.

Cache the model on every node first:

```bash
python scripts/setup/warmup_models.py             # on every node
```

Then, on the head node, run the experiment using the following command. This will run Ray Data-dynamic, -static, -staged, and -microbatch.

```bash
bash scripts/run_fig7b.sh
```

Results are saved to `results/video_classification/`.

## Fault tolerance (Figure 7c)

This experiment uses the same workload as before, but injects executor and node failures during the run. It needs **2 nodes**: 1 g5.xlarge (4 vCPU, 1 GPU) and 1 m7i.2xlarge (8 vCPU, no GPU).

**Environment.** Check that you have the `raydata` environment on both nodes. If not, set up the environment by following the instructions in the "Setting up from scratch" section.

**Data and model.** Similar as before, the dataset is read from S3, so every node needs AWS credentials.

Cache the model on every node first:

```bash
python scripts/setup/warmup_models.py             # on every node
```

To run the experiment, use the commands below on the GPU node. `<cpu-node>` is an ssh destination for the CPU-only node, e.g. `ubuntu@<cpu-node-private-ip>`. `<head-ip>` is the GPU node’s private IP. Because the script uses ssh to disconnect and reconnect the CPU node from the GPU node, we need to set up the appropriate credentials first. To do this, you can copy a private key to the GPU node and add the corresponding public key to the CPU node's `~/.ssh/authorized_keys`. Then, check the connection with `ssh ubuntu@<cpu-node-private-ip> hostname`.

### Executor failure experiment
```bash
# Executor failure: kills one worker process at t=15 min
bash scripts/run_fig7c.sh executor <cpu-node> <head-ip>
```

### Node failure experiment
```bash
# Node failure: disconnects the CPU-only node at t=15 min, rejoins it at t=30 min
bash scripts/run_fig7c.sh node <cpu-node> <head-ip>
```

Please also note that the results in the AE paper copy were produced before we fixed two bugs in this benchmark, so you can expect the throughput to be higher when you reproduce the experiment. However, this does not affect the conclusions of the figure.

Results are saved to `results/fault_tolerance/`.

## ResNet-50 training (Figure 8a)

This experiment uses **1 GPU node**.

**Environment.** Check that you have the `raydata-training` environment. If not, set up the environment by following the instructions in the "Setting up from scratch" section.

**Data.** Because ImageNet cannot be redistributed, we have provided **download instructions** in `scripts/setup/fetch_imagenet.sh`. You can also generate a substitute dataset:

```bash
python scripts/setup/make_synthetic_images.py --out /tmp/imagenet-synth --count 2000 --classes 10
```

To run the experiment:

```bash
bash scripts/run_fig8a.sh /tmp/imagenet-synth
```

Results are saved to `results/resnet_training/`.

## Stable Diffusion pretraining (Figure 8b)

This experiment is in a separate directory with its own dependencies and instructions. Please refer to the instructions in `experiments/ray-data-diffusion/`.

```bash
cd experiments/ray-data-diffusion && cat README.md
```

## Memory-aware pipelining (Figure 9)

This experiment needs **1 m6i.2xlarge** (8 vCPU, 32 GB) node.

**Environment.** Check that you have the `raydata-fig9` environment. If not, set up the environment by following the instructions in the "Setting up from scratch" section.

To run the experiment:

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

Results are saved to `results/memory_pipelining/`.

## Partition size (Figure 10a)

This experiment needs **1 node with 8 CPU cores and at least 15 GB free memory space in `/dev/shm`**.

**Environment.** Check that you have the `raydata` environment. If not, set up the environment by following the instructions in the "Setting up from scratch" section.

To run the experiment:

```bash
bash scripts/run_fig10a.sh
```

Results are saved to `results/partitioning/`. `PARTITION_NUM_ROWS=2048 PARTITION_SIZES=1,64,1024` runs a short version.

## Scalability (Figure 10b)

This experiment needs **a CPU cluster**: 1 m8i.4xlarge (16 vCPU, 64 GiB) head node and up to 32 m8i.2xlarge (8 vCPU, 32 GiB) worker nodes.

**Environment.** Check that you have the `raydata` environment on every node. If not, set up the environment by following the instructions in the "Setting up from scratch" section. Then start a Ray cluster (see "Starting a Ray cluster") with the m8i.4xlarge as the head node.

The recommended workflow is to run the script, add nodes, and run it again. Note that the supplied argument `<N>` must match the current cluster size.

To run the experiment:

```bash
bash scripts/run_fig10b.sh <N>
```

Results are saved to `results/scalability/`.


## Common failures

We will keep this section updated with common failures and their solutions.