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


| figure | experiment                     | archived results                        | plot script                     |
| ------ | ------------------------------ | --------------------------------------- | ------------------------------- |
| 7a     | Retrieval-augmented generation | `results-archive/rag/`                  | `plots/rag.py`                  |
| 7b     | Video classification           | `results-archive/video_classification/` | `plots/video_classification.py` |
| 7c     | Fault tolerance                | `results-archive/fault_tolerance/`      | `plots/fault_tolerance.py`      |
| 8a     | ResNet-50 training             | `results-archive/resnet_training/`      | `plots/resnet_training.py`      |
| 9      | Memory-aware pipelining        | `results-archive/memory_pipelining/`    | `plots/memory_pipelining.py`    |
| 10a    | Partition size                 | `results-archive/partitioning/`         | `plots/partitioning.py`         |
| 10b    | Scalability                    | `results-archive/scalability/`          | `plots/scalability.py`          |


Figure 8b (Stable Diffusion) instructions can be found separately in `experiments/ray-data-diffusion/`.

To plot your own runs, append `--results <out-dir>` to the command. By default, results are written to 

```
results/<experiment>/.../
```

For example, the fault tolerance experiment script writes `results/fault_tolerance/node_failure.csv`. 

To change the output directory, set the `RESULTS_DIR` environment variable to the desired location.

```bash
bash scripts/run_fig7c.sh node <cpu-node> <head-ip>   # writes results/fault_tolerance/
python plots/fault_tolerance.py --results results     # reads  results/fault_tolerance/

# or anywhere you like:
RESULTS_DIR=/data/myrun bash scripts/run_fig7c.sh node <cpu-node> <head-ip>
python plots/fault_tolerance.py --results /data/myrun --outdir /data/myfigs
```



## Set up your node for running experiments

Ray Data is a dynamic and scalable system that works with heterogeneous clusters. Please refer to this table to see the node types for each experiment:


| figure | nodes                                      | notes                                                       |
| ------ | ------------------------------------------ | ----------------------------------------------------------- |
| 7a     | 1x `p5e.48xlarge` (8x H200)                | could also run on other GPU instances, see section 7a below |
| 7b     | 1x `g5.4xlarge` + 3x `g5.2xlarge`          | 4-node cluster, g5.4xlarge as head                          |
| 7c     | 1x `g5.xlarge` + 1x `m7i.2xlarge`          | 2-node cluster, g5.xlarge as head                           |
| 8a     | 1x `g5.xlarge`                             | single GPU node                                             |
| 8b     | see `experiments/ray-data-diffusion/`      |                                                             |
| 9      | 1x `m6i.2xlarge`                           | single CPU node                                             |
| 10a    | 1x `m7i.2xlarge`                           | single CPU node                                             |
| 10b    | 1x `m8i.4xlarge` + up to 32x `m8i.2xlarge` | CPU cluster, m8i.4xlarge as head                            |




### Setting up using public AMI

We have provided a public AMI in `us-west-2` (`ami-0ba8b0aff59c56f24`). This AMI contains all environments needed for reproducing the experiments. To launch an instance, run the following command:

```bash
aws ec2 run-instances --region us-west-2 --image-id ami-0ba8b0aff59c56f24 \
    --instance-type <instance-type> --key-name <your-key> \
    --associate-public-ip-address   # so the instance has outbound internet access
    # accounts without a default VPC also need: --subnet-id <subnet> --security-group-ids <sg>
```

For multi-node experiments, the nodes must be able to reach each other (by default, Ray uses port 6379 for the head and many ephemeral worker ports). This can be checked and configured in EC2 security group settings.

Once the instances are up and running, you can log in as `ubuntu`. For each instance, run the following commands to clone the repo and activate the environment:

```bash
cd ~/ray-data-eval
git remote set-url origin https://github.com/ray-data-eval/ray-data-eval.git
git pull
conda activate raydata
python scripts/setup/warmup_models.py       # model cache; rerun after any reboot
```



### Setting up from scratch

If you launched instances using our provided AMI, you can skip this section.

We provide instructions for setting up each environment separately. Here is a checklist of all experiments and their corresponding environments:


| figures          | environment        | dependencies                                       |
| ---------------- | ------------------ | -------------------------------------------------- |
| 7a               | `raydata-rag`      | vLLM 0.7.3, torch 2.5.1+cu124, faiss 1.8.0         |
| 7b, 7c, 10a, 10b | `raydata`          | torch 2.8.0+cu128, transformers 5.5.4              |
| 8a               | `raydata-training` | torch 2.4.0+cu121, TensorFlow 2.16.1, numpy 1.26.4 |
| 9                | `raydata-fig9`     |                                                    |




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

For multi-node experiments that span multiple nodes, please first set up the same environment on every node, then run the following commands:

```bash
ray start --head --disable-usage-stats      # head node
ray start --address=<head-ip>:6379          # each worker
```

---



## Retrieval-augmented generation (Figure 7a)

This experiment uses **a single node with 8x H200 GPUs and 256 vCPUs**. To fully reproduce the experiment, you can use an AWS p5e.48xlarge instance (8x H200, 192 vCPUs). 

The experiment could also run on other instances, such as `g5.48xlarge` or `g5.24xlarge`. If you run on a node with fewer than 8 GPUs, you should set `GPU_COUNT` to the actual GPU count. For instance, this is the command we ran on a `g5.24xlarge`:

```bash
GPU_COUNT=4 VLLM_EXTRA_ARGS="--max-model-len 4096 --gpu-memory-utilization 0.95 --enforce-eager" \
    bash scripts/run_fig7a.sh
```

**Environment.** Check that you have the `raydata-rag` environment. If not, set up the environment by following the instructions in the "Setting up from scratch" section.

**Data and model.** With `raydata-rag` activated, run the following commands to download TriviaQA, build the knowledge base, and cache the model.

Note that to download the model, you need a Hugging Face account with access to [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). You can run `hf auth login` to log in to your Hugging Face account.

```bash
curl -LO http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz
tar xzf triviaqa-rc.tar.gz                       # gives qa/web-train.json

pip uninstall -y faiss-cpu                       # faiss-cpu and faiss-gpu cannot coexist
pip install faiss-gpu-cu12==1.8.0.2              # for the build only
python experiments/ray_data_eval/rag/build_kb_triviaqa_v2.py \
    --dataset qa/web-train.json --output-prefix kb --nlist 8192
pip uninstall -y faiss-gpu-cu12
pip install faiss-cpu==1.8.0                     # for the benchmark

huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --exclude "original/*"
# (newer huggingface_hub versions renamed the CLI: use `hf download` / `hf auth login`)
```

Then run the experiment:

```bash
bash scripts/run_fig7a.sh
```

This runs Ray Data-Dynamic at 1, 2, 4 and 8 GPUs and the staged baseline at 1.

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

The script runs on the GPU node and uses ssh to disconnect and reconnect the CPU node, so we need to set up passwordless ssh from the GPU node to the CPU node. To do this, you can copy a private key to the GPU node, add the corresponding public key to the CPU node's `~/.ssh/authorized_keys`, and check with `ssh ubuntu@<cpu-node-private-ip> hostname`.

For example, with the GPU node at `10.0.36.240` (private IP address) and the CPU node at `10.0.34.10` (private IP address):

### Executor failure experiment

```bash
# Executor failure: kills one worker process at t=15 min
bash scripts/run_fig7c.sh executor <cpu-node> <head-ip>

# for example:
bash scripts/run_fig7c.sh executor ubuntu@10.0.34.10 10.0.36.240
```



### Node failure experiment

```bash
# Node failure: disconnects the CPU-only node at t=15 min, rejoins it at t=30 min
bash scripts/run_fig7c.sh node <cpu-node> <head-ip>

# for example:
bash scripts/run_fig7c.sh node ubuntu@10.0.34.10 10.0.36.240
```

Note that the results in the AE paper copy were produced before we fixed two bugs in this benchmark, so you can expect the throughput to be higher when you reproduce the experiment. However, this does not affect the shape or conclusions of the figure.

Results are saved to `results/fault_tolerance/`.

## ResNet-50 training (Figure 8a)

This experiment uses **1 GPU node**.

**Environment.** Check that you have the `raydata-training` environment. If not, set up the environment by following the instructions in the "Setting up from scratch" section.

**Data.** Our public S3 bucket hosts a copy of ImageNet for artifact evaluation:

```bash
# download to local disk (~150 GB), for the _local series:
aws s3 sync --no-sign-request \
    s3://ray-data-eval-us-west-2/imagenet/ILSVRC/Data/CLS-LOC/train ~/imagenet/train

# or pass the S3 root directly to the script for the _s3 series:
bash scripts/run_fig8a.sh s3://ray-data-eval-us-west-2/imagenet/ILSVRC/Data/CLS-LOC
```

To run the experiment on the downloaded copy:

```bash
bash scripts/run_fig8a.sh <path-to-imagenet>
```

Results are saved to `results/resnet_training/`.

## Stable Diffusion pretraining (Figure 8b)

This experiment is built on a codebase for pretraining Stable Diffusion on a pool of a pool of 704 CPUs and 72 heterogeneous GPU. Please refer to the separate README in `experiments/ray-data-diffusion/`. Although we cannot share the internal training traces, we hope this codebase is helpful for demonstrating the usefulness of Ray Data in pretraining frontier models.

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

This experiment needs **1 node with 8 CPU cores and at least 15 GB free memory space in** `/dev/shm`, e.g. an m7i.2xlarge (8 vCPU, 32 GB).

**Environment.** Check that you have the `raydata` environment. If not, set up the environment by following the instructions in the "Setting up from scratch" section.

To run the experiment:

```bash
bash scripts/run_fig10a.sh
```

Results are saved to `results/partitioning/`.

## Scalability (Figure 10b)

This experiment needs **a CPU cluster**: 1 m8i.4xlarge (16 vCPU, 64 GiB) head node and up to 32 m8i.2xlarge (8 vCPU, 32 GiB) worker nodes.

**Environment.** Check that you have the `raydata` environment on every node. If not, set up the environment by following the instructions in the "Setting up from scratch" section. Then start a Ray cluster (see "Starting a Ray cluster") with the m8i.4xlarge as the head node.

To run the experiment:

```bash
bash scripts/run_fig10b.sh <N> # N should match the current cluster size
```

Results are saved to `results/scalability/`.

## Common failures

We will keep this section updated with common failures and their solutions.