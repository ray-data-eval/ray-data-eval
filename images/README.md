# Prebuilt images

Reviewers should use the AMI (below) once it is released; until then, follow
the manual setup in the main README. The Dockerfiles here are the recipe it
was built from: each runs the same commands the main README documents, so the
image and the from-scratch instructions cannot drift.

## The AMI

`ami-0ba8b0aff59c56f24` in `us-west-2` -- built and verified, and will be made
public once release permissions are in place. Built from artifact commit `aaf422c` on a
`g5.xlarge` with the four environments below installed exactly as the main
README says, plus `scripts/setup/warmup_models.py`. Its login banner names the
commit; `git pull` in `~/ray-data-eval` brings the checkout up to date. A new
AMI is only needed when what is installed changes: `env/requirements-*.txt`,
`scripts/setup/install_ray_data*.sh`, or the Ray fork branches.

## The Dockerfiles

Four, one per environment.

| image | figures | node |
|---|---|---|
| `Dockerfile.raydata` | 7b, 7c, 10a, 10b | GPU |
| `Dockerfile.training` | 8a | GPU |
| `Dockerfile.rag` | 7a | 8x H200, 256 vCPU |
| `Dockerfile.fig9` | 9 | 1 m6i.2xlarge, no GPU |

Figure 8b is not imaged: it is vendored with its own environment (torch 2.1.0)
and instructions under `experiments/ray-data-diffusion/`.

Build from the repository root:

```bash
docker build -f images/Dockerfile.raydata -t raydata:nsdi27 .
```

Run with the GPU and AWS credentials passed through:

```bash
docker run --gpus all -it \
    -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION \
    raydata:nsdi27
```

**Every node of a multi-node experiment must run the same image.** Ray refuses
to join nodes whose Python or Ray versions differ, and this is the most common
way a hand-built cluster fails.

Two things stay outside the images: AWS credentials, because the video dataset
is read from S3, and the ImageNet and TriviaQA downloads, which cannot be
redistributed.

## Rebuilding the AMI

Launch a `g5.xlarge` from the Deep Learning OSS Nvidia Driver AMI (Ubuntu
24.04) with a public IP and a 200 GB root volume, install Miniconda, accept its
channel terms (`conda tos accept`), create the four environments with the main
README's commands, run `python scripts/setup/warmup_models.py`, write the
commit into `/etc/motd`, empty `~/.ssh/authorized_keys`, then
`aws ec2 create-image` and
`aws ec2 modify-image-attribute --launch-permission "Add=[{Group=all}]"`.
AMIs are region-scoped; the dataset bucket is in `us-west-2`.
