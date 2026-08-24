# Prebuilt images

Four images, one per environment. Each Dockerfile runs the same commands the
main README documents, so the image and the from-scratch instructions cannot
drift.

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

## Or as an AMI

To hand reviewers a machine image instead, launch the documented instance type,
follow the "Set up your node" section of the main README, run
`python scripts/setup/warmup_models.py`, and create an AMI from the result. Share it
with `aws ec2 modify-image-attribute --launch-permission "Add=[{Group=all}]"`.
AMIs are region-scoped, so copy it to the regions reviewers will use. The
dataset bucket is in `us-west-2`.
