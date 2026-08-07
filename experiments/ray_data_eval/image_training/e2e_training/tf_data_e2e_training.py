import tensorflow as tf
import tensorflow_io as tfio  # noqa: F401
import torch
import torch.nn as nn
import torchvision.models as models

from util import IMAGENET_WNID_TO_ID
import os
import time
import argparse
import functools
from ray_data_eval.video_inference.ray_data_pipeline_helpers import postprocess

USE_LOCAL = False

traindir = os.path.join("/home/ubuntu/image-data/ILSVRC/Data/CLS-LOC", "train")
vardir = os.path.join("/home/ubuntu/image-data/ILSVRC/Data/CLS-LOC", "var")

parser = argparse.ArgumentParser(description="tf.data ImageNet Training")
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
)


@tf.function
def load_and_decode_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    return image


@tf.function
def train_transform(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, [256, 256])
    image = tf.image.random_crop(image, [224, 224, 3])
    image = (image - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225])
    return image


@tf.function
def val_transform(image):
    image = tf.image.resize(image, [256, 256])
    image = tf.image.central_crop(image, central_fraction=0.765625)  # Crop to (224, 224)
    return image


def list_filenames_labels(dir_path, category=None):
    filenames = []
    labels = []
    for path in os.listdir(dir_path):
        full_path = os.path.join(dir_path, path)
        if os.path.isdir(full_path):
            new_filenames, new_labels = list_filenames_labels(full_path, category=path)
            filenames.extend(new_filenames)
            labels.extend(new_labels)
        else:
            filenames.append(full_path)
            labels.append(IMAGENET_WNID_TO_ID[category])
    return filenames, labels


def timeit(name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print(f"{name or func.__name__}: {time.time() - start:.2f} seconds")
            return result

        return wrapper

    if callable(name):
        return decorator(name)
    else:
        return decorator


def train_one_epoch(dataset, model, criterion, optimizer, device, start_time):
    end = start_time
    running_loss = 0.0
    i = 0

    for images, labels in dataset:
        i += 1

        images = torch.tensor(images.numpy(), dtype=torch.float).permute(0, 3, 1, 2).to(device)
        labels = torch.tensor(labels.numpy(), dtype=torch.long).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            print(f"Batch {i}, loss: {loss.item():.4f}")

        batch_end_time = time.time()
        elapsed_time = batch_end_time - end
        print(
            "[Completed Batch]",
            batch_end_time,
            len(images),
            "[Training Tput]",
            len(images) / elapsed_time,
            flush=True,
        )
        end = time.time()


@timeit
def main():
    args = parser.parse_args()

    # Get the list of filenames and labels
    train_filenames, train_labels = list_filenames_labels(traindir, IMAGENET_WNID_TO_ID)
    if not USE_LOCAL:
        train_filenames = [
            path.replace("/home/ubuntu/image-data/", "s3://ray-data-eval-us-west-2/imagenet/")
            for path in train_filenames
        ]

    start_time = time.time()
    print("[Start Time]", start_time, flush=True)

    model = models.__dict__["resnet50"]()
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Pipeline
    # - interleave (load_and_decode_image)
    # - map (train_transform)
    # - batch + prefetch
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))

    # Set number of threads to use
    # options = tf.data.Options()
    # options.threading.private_threadpool_size = 4096 * 4
    # train_dataset = train_dataset.with_options(options)

    train_dataset = train_dataset.interleave(
        lambda filename, label: tf.data.Dataset.from_tensors((filename, label))
        # .with_options(options)
        .map(
            lambda x, y: (load_and_decode_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_dataset = train_dataset.map(
        lambda x, y: (train_transform(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_dataset = train_dataset.batch(batch_size=args.batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    train_one_epoch(train_dataset, model, criterion, optimizer, device, start_time)
    postprocess(f"tf_data_e2e_training_g5_xlarge_batch_{args.batch_size}.out")


if __name__ == "__main__":
    # https://github.com/tensorflow/tensorflow/issues/42738
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    main()
