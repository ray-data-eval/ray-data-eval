# Plotting

One script per experiment. Each reads `results-archive/` and writes a PDF and
PNG into `figures/`.

```bash
python plots/rag.py                      # uses results-archive/
python plots/rag.py --results results    # uses your own run
```

Every script came from the code that produced the published figure.

\* Four of those originals carry their measurements as literal arrays in the
source rather than reading a file -- that is still the data, so it has been
written out to `results-archive/` and the scripts read it like the rest.

| Script | Figure | From | Original read data? |
|---|---|---|---|
| `rag.py` | 7a | `rag-ray/plot/RAG/jct.py` | in the script* |
| `video_classification.py` | 7b | `~/Dev/video-inference/video_inference.ipynb` | yes |
| `fault_tolerance.py` | 7c | `ronw-vm:ray_data_eval/video_inference/video_inference.ipynb` | yes |
| `resnet_training.py` | ResNet-50 | `ray-data-eval/.../plotting/image_training.ipynb` | yes |
| `memory_pipelining.py` | 9 | *not found* — values read from the paper | — |
| `partitioning.py` | 10a | `ray-data-eval/.../partitioning_benchmark/plot_throughput.py` | in the script* |
| `scalability.py` | 10b | `rag-ray/plot/scalability/plot_scalability.py` | in the script* |

`_style.py` holds the rcParams the originals each repeated. One behaviour
change: they set `text.usetex: True`, which crashes without LaTeX, so LaTeX is
now used only when available.

## What plots today

| | |
|---|---|
| `rag` | 159.1 / 120.4 / 63.9 / 33.6 / 18.7 min — matches the paper exactly |
| `video_classification` | Ray Data-dynamic 9.0 min through Spark 115.7 min |
| `fault_tolerance` | all four curves at ~95% of GPU saturation; failure at t=15 |
| `scalability` | Ray Data 1.81x the best baseline at 32 nodes |
| `memory_pipelining` | Ray Data 1.33x optimal, survives to 6 GB |
| `partitioning` | peak 280.7 rows/s at 64 MB |
| `resnet_training` | Ray Data 90-93% of max GPU, tf.data (S3) 12% |

All seven plot from `results-archive/`.

## Figure 10b's data

*`plot_scalability.py` holds its three series as literal arrays rather than
reading a file. Those arrays are the data, so they are stored as
`results-archive/scalability/{ray_data,ray,ray_streaming}.csv` and the script
reads them like every other figure.

`rag-ray/logs/ray_data_vs_ray_original_32nodes_*/` holds a separate sweep that
grows the dataset with the cluster -- 10 GB at 2 nodes up to 320 GB at 33 --
so it measures something else and reaches much larger absolute throughputs.
It is kept out of the archive, since it measures something else.

## Figure 7b has three traps

1. **Six series, not three.** Cameo (LLF) is *not* one of them — the original
   notebook has seven Cameo lines commented out. `cameo_llf.csv` is archived
   for a separate claim in section 2.2.
2. **Each series is resampled at its own interval**, from 2.5 s for
   Ray Data-microbatch to 50 s for Spark. Not cosmetic: at a uniform interval the
   microbatch sawtooth flattens, and that sawtooth is the figure's argument.
3. **`Ray Data-static` loads from `radar-round-robin.csv`.** Label and filename
   disagree in the original; the label is what the paper prints.

The broken x-axis exists because Spark runs to 117 min while everything else
finishes inside 70.

## Figure 9 is a heatmap, not a bar chart

`microbenchmarks/plot.py` writes `synthetic.pdf`, a grouped bar chart at two
memory settings against a dashed minimum of 153 s. That figure is not in the
paper. Figure 9 is `synthetic_heatmap.pdf`: six systems by six memory limits,
grey where the system OOMed, and the optimum is 150 s from section 5.3.1's
formula `(160 x 5s + 800 x 0.5s) / 8`.

`plots/memory_pipelining.py` draws the heatmap. Its values were read from the
submitted PDF, since no plotting script for it exists in any repo we have.

## Smaller gaps

- `fault_tolerance.py` takes its max-GPU line from `MAX_GPU_TPUT`, not from
  `inference_tput.csv`, which nothing reads.
- The two checkpoint curves are stitched from segments, deducting the work
  each restart rolled back. Where a restart resumed from is mostly computed:
  the final segment ran to the end, so it resumed at total batches minus the
  batches it ran, and the total comes from the uninterrupted runs in the same
  directory. Earlier restarts cannot be recovered that way -- they cancel out
  of the total, since redone work does not move the finish line -- so those
  sit in `<name>_segments.csv`. `--check-segments` shows where each number
  came from and that they account for a complete run.
- `partitioning.py`'s x-axis is labelled "Partition size (MB)" but the
  variable is `num_rows_in_block`. One row is 1 MB here, so they coincide.
- `results-archive/video_classification/cameo_llf.csv` is the one archived
  file no figure reads. It supports the Cameo comparison in section 2.2,
  which the paper states without plotting: 558.7 s against Ray Data-dynamic's
  561.6 s, i.e. within 1%.
