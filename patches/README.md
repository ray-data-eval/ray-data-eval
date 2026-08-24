# Patches to the vendored originals

Everything under `experiments/` is upstream code copied unmodified, except for
the patches listed here. Each is applied in place and the diff is kept, so
the change is auditable.

| Patch | File | Why |
|---|---|---|
| `setting-env.patch` | `microbenchmarks/memory_pipelining/setting.py` | Read benchmark parameters from the environment instead of hardcoding them. The committed defaults are the authors' development scale (20 tasks); §5.3.1's configuration is 160. **Defaults are unchanged**, so running the original directly behaves exactly as before. |

## Provenance

| Directory | Source | Commit |
|---|---|---|
| `experiments/ray_data_eval/` (except `rag/`, `scalability/`) | github.com/frank-lsf/ray-data-eval | `c855896` |
| `experiments/ray_data_eval/{rag,scalability}/` | github.com/efeslab/ray, branch `yilegu` | `cd188c8` |

The layout mirrors upstream `ray_data_eval/`, one directory per experiment.
`rag/` and `scalability/` come from the other repo and are folded in as
sibling experiment directories rather than kept in a second tree.

Each patch is a unified diff against the upstream file and applies with
`patch -p1` from `experiments/`. All four were checked against upstream
when generated.

## Verifying a patch

```bash
diff -u <(git -C ../ray-data-eval show c855896:ray_data_eval/microbenchmarks/setting.py) \
        experiments/ray_data_eval/microbenchmarks/memory_pipelining/setting.py
```

---

## `microbenchmarks-imports.patch`

`microbenchmarks/memory_pipelining/raydata/producer_consumer_gpu.py` imports

```python
from ray_data_eval.microbenchmarks import timeline_utils
```

but `timeline_utils.py` lives in `memory_pipelining/raydata/`, and
`microbenchmarks/` has no `__init__.py`. **The benchmark does not run as
committed** — it fails at import. The patch corrects the module path, and
empty `__init__.py` files were added to `microbenchmarks/` and
`memory_pipelining/raydata/` so the package is importable.

---

## `video-inference-env.patch`

`video_inference/ray_data_pipeline_map.py` hardcodes its `DataContext`
settings. The patch adds one thing, a no-op when the environment is unset:

- `RAY_DATA_CTX_<setting>` environment variables are applied to the `DataContext`,
  so the runner can select a baseline emulation (microbatch, Drizzle, Cameo
  LLF) without editing the benchmark. Unknown settings raise, rather than
  silently doing nothing — `DataContext` is a plain dataclass, so a typo or a
  missing Ray Data installation would otherwise produce a mislabelled result.
One behavioural change. The file as committed hardcoded
`scheduling_policy = "microbatch"` and the BSP settings, so running it
unmodified gave the Spark Streaming baseline rather than Ray Data. With no
`RAY_DATA_CTX_*` set it now leaves the `DataContext` at its own defaults, which
is Ray Data. The effective policy is printed at startup either way, so which
system ran is never in doubt.

---

## `partitioning-env.patch`

`microbenchmarks/partitioning/raydata.py` fixes `NUM_ROWS = 8192` and hardcodes its
eleven partition sizes inside `main()`, so a full sweep is the only thing it
can do -- about twenty minutes. `PARTITION_NUM_ROWS` and `PARTITION_SIZES`
now override both. **Defaults are unchanged**, so running it directly behaves
exactly as before.

---

## `ray-image-transform.patch`

Applies to Ray itself, not to `experiments/`. `scripts/setup/install_ray_data.sh`
applies it when it installs Ray Data.

`image_training/e2e_training/ray_data_e2e_training.py` calls

```python
ray.data.read_images(traindir, partitioning=..., mode="RGB", transform=train_transform)
```

`transform=` is not part of upstream Ray. It was added in
[ronyw7/ray `feat/im-transform`](https://github.com/ronyw7/ray/tree/feat/im-transform)
to apply the torchvision transform during decode rather than as a separate
`map`, which is the point of Figure 8a: where the transform executes.

That branch is based on Ray master from March 2024, roughly 2.10. This patch
is the same change ported to 2.40.0, where `ImageDatasource` has moved to
`ray/data/_internal/datasource/`. Four lines add the parameter, store it, and
apply it after resize and convert; two more thread it through `read_images`.

Without it the run dies immediately with
`TypeError: read_images() got an unexpected keyword argument 'transform'`.
