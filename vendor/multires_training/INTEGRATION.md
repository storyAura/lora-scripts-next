# 接入新训练器

本包覆盖「多分辨率同时训练」闭环：档位数学 → 预处理 staging → latent 命名 → sample 扩展 → 同形分桶 → compile budget。下面按接线顺序说明。

## 1. 安装

```bash
pip install -e path/to/vendor/multires_training
# 或复制 multires_training/ 目录进新仓库并加入 PYTHONPATH
```

> 本仓库不用 pip 安装：见 [VENDOR.md](VENDOR.md)（惰性 `sys.path` 注入），且**未采用**下文第 3 节的 staging 目录方案，原因见
> [`docs/proposal/multires_per_image_migration.md`](../../docs/proposal/multires_per_image_migration.md) § 1。

依赖：`numpy`、`Pillow`（仅 staging resize）。训练器本体（torch / VAE / TE）由你方提供。

## 2. 配置 knobs

| Key | 类型 | 含义 |
|---|---|---|
| `target_res` | `list[int]` | 选中档位，允许值见 `ALLOWED_TARGET_RES` |
| `multires_per_image` | `bool` | `true` 时每图每档各训一次；要求 ≥2 档 |

校验：

```python
from multires_training import validate_multires_target_res, normalize_target_res

tiers = validate_multires_target_res(normalize_target_res(cfg.target_res))
```

## 3. 预处理接线

```python
from multires_training import make_staging_plan, stage_multires_images

plan = make_staging_plan(
    source_dir=cfg.source_dir,
    resized_dir=cfg.resized_dir,
    cache_dir=cfg.lora_cache_dir,
    target_res=cfg.target_res,
    multires_per_image=cfg.multires_per_image,
)
stage_multires_images(plan)          # 写 PNG
for d in plan.vae_input_dirs:        # 你的 VAE：扫这些目录
    your_vae_cache(d, out=plan.cache_dir)
# TE / PE / mask 只扫 plan.sidecar_image_dir (= resized/)
```

磁盘约定（可改 `LatentCacheConvention.suffix`）：

```
resized/                      # nearest-tier；sidecar 路径
multires/<edge>/...           # 每档 staging（仅 multires 模式）
lora/{stem}_{WxH}_anima.npz   # 每档一份，同 stem
```

NPZ 内必须含（`W,H` 为像素）：

- `latents_{H/8}x{W/8}`
- `original_size_{H/8}x{W/8}`
- `crop_ltrb_{H/8}x{W/8}`

## 4. Dataset 接线（核心）

**顺序必须是：先按源图做 train/val split，再 expand。**

```python
from multires_training import expand_dataset, build_shape_buckets, samples_to_bucket_items

train_paths, val_paths = your_split(source_image_paths)  # 按源图

train_samples = expand_dataset(
    train_paths,
    target_res=cfg.target_res,
    cache_dir=cfg.lora_cache_dir,
    image_dir=cfg.resized_dir,
    num_repeats=cfg.num_repeats,
)
# MultiresSample: source_path, image_key, width, height, latents_npz, edge, stem

epoch = build_shape_buckets(
    samples_to_bucket_items(train_samples),
    batch_size=cfg.batch_size,
    keep_incomplete_batches=True,  # multires 必开
)
```

训练 step：

1. 取 `epoch.indices[i]` → `epoch.batch_keys(...)`
2. 用 `image_key` 找回 `MultiresSample`
3. 从 `latents_npz` 加载 latent；TE/PE/mask 用 `stem` / `source_path`

硬失败语义（应在启动期暴露）：

| 异常 | 原因 |
|---|---|
| `FileNotFoundError` | 无 cache / 缺某一档 |
| `ValueError` | NPZ 缺 key / 同档多份 usable |

## 5. Compile / dynamic seq

```python
from multires_training import derive_token_budget

lo, hi, counts = derive_token_budget(epoch.resos, sample_prompt_sizes=prompt_whs)
# 把 [lo, hi] 交给 mark_dynamic / compile_dynamic_seq
# 不要只用 target_res 的理论 band 当唯一真相
```

## 6. 接入 checklist

- [ ] 配置暴露 `target_res` + `multires_per_image`
- [ ] preprocess 写 `multires/<edge>/` 且 VAE 扫 `plan.vae_input_dirs`
- [ ] TE/PE/mask 仍只基于 nearest-tier / stem
- [ ] train/val **先 split 再 expand**
- [ ] `expand_dataset` 接到 dataloader 构造
- [ ] `keep_incomplete_batches=True`
- [ ] compile budget 来自 `derive_token_budget(实际 shapes)`
- [ ] 跑通包内 `tests/test_integration.py`

## 7. 最小集成测试（新训练器侧建议）

在新仓库加一条 smoke：

1. 合成 1 张图 → `stage_multires_images`
2. `write_stub_latent_npz`（或真实 VAE）写两档 cache
3. `expand_dataset` 得到 2 条 sample、同一 `source_path`
4. `build_shape_buckets(batch_size=2)` 后 `all_keys_in_epoch()` 覆盖全部 sample
5. 删掉一档 cache → `expand_*` 抛 `FileNotFoundError`

包内已有完整版：`tests/test_integration.py`。

## 8. 与 staged resolution 的边界

| | 本包 | staged |
|---|---|---|
| 多档出现时机 | **每个 epoch 同时** | 按进度切换活跃子集 |
| 是否在本包 | 是 | 否（勿混入） |

## 9. 从本仓库原实现对照

| 本包 | 原路径 |
|---|---|
| `tiers` | `library/datasets/buckets.py` |
| `cache` | `library/io/cache.py` |
| `expand` | `library/datasets/dreambooth.py` (`multires_cache_variants`) |
| `staging` | `library/preprocess/images.py` (multires 分支) |
| `batching` | `library/datasets/base.py` (`keep_incomplete_batches`) |
| `budget` | `train.py` `_derive_token_budget` |
