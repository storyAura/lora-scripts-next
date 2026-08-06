# multires-training

可移植的 **多分辨率同时训练**（`multires_per_image`）工具包：同一 epoch 内，每张源图在所有选中 free-fit 档位各出现一次。

从 MonadForge / Anima 抽出的纯逻辑，**不依赖** `train.py` / Accelerate / DiT。可直接 `pip install -e` 接到新训练器。

> 不是分阶段分辨率（staged resolution）。两者对比见 [INTEGRATION.md](INTEGRATION.md)。

## 安装

```bash
cd vendor/multires_training
pip install -e ".[dev]"
# 或在新训练器仓库里把本目录 vendoring 进去（本仓库即如此，见 VENDOR.md）
```

## 模块地图

| 模块 | 作用 |
|---|---|
| `tiers` | `EDGE_TOKEN_BANDS` / `choose_edge` / `freefit_bucket` / band 匹配 |
| `cache` | `{stem}_{WxH}_anima.npz` 解析、NPZ 校验、stub 写入 |
| `expand` | 源图 → 每档一个 `MultiresSample`（**先 split 再 expand**） |
| `staging` | `resized/` + `multires/<edge>/` 布局与最小 resize |
| `batching` | 按 `(W,H)` 分桶；`keep_incomplete_batches` 防丢档 |
| `budget` | 从真实 shapes 推导 compile token range |

## 30 秒用法

```python
from pathlib import Path
from multires_training import (
    make_staging_plan,
    stage_multires_images,
    write_stub_latent_npz,
    expand_dataset,
    build_shape_buckets,
    samples_to_bucket_items,
    derive_token_budget,
)

root = Path("data")
plan = make_staging_plan(
    source_dir=root / "src",
    resized_dir=root / "resized",
    cache_dir=root / "lora",
    target_res=[512, 768],
    multires_per_image=True,
)
report = stage_multires_images(plan)

# 新训练器用自己的 VAE 扫 plan.vae_input_dirs；测试可用 stub：
for stem, entries in report.items():
    for edge, (w, h) in entries:
        write_stub_latent_npz(plan.cache_dir / f"{stem}_{w:04d}x{h:04d}_anima.npz", w, h)

samples = expand_dataset(
    [str(p) for p in (root / "resized").glob("*.png")],
    target_res=[512, 768],
    cache_dir=str(plan.cache_dir),
    image_dir=str(plan.resized_dir),
)
epoch = build_shape_buckets(
    samples_to_bucket_items(samples),
    batch_size=2,
    keep_incomplete_batches=True,
)
lo, hi, counts = derive_token_budget(epoch.resos)
```

## 行为契约

- `multires_per_image=true` 且 `target_res` ≥ 2
- 允许档位：`512 768 896 1024 1280 1536`
- TE / PE / mask / caption 按 **stem 共享**；VAE latent **每档一份**
- 缺档 / 坏 NPZ / 同档多份 usable cache → **硬失败**
- 分桶保留 incomplete tail，避免 `batch_size>1` 静默丢档

## 测试

```bash
cd vendor/multires_training
pytest -q
```

## 文档

- [INTEGRATION.md](INTEGRATION.md) — 接入新训练器的步骤与 checklist
- 仓库内源码对照：[`docs/proposal/multires_per_image_migration.md`](../../docs/proposal/multires_per_image_migration.md)
