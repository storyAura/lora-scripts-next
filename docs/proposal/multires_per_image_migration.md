# `multires_per_image`（多分辨率同时训练）迁移与实现记录

> 来源包：`vendor/multires_training/`（从 MonadForge / Anima 抽出的可移植逻辑）
> 落地范围：Anima LoRA(`sd3-lora` / `anima-lora`) 与 Anima 全量微调(`anima-finetune`) 标准路径
> 状态：**已实现**（2026-08-06），Anima Fast 未接入
> 包内通用接入说明：[`vendor/multires_training/INTEGRATION.md`](../../vendor/multires_training/INTEGRATION.md)

## 0. 功能定义

开启 `multires_per_image` 后，**同一 epoch 内**每张源图在所有选中 free-fit 档位各出现一次。

| | 关闭（默认） | 开启 |
|---|---|---|
| 分辨率语义 | 单目标 `resolution` + ARB 分桶 | 多档 `target_res` free-fit |
| 每图样本数 | 1 | = 档位数（≥2） |
| 分桶来源 | `BucketManager.select_bucket`（ARB） | 每档按原图宽高比算 free-fit 桶 |
| `enable_bucket` / min / max / steps | 生效 | **被忽略**（adapter 会给出 warning） |

**不是**：staged resolution（按训练进度切换活跃档位）；与 UI 里的 `multires_noise_*`（金字塔噪声）无关。

允许档位（包契约，勿擅自扩）：`512 768 896 1024 1280 1536`。

## 1. 关键设计决定：不做 staging 目录

包内 `INTEGRATION.md` 给的通用方案是把图片按档位物理 resize 到 `multires/<edge>/`，再让 VAE 逐目录编码，产出 `{stem}_{桶宽x桶高}_anima.npz`。**本训练器没有采用**，因为 sd-scripts 已经具备等价能力：

| 需求 | sd-scripts 现成能力 |
|---|---|
| 按桶尺寸缩放+裁剪 | `trim_and_resize_if_required(resized_size → bucket_reso)`，缓存与实时读图都走它 |
| 一图多分辨率 latent | `AnimaLatentsCachingStrategy` 用 `multi_resolution=True`，NPZ key 带 `_{H/8}x{W/8}` 后缀；`save_latents_to_disk` **合并**已有 key |
| 不完整批次不丢档 | `make_buckets` 的 `math.ceil` 已保留尾批 |

因此实现取更薄的一条路：

- **不复制图片**，不产生 `resized/` + `multires/<edge>/` 目录树（少一次重采样，省磁盘）
- **一张源图一个 npz**，文件名仍按原图像素（`{stem}_{原宽x原高}_anima.npz`），各档 latent 以分辨率后缀共存于同一文件
- 复用包的**档位数学**（`tiers.py`：`freefit_bucket` / `freefit_band_for_edge` / `validate_multires_target_res`）与 **budget**（`budget.py`），不使用包的 `staging.py` / `cache.py` / `expand.py` 的磁盘约定

代价：与包的磁盘命名约定不同，故包内 `expand_dataset()` 不能直接读本训练器的缓存目录（若日后需要跨训练器共享缓存，需要写一层命名适配）。

## 2. 实现清单（代码位置）

### 2.1 包 vendor 化

| 项 | 内容 |
|---|---|
| 位置 | `vendor/multires_training/`（含 `pyproject.toml`、`README.md`、`INTEGRATION.md`、`tests/`） |
| 导入方式 | 两个桥接模块各自把该目录插入 `sys.path`（惰性），无需 `pip install -e`、无需改 PYTHONPATH |
| 依赖 | `numpy`、`Pillow`（均为本仓库既有依赖） |

### 2.2 GUI 侧

| 文件 | 改动 |
|---|---|
| `mikazuki/multires.py`（新增） | 桥接：`normalize_target_res` / `validate_target_res` / `format_target_res` / `allowed_target_res`；重量级 import 惰性化 |
| `mikazuki/schema/sd3-lora.ts`、`anima-finetune.ts` | 数据集设置新增 `multires_per_image`(bool) 与 `target_res`(逗号分隔字符串) |
| `mikazuki/training_validation.py` | 启动前硬失败：非 Anima 训练类型、档位 <2、未知档位、`random_crop` 冲突 |
| `mikazuki/anima_backend/adapter.py` | `SUPPORTED_FIELDS` 收录两键；`_normalize_multires_fields()` 归一为 `target_res="512,1024"`；关闭时**丢弃**两键；开启且 `enable_bucket` 时追加 ARB 被忽略的 warning |
| `frontend/dist/assets/sd-schema-i18n-en.json` | 两条新文案的英文；已 bump SPA cache key |

`target_res` 走**字符串**而非数组：TOML 最终喂给 argparse，`--target_res` 是 `str` 选项，数组无法安全通过。

### 2.3 训练侧（`vendor/sd-scripts`）

| 文件 | 改动 |
|---|---|
| `library/anima_multires.py`（新增） | 纯 Python（不 import torch）：`validate_target_res`、`plan_image_tiers`（每档 → 桶尺寸 + `resized_size`）、`cover_resized_size`、`expand_image_data`（ImageInfo 逐档克隆）、`bucket_resolutions`、`derive_token_budget`、`shard_index` |
| `library/train_util.py` | `BaseDataset.__init__` 接收并校验两键；新增 `_prepare_multires_buckets()`；`make_buckets()` 增多分辨率分支（旁路 ARB）；`__getitem__` 实时读图分支兼容 `multires_per_image`；`new_cache_latents` 多进程分片改为**按源图**；`add_dataset_arguments` 增 `--multires_per_image` / `--target_res` |
| `library/config_util.py` | `BaseDatasetParams` 新字段；`DATASET_ASCENDABLE_SCHEMA` 接受两键；`ARGPARSE_NULLABLE_OPTNAMES` 加 `target_res`；`print_info` 打印档位并提示 ARB 被忽略 |
| 三个 dataset 子类 | `DreamBoothDataset` / `FineTuningDataset` / `ControlNetDataset` 透传两键（ControlNet 转发给 delegate） |

流程：

```
make_buckets()
  载入 image_size
  → multires 分支：expand_image_data()  # 一图 → N 档 ImageInfo（image_key 带 ::anima-multires=WxH）
      每档 bucket_reso = freefit_bucket(原宽,原高, band(档位))
      每档 resized_size = cover 缩放（保证中心裁剪有像素）
      image_size 保持原图 → 各档共用同一 npz 路径
  → BucketManager(predefined = 各档桶集合)，跳过 select_bucket
  → add_image / buckets_indices（ceil 保尾批）
  → 日志：样本数、来自真实桶形状的 token budget [lo, hi]
```

### 2.4 多进程缓存安全

同一图各档写**同一个 npz**（read-modify-write）。原逻辑 `i % num_processes` 会把同图不同档分给不同进程，多卡下会并发改写同一文件。改为 `anima_multires.shard_index(absolute_path, num_processes)`（crc32），保证同图恒由同一进程缓存。

## 3. 测试

| 文件 | 覆盖 |
|---|---|
| `tests/test_multires_per_image.py` | `target_res` 解析（含全角逗号）/ 校验拒绝单档与未知档；validation 四类硬失败；adapter 归一与关闭时丢弃；档位规划（16 对齐、落在 token band、`resized_size` 覆盖桶）；ImageInfo 扩展；预烘焙 latent 拒绝；`shard_index` 稳定；`build_shape_buckets` 全覆盖 |
| `tests/test_multires_dataset_integration.py` | 真实 `DreamBoothDataset.make_buckets()`：2 图 × 2 档 = 4 样本、4 个桶、同图 npz 路径唯一、大 batch 不丢样本、关闭时回归 ARB、单档构造即报错；blueprint 参数 / user-config schema / CLI flag 三处 plumbing |
| `vendor/multires_training/tests/` | 包自身逻辑（档位数学、cache 约定、expand、batching、端到端） |

## 4. 已知边界与后续

- **会放大小图**：free-fit 是 cover 适配，档位高于原图时会上采样（例：768×1024 在 1024 档得到 896×1200）。这是档位语义本身的结果，`bucket_no_upscale` 不再参与。
- **TE / caption 缓存重复计算**：TE 输出按源图路径命名，磁盘缓存下第二档起命中跳过；纯内存缓存（`cache_text_encoder_outputs` 不落盘）会按档各算一次，浪费但结果正确。
- **metadata（fine tuning）数据集**：若 `latents_npz` 已预烘焙则直接报错，多分辨率要求 latent 由训练器缓存。
- **Anima Fast** (`anima-lora-fast`)：未接入，Fast 有自己的 preprocess/preflight。
- **步数换算**：样本数按档位数成倍增加，epoch 数需相应下调（schema 文案已提示）。
- 若日后要与包的磁盘约定互通（跨训练器共享缓存），需要在 `AnimaLatentsCachingStrategy.get_latents_npz_path` 上加 multires 命名分支，并让 `expand_dataset()` 成为缓存发现入口。

## 5. 路径速查

| 资源 | 路径 |
|---|---|
| 包 README / 通用接入 | `vendor/multires_training/README.md`、`INTEGRATION.md` |
| 档位数学 / budget | `vendor/multires_training/multires_training/tiers.py`、`budget.py` |
| GUI 桥接 | `mikazuki/multires.py` |
| 校验 | `mikazuki/training_validation.py` |
| Adapter | `mikazuki/anima_backend/adapter.py` |
| 训练侧桥接 | `vendor/sd-scripts/library/anima_multires.py` |
| 数据集 / 分桶 | `vendor/sd-scripts/library/train_util.py` |
| Latent 策略 | `vendor/sd-scripts/library/strategy_anima.py` |
