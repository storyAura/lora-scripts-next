# vendor/multires_training

多分辨率同时训练（`multires_per_image`）的**可移植逻辑包**，来自 MonadForge / Anima，独立于本训练器。

## 如何被导入

不通过 `pip install`。两个桥接模块在需要时把本目录插入 `sys.path`（惰性 import，未开启该功能时不付出 numpy / Pillow 的导入代价）：

| 侧 | 桥接模块 |
|---|---|
| GUI / 后端 | `mikazuki/multires.py` |
| 训练子进程 | `vendor/sd-scripts/library/anima_multires.py` |

## 本仓库实际使用的部分

| 模块 | 是否使用 | 说明 |
|---|---|---|
| `tiers.py` | ✅ | 档位数学：`freefit_bucket` / `freefit_band_for_edge` / `validate_multires_target_res`，**唯一档位真相** |
| `budget.py` | ✅ | 由真实桶形状推导 token 区间，用于日志与 compile 预算 |
| `batching.py` | 测试中使用 | sd-scripts 的 `make_buckets` 已用 `ceil` 保留尾批，等价能力 |
| `staging.py` / `cache.py` / `expand.py` | ❌ | 本仓库不复制图片、不按桶尺寸命名 npz；理由见 `docs/proposal/multires_per_image_migration.md` § 1 |

## 修改约定

- 允许：数值/逻辑修复，且必须同步跑 `cd vendor/multires_training && pytest -q`
- 允许：新增档位或改动 `EDGE_TOKEN_BANDS` 前，先确认与 Anima RoPE 上限、`patch=16` 假设一致；改了要同步
  `mikazuki/multires.py` 的 `DOCUMENTED_ALLOWED_TARGET_RES`（有测试断言两者一致）与两个 schema 的文案
- 禁止：把本训练器专属的路径/配置知识写进本包（它要保持可移植）；那类代码放两个桥接模块里
