# `vendor/lycoris/` 来源说明

本目录是**本地魔改版 LyCORIS**，随仓库一起分发，用于覆盖 pip 安装的上游包。

## 为什么要 vendor

`requirements.txt` 里的 `lycoris-lora==3.3.0` 是上游官方包，**不包含**本项目的本地扩展算法与 Anima 适配修复。此前这些改动只存在于 `venv/Lib/site-packages/lycoris/`（被 `.gitignore` 排除），导致：

- 换机器 / 重建 venv 就全部丢失
- 对 `lycoris-lora` 执行 `--force-reinstall` 会静默退回官方版
- 需要人工在两个副本之间同步，漏同步不会有任何报错

vendor 进仓库后，改动跟着 git 走，并由脚本 + 测试保证一致。

## 安装 / 同步

创建或重装 venv 之后执行：

```bash
python scripts/sync_vendored_lycoris.py
```

该脚本把本目录覆盖到当前解释器的 `site-packages/lycoris/`。
`--check` 只报告差异不复制（退出码 1 表示有漂移）。

`tests/test_vendored_lycoris.py` 会校验已安装副本与本目录一致 —— **如果你直接改了
venv 里的 lycoris，这个测试会失败**，提醒你把改动同步回 `vendor/lycoris/` 并提交。

## 相对上游 3.3.0 的改动

### 新增本地扩展算法（仅作用于 Linear 层，Conv/Norm 层自动跳过）

| 算法 | 文件 | 说明 |
|---|---|---|
| GLoKR | `modules/glokr.py` | LoKr + GLoRA 三路径融合，详见同目录 `GLOKR.md` |
| T-GLoKR | `modules/glokr.py` | GLoKR + 时间步门控（`train_time_gates`），本仓库原创 |
| BoKR | `modules/bokr.py` | LoKr + BoRA 双向权重解耦 |
| BoRA | `modules/bora.py` | 双向范数解耦的 LoRA 变体 |
| GloKrSora | `modules/gsokr.py` | GLoKR 的 SoRA 稀疏化变体（实验性） |
| GLoRA-BOFT | `modules/glora_boft.py` | GLoRA + 蝶形正交变换 |

### 修复（2026-07-25）

- `kohya.py`：`create_network` 新增 `extra_algo_kwargs` 转发（`train_gates` /
  `init_mode` / `g_norm_mode` / `bora_iters` / `use_g_out` / `use_sora` /
  `sora_r` / `sora_epsilon` / `train_time_gates` / `time_gate_dim`）。此前这些
  参数经 `network_args` 传入会被**静默丢弃**
- `kohya.py`：`LycorisNetworkKohya` 新增 `set_current_timestep` /
  `clear_current_timestep`，供时间步感知算法使用
- `modules/bokr.py`：`__init__` 补回第 9 位 `use_tucker` 占位参数。kohya 按位置
  传参，缺失会与 `use_scalar` 冲突，选 bokr 即 `TypeError` 无法建网
- `modules/gsokr.py`：`factor` / `sora_r` / `sora_epsilon` 显式转数值。
  `network_args` 传入的都是字符串，直接参与比较会 `TypeError`
- `modules/glokr.py`：时间门控全程 fp32 计算（模块 `.to(bf16)` 会把频率 buffer
  一并降精度，与 `.float()` 的门控权重 `torch.dot` 时 dtype 不符）；
  `time_gate_freqs` 改 `persistent=False`
- `modules/norms.py`：`train_norm` 跳过无仿射权重的归一化层。Anima DiT 的
  LayerNorm 全是 `elementwise_affine=False`（`weight` 为 `None`），此前会照常
  包装并在首次 forward 崩溃（`'NoneType' object has no attribute 'to'`）；
  `bias` 判断由 `hasattr` 改为 `is not None`（`bias=False` 时同样是陷阱）

## 注意

- 请勿对 `lycoris-lora` 执行 `pip install --force-reinstall` / `-I`，否则会覆盖
  本目录的改动；若已执行，重新跑一次同步脚本即可
- 上游许可证随包保留，本目录未改动许可条款
