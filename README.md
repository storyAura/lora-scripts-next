<p align="center">
  <img src="assets/readme/next-story-trainer-cover.png" alt="Next Story Trainer" width="880" />
</p>

<h1 align="center">Next Story Trainer</h1>

<p align="center">
  <b>Windows 一键 LoRA / 全量微调训练工具</b> — 支持 <b>Anima</b> / SD 1.5 / SDXL / Flux<br/>
  解压即用，无需配环境。Anima LoRA 约 12GB 显存即可起步；<b>Anima 全量微调建议 24GB 级显存</b>。<br/>
  <sub>基于 <a href="https://github.com/kohya-ss/sd-scripts">kohya-ss/sd-scripts</a>，秋叶系 GUI 体验。仓库 <a href="https://github.com/storyAura/lora-scripts-story-next">storyAura/lora-scripts-story-next</a>。</sub>
</p>

<p align="center">
  <a href="https://github.com/storyAura/lora-scripts-story-next"><img src="https://img.shields.io/github/stars/storyAura/lora-scripts-story-next?style=flat-square&label=stars&logo=github&color=8b5cf6" alt="stars"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/storyAura/lora-scripts-story-next?style=flat-square&color=ec4899" alt="license"/></a>
</p>
<p align="center">
  <a href="README.en.md"><b>English</b></a>
  ·
  <a href="NOTICE.md"><b>致谢 & 许可</b></a>
</p>

---

## 与原分支的区别

本仓库（**Next Story Trainer** / `lora-scripts-story-next`）在上游 Next Trainer 产品线基础上独立演进，面向日常 Anima 训练与多任务排队，而不是镜像上游每一个实验分支。相对原分支，当前主打差异包括：

| 方向 | 本仓库 | 原分支常见定位 |
|------|--------|----------------|
| **训练队列** | 侧栏「训练队列」：多任务入队、自动传送带、历史再训 / 编辑 | 以单次提交为主 |
| **长驻稳定性** | 日志直播超长不断更、任务/日志有界、停止后杀净进程树再开训 | 偏基础任务生命周期 |
| **Anima 算法面** | 标准 LoRA / LoKr / T-LoRA、T-GLoKR、Fast 插件路径、全量微调入口；**多分辨率同时训练**（同 epoch 多档 free-fit） | 能力随上游版本变化 |
| **界面与品牌** | Next Story Trainer 品牌、中英界面词表、本仓 Releases / 联系方式 | 上游品牌与发布渠道 |
| **不收录** | 不维护上游的 **Anima Edit** 等实验分支入口 | 可能另有独立实验分支 |

日常训练请以本仓库 Releases 与文档为准；上游归属与许可见文末致谢与 [NOTICE.md](NOTICE.md)。

---

## 三步开始训练

```
1. 下载  →  从 [Releases](https://github.com/storyAura/lora-scripts-story-next/releases) 下载整合包（如 **SD-Trainer-v2.8.2.7z**），解压
2. 启动  →  双击 run_gui.bat（首次自动安装依赖 ~3 GB）
3. 训练  →  浏览器打开 http://127.0.0.1:28000，选模型、填参数、开练
```

> **要求：** Windows 10/11，NVIDIA 显卡（RTX 20+），~7 GB 磁盘。

打标模型目录、命令行训练、从旧版升级等补充说明见 **[整合包补充说明](docs/portable-getting-started.md)**。

<details>
<summary><b>从源码安装（Linux / 高级用户）</b></summary>

```sh
git clone https://github.com/storyAura/lora-scripts-story-next.git
cd lora-scripts-story-next

# Windows
run_gui.bat

# Linux
bash install.bash && bash run_gui.sh

# 可选：安装 Flash Attention 2 加速 Anima 训练
# Windows
install_flash_attn.bat
# Linux
bash install_flash_attn.sh
```

命令行训练入口：`train_anima_by_toml.sh`（标准 Anima）、`train_anima_fast_by_toml.sh`（Fast 插件）。推荐 Python **3.10**。详见 [Flash Attention 2 文档](docs/flash-attention.md)。

</details>

---

## 支持什么

| 模式 | 模型 / 脚本 | 说明 |
|------|-------------|------|
| **Anima LoRA** | LoRA · LoKr · **T-LoRA** | Flash Attention 2 / xformers / SDPA · 约 12GB 显存起 · 可选 **多分辨率同时训练**（同 epoch 多档） |
| **Anima LoRA Fast** | 仅 LoRA（进阶插件） | 可选 [anima_lora](https://github.com/sorryhyun/anima_lora) 运行时 · 建议 16GB+ · 见 [`docs/anima-fast.md`](docs/anima-fast.md) |
| **Anima 全量微调** | 完整 DiT（`anima_train.py`） | 侧栏 **全量微调 → Anima Finetune** · **约 24GB 显存**（4090 档） · 同样支持多分辨率同时训练 |
| SD 1.5 / SDXL LoRA | LoRA · LoHa · LoKr | xformers / SDPA |
| SD 1.5 / SDXL 全量微调 | Dreambooth / SDXL finetune | 侧栏 **全量微调 → Stable Diffusion** |
| Flux | LoRA | xformers / SDPA |

<p align="center">
  <img src="assets/readme/screenshot-anima-lora.png?v=20260806-nst" alt="Anima LoRA 训练界面" width="920" />
</p>

<p align="center"><sub>Anima LoRA（专家模式）— 侧栏导航、中栏模型与数据集表单、右栏配置预览</sub></p>

<p align="center">
  <img src="assets/readme/screenshot-anima-fast.png?v=20260806-nst" alt="Anima LoRA Fast 模式界面" width="920" />
</p>

<p align="center"><sub>Anima LoRA Fast — 可选 <code>sorryhyun/anima_lora</code> 插件路径；可用页内按钮或 <code>scripts/cli/install_anima_fast.*</code> 安装。见 <a href="docs/anima-fast.md">docs/anima-fast.md</a></sub></p>

<p align="center">
  <img src="assets/readme/screenshot-train-queue.png?v=20260806-nst" alt="训练队列界面" width="920" />
</p>

<p align="center"><sub>训练队列 — 多任务提交、自动传送带、历史记录（再训 / 编辑 / 删除）</sub></p>

<p align="center">
  <img src="assets/readme/screenshot-train-params.png?v=20260806-nst" alt="训练参数说明" width="920" />
</p>

<p align="center"><sub>帮助 → 训练参数说明 — 新手速查真正需要动的参数</sub></p>

---

## 训练监控

训练启动后自动打开监控页（默认端口 6008，可自动回退），GPU 状态、训练参数、Loss 曲线、预览图、日志一站式查看。

**为长时间训练加固（v2.9.1+）**：日志直播超过 1.5 万行不再断更；任务与日志历史占用有上限，挂机多天不再越吃越多内存；监控页 TensorBoard 曲线解析带缓存。

**v2.9.2**：停止后再立刻开训时，旧训练进程树会杀干净后再启动下一场，避免「双训练」导致 Loss/Epoch 乱跳；监控页 Loss、学习率与参数绑定当前任务与 TensorBoard。

**v2.9.3**：GSoKR BF16 合并路径修正；落地 Next Story Trainer 品牌素材；英文界面词表（Chrome / Schema / Help）与 Markdown 说明整块翻译；联系方式与 Github 指向本仓库；LoRA 训练页介绍对齐。

**v2.9.4**：**多分辨率同时训练**（`multires_per_image`）——Anima LoRA / Finetune 可在同一 epoch 内按多个 free-fit 档位（如 `512,1024`）各训一次；样本数按档位成倍增加，请下调 epoch。说明见 [`docs/proposal/multires_per_image_migration.md`](docs/proposal/multires_per_image_migration.md)。

<p align="center">
  <img src="assets/readme/screenshot-train-monitor.png?v=20260806-nst" alt="训练监控仪表盘" width="920" />
</p>

<p align="center"><sub>训练监控 — 状态、参数与预览图一站式查看</sub></p>

---

<details>
<summary><b>显存参考（Anima，1024 分辨率，RTX 4090 实测）</b></summary>

**Anima LoRA**

| 显存 | 配置 | 备注 |
|------|------|------|
| ≥ 24 GB | 默认参数 | 最省心 |
| ≥ 16 GB | `gradient_checkpointing` | 推荐日常 |
| ≥ 12 GB | 梯度检查点 | 稳定 |
| ≥ 10 GB | 梯度检查点 + `blocks_to_swap=16` | 速度略降 |
| ≥ 8 GB | 梯度检查点 + swap 24 + 缓存 TE + LoKr | 极限 |

**Anima 全量微调**（更新完整 DiT 权重 — 请用 WebUI **Anima Finetune**，不是 LoRA 页）

| 显存 | 配置 | 备注 |
|------|------|------|
| ≥ 24 GB | 默认 + latents/TE 缓存 | 实测专用显存约 **23–24 GB**；建议 4090 及以上 |

</details>

<details>
<summary><b>文档</b></summary>

| 主题 | 链接 |
|------|------|
| **整合包补充说明（打标 / CLI / 升级）** | [docs/portable-getting-started.md](docs/portable-getting-started.md) |
| Anima LoRA 训练指南 | [docs/anima-training.md](docs/anima-training.md) |
| **Anima Fast 模式（进阶插件）** | [docs/anima-fast.md](docs/anima-fast.md) |
| 开源归属与 NOTICE | [NOTICE.md](NOTICE.md) |
| Anima 后端（LoRA + 全量微调） | [docs/anima-backend.md](docs/anima-backend.md) |
| Anima 全量微调示例 TOML | [docs/examples/anima-full-finetune.toml](docs/examples/anima-full-finetune.toml) |
| Flash Attention 2 | [docs/flash-attention.md](docs/flash-attention.md) |
| 训练监控 & SSE 接口 | [docs/train-monitor.md](docs/train-monitor.md) |
| 打标模型目录（`tagger-models/`） | [docs/tagger-models.md](docs/tagger-models.md) |
| Docker 部署 | [docs/docker.md](docs/docker.md) |
| CLI 参数 | [docs/cli-args.md](docs/cli-args.md) |

</details>

---

## 仓库目录说明

| 位置 | 用途 |
|------|------|
| 根目录 | 仅保留契约入口 + 薄转发器，详见 [docs/repo-layout.md](docs/repo-layout.md) |
| `scripts/portable/` | 整合包启动逻辑 |
| `scripts/autodl/` | 云 GPU 运维（根目录同名文件为转发） |
| `scripts/cli/` | 命令行训练入口（普通 SD/SDXL/Flux、Anima 标准、Anima Fast） |
| `legacy/` | 打标 / notebook 等，日常可忽略 |
| `doc/local/` | 本地交接与 Issue 草稿（不上传 GitHub） |
| `docs/` | 公开文档（含 AutoDL 部署等） |

---

## 常见问题

<details>
<summary><b>无法运行 run_gui.ps1 / 未数字签名</b></summary>

推荐直接双击 `run_gui.bat`。如果一定要运行 `.ps1`：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gui_source.ps1
```

</details>

<details>
<summary><b>解压后路径嵌套两层</b></summary>

若路径出现 `...\lora-scripts-next-2.5.0\lora-scripts-next-2.5.0\`，请进入内层含 `run_gui.bat` 的目录。

</details>

<details>
<summary><b>torch 安装失败 / No matching distribution</b></summary>

**源码安装**（`run_gui.bat` 首次自动装依赖、或手动 `install-cn.ps1`）常见原因：

1. **Python 版本不对** — 需要 **3.10 或 3.11、64 位**。3.12/3.13 没有对应 CUDA 预编译包，pip 会报「找不到匹配版本」。
2. **仓库太旧** — 若脚本里仍是 `torch 2.0.x + cu118`，请 `git pull` 到最新，或改用 [Releases](https://github.com/storyAura/lora-scripts-story-next/releases) 整合包。
3. **半装坏的 venv** — 删掉项目下的 `venv` 文件夹后重装。

**不想折腾环境**：直接下载最新整合包（或 Releases 页当前最新版），解压双击 `run_gui.bat`（内置 Python，无需自装 torch）。

重装示例（PowerShell，在项目根目录）：

```powershell
Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue
py -3.10 -m venv venv
.\venv\Scripts\activate
powershell -ExecutionPolicy Bypass -File .\install-cn.ps1
```

</details>

<details>
<summary><b>打标模型放在哪 / 还要下载吗</b></summary>

- **默认模型**：`wd14-convnextv2-v2`（HuggingFace：`SmilingWolf/wd-v1-4-convnextv2-tagger-v2`，revision `v2.0`）
- **推荐路径（优先）**：项目根目录 **`tagger-models/wd14/wd14-convnextv2-v2/`**，需包含 `model.onnx` 与 `selected_tags.csv`
- **手动放置**：若 WebUI 下载失败，可从 HF / 镜像下载上述两个文件，放入上述目录后重启即可，**无需改 huggingface 缓存**
- **回退路径**：文件不完整时仍会用 `huggingface/hub/`（`HF_HOME=huggingface`）或在线下载
- **整合包**：新版 7z 已内置 `tagger-models/`，一般无需再下
- **源码**：`install-cn.ps1` 或 `python scripts/prefetch_default_tagger.py` 会写入 `tagger-models/`；`run_gui.bat` 启动前也会自动补全
- **完整说明**：[`docs/tagger-models.md`](docs/tagger-models.md)

</details>

<details>
<summary><b>整合包：能开网页但无法开始训练（旧版 v2.5.2 等）</b></summary>

请升级到 **最新 Release**；若你仍在 v2.5.2，可先参考 [`docs/portable-upgrade-2.5.2-to-2.5.3.md`](docs/portable-upgrade-2.5.2-to-2.5.3.md)，再整包更新到当前最新版。

</details>

<details>
<summary><b>整合包更新后打不开 / 启动脚本过时</b></summary>

整合包布局固定为：根目录 `run_gui.bat` + `python_embeded/` + `SD-Trainer/`。

- **用 `Update-SD-Trainer.bat` 拉代码后**：脚本会尝试刷新根目录 `run_gui.bat`；若仍失败，从新 Release 解压覆盖，或手动运行 `SD-Trainer\scripts\portable\sync_portable_root_launchers.bat`。
- **只解压过旧 7z、没有 `SD-Trainer\scripts\portable\`**：需下载新版 7z，或至少用新版替换整个 `SD-Trainer` 文件夹与根目录 `run_gui.bat`。
- 实际启动逻辑在 `SD-Trainer\scripts\portable\launch_portable.bat`，随项目更新，不要删改 `python_embeded` / `SD-Trainer` 文件夹名。

</details>

---

<details>
<summary><b>更新日志</b></summary>

| 日期 | 版本 |
|------|------|
| 2026-08-06 | **v2.9.4** — **多分辨率同时训练**（`multires_per_image`）：同 epoch 多档 free-fit；预览正提示词按行出多图 · 见 [CHANGELOG.md](CHANGELOG.md)、[`docs/proposal/multires_per_image_migration.md`](docs/proposal/multires_per_image_migration.md) |
| 2026-08-05 | **v2.9.3** — GSoKR BF16 合并修正；Next Story Trainer 品牌素材；英文 i18n（含 Schema Markdown 说明）；联系方式 / Github → 本仓库；LoRA 训练页介绍对齐 · 见 [CHANGELOG.md](CHANGELOG.md) |
| 2026-08-04 | **v2.9.2** — **严重**：停止后再开训可能双训练（旧进程未杀净）已修；监控页 Loss/学习率/参数与当前任务、TensorBoard 对齐 · 见 [CHANGELOG.md](CHANGELOG.md) |
| 2026-07-27 | **v2.9.1** — 长驻稳定性：日志直播超 1.5 万行不断更、任务/日志历史有界、监控 TensorBoard 解析缓存；**T-GLoKR** 时间步门控算法、预览 Beta 调度器、lora_type 表单修复；中文界面水合后右侧训练控件保持完整翻译 · 见 [CHANGELOG.md](CHANGELOG.md) |
| 2026-07-22 | **v2.9.0** — Anima Fast 桶分辨率控制、LoKr 配置预览修复、本地打标模型离线优先、整合包数据目录 junction |
| 2026-06-27 | **v2.8.2** — 整合包：**SDXL 训练**、**打标**、**预览图**、**训练配置导入** 四项修复；内置打标模型与 SDXL tokenizer 缓存 · 见 [CHANGELOG.md](CHANGELOG.md) |
| 2026-05-28 | **v2.7.0** — **Anima LoRA Fast 模式**（可选 `anima_lora` 插件）：WebUI 入口、页内安装、训练监控同步、性能对标与用户文档 · 见 [`docs/anima-fast.md`](docs/anima-fast.md) |
| 2026-05-28 | **v2.6.0** — **Anima 全量微调** WebUI（`anima-finetune`）、`anima_train.py` 封装、全量微调导航、监控类型修正；约 24GB 显存参考 |
| 2026-05-27 | **v2.5.3** — 便携包依赖健康检查、侧栏版本号 |
| 2026-05-21 | **v2.5.0** — UI 焕新：侧栏导航重构、首页传送门、训练监控仪表盘新增 GPU 指标；CSS 去重清理 |
| 2026-05-21 | **v2.4.0** — 训练稳定性：环境隔离、NaN 过滤、采样保护、attn_mode 降级、路径规范化；整合包 tkinter 修复 |
| 2026-05-20 | **v2.3.0** — 训练监控升级：TensorBoard 同源曲线、参数速查、日志同步 |
| 2026-05-19 | **v2.2.0** — 整合包 flash-attn 治本、闪退日志、跨盘监控 |
| 2026-05-19 | **v2.1.0** — Flash Attention 2 预编译 wheel、按步数保存 |
| 2026-05-18 | **v2.0.0** — 整合包首发、AMD 检测、bf16 修复 |

详见 [CHANGELOG.md](CHANGELOG.md)。

</details>

<details>
<summary><b>致谢</b></summary>

本仓库 fork 自 [wochenlong/lora-scripts-next](https://github.com/wochenlong/lora-scripts-next)，并继承秋叶系 GUI 与 kohya 训练栈：

[wochenlong/lora-scripts-next](https://github.com/wochenlong/lora-scripts-next) · [Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts) · [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) · [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) · [T-LoRA](https://github.com/ControlGenAI/T-LoRA) — 完整归属见 [NOTICE.md](NOTICE.md)

</details>

---

<p align="center"><sub>维护者：<b><a href="https://github.com/storyAura">@storyAura</a></b> · <a href="CONTRIBUTORS.md">贡献者</a></sub></p>
