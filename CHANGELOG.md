# 更新日志

本文件记录 **storyAura/lora-scripts-story-next** 的发行说明（v2.9.0 及更早条目沿袭上游 wochenlong/lora-scripts-next）；kohya-ss/sd-scripts 的变更请见其仓库。

---
## v2.9.4 — 2026-08-06

### 多分辨率同时训练（`multires_per_image`）

- **Anima LoRA / Finetune**：数据集设置新增「多分辨率同时训练」开关与档位列表 `target_res`（逗号分隔，至少 2 个；可选 `512 / 768 / 896 / 1024 / 1280 / 1536`）。
- **行为**：同一 epoch 内，每张源图在选中的每个 free-fit 档位各训练一次（不是分阶段切换分辨率）；开启后经典 ARB 桶参数与单目标 `resolution` 不再决定训练尺寸，由档位列表接管。
- **注意**：样本数按档位数成倍增加，请相应下调 epoch；与 UI 里的金字塔噪声 `multires_noise_*` 无关；Anima Fast 本期未接入。
- **实现**：可移植逻辑在 `vendor/multires_training`；接入说明见 [`docs/proposal/multires_per_image_migration.md`](docs/proposal/multires_per_image_migration.md)。
- **UI**：`target_res` 改为档位多选勾选框（默认勾选 512 + 1024），不再手填逗号列表。

### 预览提示词

- **多行正提示词**：`positive_prompts` 按行拆成多张预览图（不再被压成单行只出一张）；文本框带行号高亮。

---
## v2.9.3 — 2026-08-05

### 训练正确性

- **GSoKR**：去掉会在 BF16 下把微小更新「加基座再减回去」吸收掉的死 `forward`；合并权重改为 fp32 后再写回；补数值回归测试，并同步 vendored LyCORIS 到 venv。

### 品牌与发布身份

- **品牌素材**：落地 `Next Story Trainer` Logo / favicon / 首页图 / Changelog 横幅 / README 封面；公开文档见 `docs/brand-design-tokens.md`、`docs/brand-refresh-change-plan.md`。
- 全站 `sd-trainer-brand.js?v=` 与 `VERSION` 对齐（不再钉死旧的 2.8.35）；`sd-nav-i18n` 继续走 SPA 缓存键，避免双份 bundle。
- 首页 Author / Github、侧栏 Github 指向 `storyAura/lora-scripts-story-next`；关于页联系方式改为 `storyaura@outlook.com` 与 QQ 群 `917336925`。
- LoRA 训练页介绍与 hydration 对齐（Anima / Flux / Stable Diffusion），去掉旧的「新手/专家 TIP」残留。

### 英文界面（i18n）

- 外置词表：`sd-chrome-i18n-en.json` / `sd-schema-i18n-en.json` / `sd-help-i18n-en.json`；`sd-nav-i18n.js` 按需加载。
- Schema 字段说明走 Markdown 时按整块 `.markdown` 翻译，避免 `` `code` `` / `*em*` 拆节点导致中文残留。
- 品牌名保护：禁止把「下一页 → Next」子串替换进 `Next Story Trainer`。
- 队列面板、帮助页、首页简介等 Chrome 文案补齐英译。

### 验证

- 标准 Anima LoRA / LoKr / T-LoRA / 续训短步数冒烟在本机 8GB 卡上全部通过。
- 静态回归：`tests/test_contact_and_i18n_static.py`、SPA 缓存键、Changelog hydration 模板与 SSR 一致。

---
## v2.9.2 — 2026-08-04

### 严重：停止后再开训可能「双训练」

- **根因**：点停止时只杀子进程、立刻标成已终止，队列会马上启动下一场；旧的 Accelerate/训练进程仍可能占着 GPU，两边往同一 `output` / TensorBoard 写 → Loss 折返、Epoch 从 20 跳回 6、预览「最新图」其实是旧 run。
- **修复**：终止改为 `TERMINATING`，整棵进程树杀干净（Windows 用 `taskkill /T /F`）后才标 `TERMINATED`；启动前被取消的任务不会再 `Popen`；队列在终止中或进程仍存活时禁止开下一场；autosave 时间戳加微秒，降低同秒撞文件。

### 训练监控数据对齐

- 参数区优先绑定**当前任务**的 `config_path`，不再被更新的其它 autosave 抢绑。
- TensorBoard 曲线优先当前配置的 `logging_dir`；卡片 / Hero 的 Loss、学习率与 TB scalar 同源回填；Epoch 显示补全为 `当前/总轮数`。

---
## v2.9.1 — 2026-07-27

### 稳定性：长驻服务内存保险

- **训练日志直播不再断更**：SSE 游标改为绝对行号，日志超过 15000 行（环形缓冲写满）后 `/train-log` 与监控页继续收到新行；日志页面仅保留最近 5000 行 DOM，超长训练不再拖垮浏览器。
- **任务历史有界**：只保留最近 16 个已结束任务，对应日志缓冲随之释放，连续多次训练不再累积内存。
- **监控页轻量化**：TensorBoard Loss 曲线按事件文件缓存解析结果，文件未变化不再每次轮询全量重读；单次载入上限 10000 条。
- 标签编辑器 WebSocket 代理单侧断开即拆除整对连接，不再遗留挂死连接。

### 训练与预览

- **T-GLoKR**：时间步门控算法 + timestep 供给管线，`tglokr` 升为独立 LoRA 类型（顺带修活 T-LoRA）；魔改 LyCORIS 纳入 `vendor/lycoris` 并加漂移检测。
- **训练链路审查**：修复 6 个训练 bug、落地 5 项显存/IO 优化（caption dropout NaN、cpu_offload 设备崩溃、bf16 时间步等），防回归测试见 `tests/test_anima_training_fixes.py`。
- **预览采样**：新增 Beta 调度器（ComfyUI 同款 Beta(0.6,0.6)）；预览逐步注入时间步，T-LoRA / T-GLoKR 预览图与训练行为一致。
- `attn_mode` 新增 `mem_efficient`（PyTorch Efficient SDPA）。

### 配置与 UI

- 修复 lora_type 切换 / 读取历史参数时残留 `network_module`、`lycoris_algo` 导致表单消失、提交必败的问题；训练值由 adapter 权威派生。
- 本地扩展 LyCORIS 算法参数全部可视化，默认即推荐配置；帮助菜单新增「训练算法说明」「训练参数说明」站内页面。

### 项目

- 仓库与项目更名为 **lora-scripts-story-next**（个人分支，便于与上游 wochenlong/lora-scripts-next 区分）。

---
## v2.9.0 — 2026-07-22

### Anima Fast

- 高于 1024 的训练分辨率会自动计算 `max_bucket_reso`，并开放最小/最大桶分辨率、桶步长与禁止放大选项（#196）。
- 用户手动填写的最大桶分辨率过小时，在训练启动前给出明确提示；非桶步长倍数会自动向上调整。
- Windows 安装 Fast 插件时，uv 默认使用系统证书库；`UnknownIssuer` 会显示代理、杀毒软件和旧版 uv 排障提示（#195）。

### LoKr 配置

- 修复 Anima 标准模式 LoKr 参数预览与下载配置中出现 `conv_dim=undefined` 等无效值的问题（#186）。
- 清理逻辑仅在 Anima 标准模式页加载，支持 SPA 跳转，不修改共享前端主 bundle。

### 本地打标

- `wd-vit-v3`、`wd14-moat-v2` 优先使用本地模型目录，不再在本地文件齐全时访问 Hugging Face（#194）。
- 兼容旧目录名称，并支持直接使用完整的 Hugging Face 本地缓存。

### Windows 整合包

- 用户数据以 `SD-Trainer/sd-models`、`output`、`logs`、`train` 为实际目录，整合包根目录保留兼容 junction。
- 启动时自动迁移旧版反向 junction；整合包移动到新路径后会修复失效 junction，并保留已有数据。
- 生成 7z 时不再跟随数据目录 junction，避免离线打标模型被重复打包。

### 文档

- 新增面向 SDXL 用户的 Anima LoRA 参数迁移说明（#193）。

---
## v2.8.35 — 2026-06-28

### 整合包（更新脚本 hotfix + 版本号）

相对 v2.8.3 整合包 7z，本版 **`VERSION` / UI 芯片为 v2.8.35**，便于区分已含更新脚本修复的构建；**无前端 layout/app bundle 变更**。

- **`Update-SD-Trainer.bat` / `Update-SD-Trainer-Release.bat`**：路径引号、`CRLF`、PowerShell 5.1 UTF-8 BOM（`UPDATER_VERSION` 4）
- 新增 **`Fix-Portable-Bats.bat`**
- 仍含 v2.8.3 后端/Hub/junction 等更新；SPA 仍为稳定 dist（`20260627-config-import`）

---
## v2.8.3 — 2026-06-28

### 后端与整合包（无前端 layout 变更）

相对 v2.8.2，本版 **仅版本号与后端/便携逻辑** 更新；**未** 重新合入会破坏表单挂载的 LoKr 前端 dist patch（#189 已回滚）。

#### 整合包：更新脚本 hotfix（同 VERSION 重发）

- **`Update-SD-Trainer.bat` / `Update-SD-Trainer-Release.bat`**：修复 `%PORTABLE_ROOT%` 尾部 `\` 导致 PowerShell `Illegal characters in path`。
- **`.bat` CRLF**：打包与 Release 合并时强制 Windows 换行；新增 **`Fix-Portable-Bats.bat`** 修复 LF-only / UTF-8 BOM 问题。
- **`.ps1` UTF-8 BOM**：兼容 Windows PowerShell 5.1 解析中文；bootstrap 在本地 **`UPDATER_VERSION`** 更高时不再被 GitHub main 旧脚本覆盖。
- **`update_from_release.ps1`**：合并 Release 后自动修复根目录 `.bat` 换行。

#### LoKr / 配置导出（#186，API only）

- 新增 **`POST /api/config/normalize-for-export`**，供后续前端或脚本统一导出规范化。
- 导入时跳过 `undefined` / `null` 等无效 LyCORIS 标量。

#### 整合包：打标与 Hub（#188）

- **`MIKAZUKI_HUB_BACKEND=auto`**；`SmilingWolf/*` 等非魔搭模型直链 Hugging Face。
- 打标下载终端进度与中文错误提示。

#### 整合包：内置文件选择器（#191）

- 启动/打包时将外层 `sd-models`、`output`、`logs`、`train`、`tagger-models` 联接进 `SD-Trainer/`。

### 说明

- UI 版本芯片显示 **v2.8.3**；SPA 业务 bundle 仍为 **v2.8.2 稳定 dist**（`20260627-config-import`）。
- LoKr 下载/预览走 API 的前端 patch 待单独 PR 验证后再发。

---
## v2.8.2 — 2026-06-27

### 整合包更新要点

面向 Windows 便携整合包 **SD-Trainer-v2.8.2.7z**（`PORTABLE_BUILD` **`2874ad1`**，约 **392 MB**）。相对 v2.7.0 / 旧版整合包，本版重点修复下列训练与 WebUI 路径：

#### SDXL 训练修复

- 修正 WebUI **`sdxl-lora`** 路由，对接当前 vendored SDXL 训练脚本栈（#146，自 v2.8.0 起）。
- 训练子进程启动允许使用 user-site 的 `torch` / `accelerate`，避免整合包环境下误报「训练接口网络请求失败」（#164）。
- 整合包预置 **`tokenizer-cache/`**（CLIP / T5 等），SDXL / Flux 离线训练不再依赖首次联网拉 tokenizer。

#### 打标修复

- 默认 WD 打标模型 **wd14-convnextv2-v2** 随包预置于 **`tagger-models/wd14/`**，「数据集打标」开箱即用。
- 打标加载增加 ONNX Runtime 诊断、CUDA 失败 CPU 回退与超时保护，避免无限 loading（v2.8.0）。
- 经典 / 原生标签编辑器空页渲染与侧栏入口修正（#165 等）。

#### 预览图修复

- 旧 autosave / 导入 TOML 缺少 **`enable_preview`** 时，前后端自动推断并保留 `sample_prompts` 等预览字段（#179、#166）。
- Anima Fast 开启预览后 **`sample_at_first`** 默认 true，确保至少出一张样图（#160）。
- Fast 环境安装完成后 **自动刷新页面**，安装进度区不再长期遮挡右侧参数预览（#180）。

#### 训练配置导入修复

- 配置文件 **全量替换导入** 时保留数值类型（整数 / 浮点不再被误转成字符串）（#171）。
- 导入校验补全 **`anima-lora`** 页规格，并与当前 schema 对齐（#179）。
- 前端参数预览 / 下载 TOML 序列化修复：union 分支字段回填、`network_args` 安全展开（#179）。

### 整合包说明

- 下载：**[Releases → SD-Trainer-v2.8.2.7z](https://github.com/wochenlong/lora-scripts-next/releases/tag/v2.8.2)**，解压后双击 **`run_gui.bat`**。
- **Anima Fast** 仍不预装插件 venv；首次在 Fast 页内安装。安装成功后会自动刷新以显示完整参数区。
- 已装旧版整合包：推荐 **`Update-SD-Trainer-Release.bat`** 合并最新 Release；Git 更新用 **`Update-SD-Trainer.bat`**。
- 用户向补充说明（打标目录、命令行训练、升级）：[`docs/portable-getting-started.md`](docs/portable-getting-started.md)

### 发版前验证（已通过）

- 默认 WD 打标、SDXL LoRA 冒烟训练、Anima Fast 安装 + 训练、旧 autosave TOML 导入后参数预览与下载配置。

---
## v2.8.1 - 2026-06-25

### Release blockers

- **Standard LoRA launch**: allow user-site `torch` / `accelerate` when source installs need them, avoiding training subprocess startup failures that surfaced as “training endpoint network request” errors.
- **Anima Fast preview**: infer preview enablement from strict prompt or sampling-interval signals when `enable_preview` is dropped or falsified by schema serialization; keep `sample_at_first` defaulting to true for enabled previews.
- **LoKr bf16 guardrails**: disable known-problematic LoKr bf16 weight-decomposition paths and avoid full-half precision for `full_matrix` adapters to work around upstream LyCORIS dtype issues.
- **Native tag editor**: keep the experimental native editor route available for direct testing, but hide it from the default sidebar until it is hardened; fix its empty-page render path.

### Packaging notes

- Build the 2.8.1 portable package only from a clean `origin/main` including PRs #160, #163, #164, #165, and #166.
- Pre-release validation must include: standard LoRA launch from the portable Python environment, Anima Fast preview prompt/TOML generation, default legacy tag editor entry, and hidden native editor sidebar entry.

---
## v2.8.0 - 2026-06-19

### Release blockers

- **Tagger startup**: hardened local WD14 ONNX loading with model-size checks, ONNX Runtime provider diagnostics, CPU fallback, and a bounded load timeout so a bad CUDA/session init no longer leaves users at an endless loading state.
- **Legacy tag editor**: restored the legacy Gradio Dataset Tag Editor as a default startup path for existing users while keeping `--disable-tageditor` as the opt-out switch.
- **SDXL LoRA training**: fixed the WebUI/API `sdxl-lora` route to use the vendored SDXL trainer path that matches the current sd-scripts strategy stack. Verified through `/api/run` -> `process.run_train()` -> TaskManager -> accelerate -> `vendor/sd-scripts/sdxl_train_network.py`, completing a 20-step SDXL smoke run and writing a LoRA `.safetensors` artifact.

### Packaging notes

- Build the 2.8.0 portable package only from a clean `origin/main` that includes the SDXL route fix from #146.
- Pre-release validation must include: default WD14 tagging from bundled `tagger-models/`, legacy `/tageditor.md` + `/proxy/tageditor/`, and a real SDXL LoRA smoke training run that reaches training steps and saves a model.

---

## v2.7.1 — 2026-06-14

### Bug 修复（前端 dist）

- **#121** 侧栏中英文切换不再丢失：语言选择由 `sessionStorage` 改存 `localStorage`，关闭重开浏览器后保持；启动时一次性迁移旧 `sessionStorage` 值并清理，检测逻辑改为纯读取无副作用。
- **#121** 英文模式下底部主题按钮残留中文「灯泡」：新增 `灯泡 → Theme` 映射，将 `.sidebar-bottom` 纳入文本替换范围，并翻译主题按钮 `title`（`toggle color mode`）提示。

### 资源版本

- `sd-nav-i18n.js` / `sd-trainer-brand.js` 的 cache 查询参数随 `VERSION` 统一为 `?v=2.7.1`（由 `scripts/patch-ui-brand-version.py` 生成），确保旧缓存的客户端加载到修复后的脚本。

---

## v2.7.0 — 2026-05-28

### Anima LoRA Fast 模式（可选插件）

- **训练入口**：WebUI 侧栏「Anima LoRA → Fast 模式」（`/lora/anima-fast.html`），`model_train_type: anima-lora-fast` 路由至 `extensions/anima_lora/` 独立 venv 与 `train.py`。
- **页内安装器**：一键克隆/快照上游 [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora)（MIT），创建 cu130 插件环境；未就绪时拒绝开训。
- **训练监控**：Loss / ETA / Epoch 与 Fast 专用 `*.progress.jsonl` 同步；预览图按活动任务 `output_dir` 发现。
- **文档与对标**：[`docs/anima-fast.md`](docs/anima-fast.md)、[`docs/examples/anima-lora-benchmark-*.toml`](docs/examples/)；4090 同参约 **2.5×** 加速（标准 Kohya ≈7.1 s/step vs Fast ≈2.8 s/step）。
- **归属**：[`NOTICE.md`](NOTICE.md) § Anima LoRA Fast Mode；Fast 页与文档致谢 upstream。

### 前端（dist）

- Fast 页安装引导、开源致谢（`anima-fast-credit`）；首页/更新日志 v2.7.0 条目。

### 整合包说明

- **不预装** Fast 插件 venv（`extensions/anima_lora/.venv`）；用户首次在 Fast 页点击「开启插件」安装，避免 7z 体积暴增。

### 环境变量

- `LORA_ENABLE_ANIMA_FAST=1`（默认开启 Fast 入口；设为 `0` 可隐藏侧栏与 API）。

### v2.7.0 整合包热更新 — 2026-06-02（第三次重发）

> 同 tag **v2.7.0** 重发 7z（`PORTABLE_BUILD` **`65df2ba`**，约 **381.3 MB**）。相对 **`18e15cc`** 构建新增下列内容；更早相对初版 7z 的修复见上一节历史说明。

#### 整合包 / 更新器

- **Bootstrap 自更新（`UPDATER_VERSION=2`）**：运行 `Update-*.bat` 前先从 GitHub `main` 同步最新更新脚本（SHA256 比对），有变更则自动重启后再执行 Git / Release 更新；网络失败回退本地 bundled 脚本。
- **版本信息展示**：更新开始时打印 **本地 VERSION / PORTABLE_BUILD / git**、**线上 main VERSION**、**最新 Release**、**本地 vs 线上 UPDATER_VERSION**。
- 团队约定与变迁索引：[Discussion #73](https://github.com/wochenlong/lora-scripts-next/discussions/73)。

#### Bug 修复（相对 `18e15cc` 7z 新增）

- **#72** Anima Fast 插件安装：`EnvironmentInstallPlan` 为 frozen dataclass，改用 `dataclasses.replace` 设置 `source_root`，修复 `cannot assign to field 'source_root'`。

#### 升级指引

| 场景 | 操作 |
|------|------|
| 已装 **v2.7.0 整合包**（任意 `PORTABLE_BUILD`） | **`Update-SD-Trainer-Release.bat`**（推荐）或 **`Update-SD-Trainer.bat`** |
| 从 v2.5.x / v2.6.x | 保留用户数据目录后 Release 更新或下载本 7z |
| 无 `.git` 的旧包 | **Release 更新** 或整包下载 |

更新后确认 **`SD-Trainer/PORTABLE_BUILD`** 第一行为 **`65df2ba`**，`scripts/portable/UPDATER_VERSION` 为 **`2`**。

### v2.7.0 整合包热更新 — 2026-06-01（第二次重发，已 supersede）

> 同 tag **v2.7.0** 重发 7z（`PORTABLE_BUILD` **`18e15cc`**，约 **381.6 MB**）。初版 Release 7z 为 `7841f19` 前后构建，不含下列修复。**已被 2026-06-02 `65df2ba` 构建取代。**

#### 整合包 / 更新器

- **同 VERSION 重发无法更新**：Release 合并去掉 `robocopy /XO`，改用 `/IS /IT`，确保重发 7z 能覆盖本地较新的文件时间戳。
- 新增 **`SD-Trainer/PORTABLE_BUILD`**（git short SHA + `built_at` + version），便于对比是否已同步最新构建。
- Release 更新后写入 **`config/.portable_release_sync.json`**（Release 资产 id / 更新时间）。
- 修复 **`update_from_release.ps1`** PowerShell 字符串解析（尾随 `\`、方括号等导致脚本后半段未执行）。
- Release 合并后**显式同步** `SD-Trainer/.git`；打包时写入 `scripts/portable/templates/Update-*.bat`，避免根目录仍为 v2.5 时代「仅 `git pull`、无 `.git` 检查仍显示完成」的旧脚本。

#### Bug 修复（本重发包相对初版 7z 新增）

- **#66** Anima Fast：SPA 进入 Fast 页侧栏「开启插件」、安装报 `Anima source root does not exist`、CLI 安装后一直「检查中」；统一 `source_root` / 安装前 clone 逻辑。
- **#71** SDXL LoRA 使用 State resume 报 `KeyError: 'step'`；将 accelerate resume step fallback port 到 `scripts/stable/`。
- **#70** 训练页移动端布局堆叠与导航折叠按钮。
- **#69** 预览图 multiline sample prompt 文件合并修复。
- **#68** Anima Finetune 训练监控 Loss 曲线显示。
- **训练**：`mixed_precision` 从 TOML 正确转发到 accelerate launch。

#### 升级指引

| 场景 | 操作 |
|------|------|
| 已装 **初版 v2.7.0 整合包** | **`Update-SD-Trainer-Release.bat`**（推荐）或 **`Update-SD-Trainer.bat`**（Git 快进；有本地差异时会自动 stash） |
| 从 v2.5.x / v2.6.x | 保留 `sd-models/`、`output/`、`logs/`、`SD-Trainer/extensions/`，解压新版或 Release 更新 |
| 无 `.git` 的旧包 | 必须用 **Release 更新** 或下载本 7z |

更新后确认 **`SD-Trainer/PORTABLE_BUILD`** 第一行为 **`18e15cc`**。

---

## v2.6.0 — 2026-05-28

### Anima 全量微调（Finetune）

- **训练入口**：WebUI 侧栏「全量微调 → Anima Finetune」（`/lora/anima-finetune.html`），`model_train_type: anima-finetune` 路由至 `scripts/dev/anima_train.py`（上游 `anima_train.py`）。
- **Schema 与适配**：新增 `mikazuki/schema/anima-finetune.ts`；`adapt_anima_config(finetune=True)` 剥离 LoRA 网络字段；默认学习率 `1e-5`。
- **导航与首页**：「全量微调」分组（Anima Finetune 在 Stable Diffusion / Dreambooth 之前）；首页 portal、新手上路引导同步。
- **训练监控**：识别 `anima_train.py` 时显示 **Anima Finetune**（不再误标为 Anima LoRA）。
- **文档与示例**：`docs/anima-backend.md`、`docs/examples/anima-full-finetune.toml`；单元测试覆盖路由、wrapper、adapter、监控类型推断。

### 前端（dist）

- 修复 SPA 页面组件 `i0` 映射缺失导致的右栏 **404 Not Found**。
- Anima Finetune 页右栏标语与说明文案（进阶玩家、充足样本与高显存）。

### 实测参考

- RTX 4090 24GB、1024 分辨率：全量微调专用显存约 **23–24 GB**（与 LoRA 的 12 GB 档不同，需单独规划显存）。

---

## v2.5.3 — 2026-05-27

### 整合包热修复（[#54](https://github.com/wochenlong/lora-scripts-next/issues/54)）

- **依赖健康检查**：便携启动不再仅以 `torch` 目录是否存在判断「已安装」；启动前会探测 `torch`、`torchvision`、`accelerate`、`diffusers`、`gradio` 等关键包，不完整时自动执行 `setup_environment.py` 修复安装，缓解「网页能开、点开始训练提示无法连接训练端」。
- **侧栏版本号**：WebUI 侧栏「Next Trainer」旁显示当前版本（读取 `/api/version`），便于确认是否已升级到 2.5.3。

### 升级说明

- **v2.5.2 整合包用户请整包升级到 v2.5.3**（不要覆盖 `sd-models/`、`output/`、`config/` 等用户目录）。详见 [`docs/portable-upgrade-2.5.2-to-2.5.3.md`](docs/portable-upgrade-2.5.2-to-2.5.3.md)。

---

## v2.5.2 — 2026-05-25

### 整合包修复

- **Git 更新可靠性**：`Update-SD-Trainer.bat` 对浅克隆仓库会自动 `--deepen=50` 补齐历史，避免新版整合包更新时报 `fast-forward update failed`。
- **GitHub 网络回退**：主仓 fetch 与 `dataset-tag-editor` 子模块更新均支持直连、`ghfast.top`、`ghproxy`、`gitmirror` 多路回退，缓解国内网络 `Connection was reset`。
- **子模块容错**：整合包已内置 `dataset-tag-editor` 文件时，更新脚本会直接复用已有文件，避免 Git 因“目录已存在且非空”导致子模块 clone 失败。
- **启动脚本路径修复**：修复 `launch_portable.bat` 与 `sync_portable_root_launchers.bat` 相对路径层级错误，确保根目录 `run_gui.bat`、`run_gui_portable.bat` 能被正确刷新。
- **PyTorch 下载源测速**：首次安装依赖时改为按实际 wheel 下载吞吐量（最多 32MB / 15 秒）选择 PyTorch 源，避免直连快的用户被误切到慢速国内镜像。
- **tkinter 打包说明**：明确整合包需要完整 CPython 3.10 的 Tcl/Tk 文件，避免文件/目录选择器不可用。

### 训练稳定性

- **SDXL 训练签名兼容**：同步 `assert_extra_args` 参数签名，修复新版 `sd-scripts` 下 SDXL LoRA / Textual Inversion 训练启动时报 `TypeError`。
- **Windows torch_compile 保护**：Windows 上自动禁用 `torch_compile` / `dynamo_backend`，避免 PyTorch 编译路径依赖 Triton 导致训练中断。

### 标签编辑器

- **默认可用性恢复**：源码和整合包用户默认启用原生标签编辑器入口。
- **启动自修复**：标签编辑器缺失时会尝试自动初始化子模块；嵌入式 Python 环境下通过 bootstrap 修复 `sys.path`，避免 `/proxy/tageditor` 404。

---

## v2.5.0 — 2026-05-21

### UI 焕新

- **侧栏导航重构**：新增分组式侧栏，训练类型（LoRA / Dreambooth）、工具（Tensorboard / 数据集打标 / 标签编辑）、帮助文档等分区清晰，支持层级折叠。
- **首页传送门**：新增 Next Trainer 首页，卡片式入口快速跳转到训练、监控、新手上路等常用功能。
- **训练监控仪表盘**：新增 GPU 实时指标（型号、负载、显存、温度、功耗），总步数大字卡片，训练参数速查（学习率、优化器、调度器、Rank/Alpha、分辨率、精度）。
- **新手上路页面**：新增指南页，帮助新用户快速了解训练流程。
- **CSS 去重清理**：清理 PR 合并产生的 7 倍重复 CSS 规则（~1660 行），`sd-trainer-ui-polish.css` 和 `style.css` 均已精简。
- **README 截图更新**：替换为最新 UI 截图（WebUI 三栏布局、训练监控仪表盘、Loss 曲线 + 预览图、训练日志）。

### 改进

- **训练监控前后端分离**：`train_status_server.py` 拆分为 `train_monitor/` 目录（`server.py` + `index.html` + `monitor.css` + `monitor.js`），便于独立维护和迭代。

---

## v2.4.0 — 2026-05-21

### 训练稳定性（整合包 + 源码）

- **训练子进程环境隔离**：设置 `PYTHONNOUSERSITE=1`，防止系统用户级 site-packages（如残缺的 sklearn）污染训练子进程，修复 `No module named 'joblib'` 等 import 链断裂崩溃（[#16](https://github.com/wochenlong/lora-scripts-next/issues/16)）。
- **NaN 值过滤**：`network_args` / `optimizer_args` 中 `key=NaN` 的无效项现在被自动剥离，修复 LyCORIS `int("NaN")` 导致训练崩溃。
- **采样保护**：若最终配置无 `sample_prompts`，自动移除 `sample_at_first` 等采样参数，避免 sd-scripts 在 step 0 因 `sample_prompts=None` 崩溃。
- **attn_mode 自动降级**：配置中指定 `xformers` / `flash` 但对应后端未安装时，自动降级到可用方案（xformers → torch SDPA），并打印 WARNING 而非直接崩溃。
- **路径规范化**：配置中的模型/数据/输出等路径字段自动将 `\` 转为 `/`，修复 Windows 手动粘贴路径时反斜杠导致的兼容性问题。

### 整合包改进

- **tkinter 支持**：`build_portable.ps1` 打包时自动复制 tkinter + Tcl/Tk，修复文件夹选择器（`/pick_file`）无法弹出（[#20](https://github.com/wochenlong/lora-scripts-next/issues/20)）；缺失时 API 返回明确错误。
- **xformers 一键安装**：新增 `install_xformers.bat`，整合包用户双击即可安装 xformers 0.0.30。
- **config.json 启动修复**：空文件不再导致 JSON 解析报错。

---

## v2.3.0 — 2026-05-20

### 训练监控体验升级

- **TensorBoard 同源 Loss 曲线**：6008 训练监控页改为读取 TensorBoard event scalar，默认展示 `loss/average`、`loss/current`、`loss/epoch_average` 与 `lr/unet` 四宫格曲线，避免相对 Loss 曲线含义不清。
- **曲线交互优化**：移除小图底部滑动条，改为 `全部 / 最近 50% / 最近 20% / 最近 10% / 恢复最新` 视野按钮；保留滚轮缩放与拖拽平移。
- **训练参数速查**：监控页顶部显示学习率、优化器、总步数、分辨率、保存频率、精度、seed 等关键参数，方便启动后快速核对配置。
- **终端日志同步**：训练日志同时输出到 CMD 终端与 6008 监控页；终端 echo 失败不会影响监控页日志采集。
- **后台更干净**：静默 6008 监控页正常轮询的 `200/304` access log，只保留真实异常与训练输出。

### 启动稳定性

- **端口冲突回退**：GUI、TensorBoard 与训练监控启动前会严格检测端口；当 6008 被占用时自动切换到可用端口，并避免多个子服务 fallback 到同一个端口。
- **清理测试入口**：移除测试用 `run_gui_anima.bat`，正式包统一使用 `run_gui.bat` 启动。

---

## v2.2.0 — 2026-05-19

### 整合包与启动

- **flash-attn / triton（治本）**：便携包不再安装 `flash-attn`；启动时自动卸载已装但不可用的 `flash-attn` / `triton`；训练使用 **xformers** 或 **PyTorch SDPA**；子进程设置 `TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa`，避免 `No module named 'triton'`（[#14](https://github.com/wochenlong/lora-scripts-next/issues/14) 相关）。
- **triton-windows**：便携包嵌入式 Python 不再安装/保留 `triton-windows`，修复因 triton 编译失败导致的崩溃。
- **run_gui.bat**：纯 cmd 启动（不依赖 `run_gui.ps1`，避免 PowerShell 执行策略报错）；增加 `sd-trainer-log.txt` 启动日志；失败时明确提示日志路径。
- **requirements.txt**：修复 PEP 508 环境标记在 `launch_utils` 中的解析（[#13](https://github.com/wochenlong/lora-scripts-next/issues/13)）。

### 训练监控与 UI

- **跨盘 output_dir**：监控页（6008）在输出目录位于其他盘符时不再断联（[#12](https://github.com/wochenlong/lora-scripts-next/issues/12)）。
- **品牌**：前端作者/链接改为本项目；临时 logo 与 favicon；监控页页头显示 logo。
- **CONTRIBUTORS.md**：贡献者单独文档；README 精简致谢链接。

---

## v2.1 — 2026-05-09

### 训练监控页（端口 6008）

- **Loss 趋势图**：参考 Weights & Biases 风格——16:10 比例、`preserveAspectRatio` 保持比例、网格与坐标轴刻度、100% 基线强调、曲线末端数值标注。
- **指标侧栏**：当前 / 最低（含 step 提示）/ 初始 / 累计下降 / 最近 Δ（着色）/ 趋势 pill；与曲线底部对齐，宽屏下主体仍为左侧曲线。
- **响应式**：窄屏（约 820px 以下）单列堆叠。
