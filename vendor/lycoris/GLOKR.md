# GLoKR 算法报告与使用说明

> 版本:2026-07-24 重构版(`modules/glokr.py`)
> 适用:DiT / Transformer 类模型(仅 Linear 层),LyCORIS 训练框架
> 验证:19 项自动化测试全部通过(见 [测试与验证](#8-测试与验证))

---

## 目录

1. [算法概述](#1-算法概述)
2. [数学原理](#2-数学原理)
3. [重构前的问题](#3-重构前的问题)
4. [修复与优化清单](#4-修复与优化清单)
5. [参数完整说明](#5-参数完整说明)
6. [使用说明](#6-使用说明)
7. [存档格式](#7-存档格式)
8. [测试与验证](#8-测试与验证)
9. [性能特征](#9-性能特征)
10. [已知边界与注意事项](#10-已知边界与注意事项)

---

## 1. 算法概述

GLoKR = **G**eneralized **LoK**onecker adapter,是 LoKr(Kronecker 积低秩适配)与
GLoRA(乘性重参数化)的融合体,并可叠加 DoRA / BoRA 权重解耦。

一句话:用极少的参数,同时学**三种**权重更新——

| 路径 | 形式 | 直觉 | 擅长 |
|------|------|------|------|
| **B 路**(增量) | `ΔW += B` | 往画布上叠新贴纸 | 新纹理、新笔触方向 |
| **A 路**(输入侧 G 路径) | `ΔW += W₀·A` | 在原镜头前加滤镜 | 重组已有特征,结构安全 |
| **C 路**(输出侧 G 路径,可选) | `ΔW += C·W₀` | 调整输出的混色方式 | 色调 / 氛围类全局风格 |

三条路径全部用 Kronecker 积参数化(参数量 ~O(√维度)),初始为零增量(不扰动底模)。
A/C 路的贡献只能落在 W₀ 的行/列空间内(数学上 col(W₀A) ⊆ col(W₀)),
这是一种"贴着预训练特征走"的归纳偏置;需要全新方向时由 B 路承担。

---

## 2. 数学原理

### 2.1 权重更新

$$
\Delta W \;=\; s_b\!\sum_{i=1}^{r}\! m_i\,(B_1^{(i)}\!\otimes B_2^{(i)})
\;+\; \frac{s_a}{\lVert W_0\rVert}\,W_0\!\sum_{i}\! m_i\,(A_1^{(i)}\!\otimes A_2^{(i)})
\;+\; \frac{s_c}{\lVert W_0\rVert}\!\sum_{i}\! m_i\,(C_1^{(i)}\!\otimes C_2^{(i)})\,W_0
$$

- `⊗` 为 Kronecker 积;`r = kron_rank`,多项求和打破单个 Kron 积"奇异值网格"的僵硬谱形,带 `1/√r` 稳定器
- `mᵢ` 为可学习混合系数(`train_gates=true` 且 `r>1` 时),默认全 1
- `s_b, s_a, s_c` = `alpha/dim`(或两因子全满秩时为 1),可各乘一个可学习门控 `gate_b/a/c`
- `‖W₀‖` 在初始化时计算并**持久化进存档**(`g_norm` 键),换底模不漂移;可选谱范数模式

各因子形状(`in = in_m×in_n`,`out = out_l×out_k`,由 `factorization` 自动分解):

| 路径 | w1 形状 | w2 形状 | Kron 结果 |
|------|---------|---------|-----------|
| B | (out_l, in_m) | (out_k, in_n) | (out, in) |
| A | (in_m, in_m) | (in_n, in_n) | (in, in) |
| C | (out_l, out_l) | (out_k, out_k) | (out, out) |

`lora_dim` 低于阈值时 w1/w2 进一步分解为低秩积(`w1 ≈ w1_a @ w1_b`),阈值逻辑与 LoKr 一致。

### 2.2 初始为恒等

所有路径的 w2(或 w2_b)初始化为 0,因此 `ΔW = 0`,训练起点等于底模。
`use_scalar=true` 时改用全局标量 `scalar=0.01` 起步(与 LoKr 同款语义)。

### 2.3 权重解耦(DoRA / BoRA)

- **DoRA**(`weight_decompose=true`):`W' = dora_scale · (W₀+ΔW)/‖W₀+ΔW‖_row`,幅度与方向解耦
- **BoRA**(`use_bora=true`):行、列两个方向依次做范数归一 + 可学习缩放;
  `bora_iters>1` 时做 Sinkhorn 式交替平衡,消除"先行后列"的顺序偏置
- 出图强度 `multiplier ≠ 1` 时,解耦缩放按 `m·(scale−1)+1` 插值(与 LoKr 家族一致),
  **`multiplier=0` 严格还原底模**

### 2.4 两种前向模式

- **合并模式**(默认):重建 `ΔW` 后 `y = org_forward(x) + x·ΔWᵀ`;
  wd 路径的减法在 fp32 中完成,只把小差值降回低精度(bf16 安全)
- **bypass 模式**(`bypass_mode=true`,量化底模自动强制):分组算法直接作用于激活,
  不在显存中重建大矩阵;与合并模式数学等价(测试锁定)

### 2.5 NKP 初始化(`init_mode="nkp"`)

对 W₀ 做 Van Loan 最近 Kronecker 积分解(重排矩阵的 SVD),
让第 i 个 kron 项的 `b_w1` 从 W₀ 的第 i 个 Kron 主成分方向出发(幅度归一到 kaiming 期望),
`a_w1` 用单位阵(组内重混起步)。w2 仍为 0,**起点仍是恒等**,
但梯度第一步就流入有意义的子空间——PiSSA 思路在 Kron 结构上的移植。
每层一次 `svd_lowrank`,建网仅慢数秒。

---

## 3. 重构前的问题

2026-07-23 版本(重构前)经代码审查 + 冒烟测试确认的问题:

| # | 严重度 | 问题 | 后果 |
|---|--------|------|------|
| 1 | 致命 | `make_module_from_state_dict` 用 `**weights` 签名,框架按位置传参 | 练完的文件加载即 `TypeError`,**无法出图/分发** |
| 2 | 致命 | 检测按精确键名匹配,`kron_rank≥2` 存档键带 `_0` 后缀 | 多项存档完全不被识别 |
| 3 | 致命 | 未实现 `get_merged_weight` | `merge_to`/`onfly_merge`/`parametrize` 全部崩溃;wd 合并数学缺失 |
| 4 | 高 | 参数 `copy_` 不在 `no_grad` 下 | 直接调用重建接口报 RuntimeError |
| 5 | 高 | `forward` 计算 `(W₀+ΔW)−W₀`,bf16 下小增量被吸收 | 比底模小 ~256 倍的细节增量在前向中被四舍五入抹除 |
| 6 | 中 | `apply_weight_decompose` 忽略 multiplier | 强度滑条对 DoRA/BoRA 无效,0 回不到底模 |
| 7 | 中 | G 路径尺度分母 `‖W₀‖` 不持久化 | 换底模挂载时强度悄悄漂移 |
| 8 | 中 | 续训:参数存于 ParameterList,自然键名与存档键不符 | `load_weights` 静默加载失败 |
| 9 | 中 | `bokr` 检测键与 `lokr` 完全相同且排序靠后 | BoKR 存档被误认成 LoKr,BoRA 缩放**静默丢弃** |
| 10 | 低 | rank_dropout 在合并/bypass 两模式行为不一致(bypass 的 A 路无 dropout) | 两种模式训练目标不同 |
| 11 | 低 | `apply_max_norm` 每步全量重建(含 W₀@A 大矩阵乘) | 开 `scale_weight_norms` 时开销翻倍 |
| 12 | 低 | `_bypass_forward_diff_a/_b` 及参数构建大段复制粘贴 | 阈值逻辑两份,易漂移 |

---

## 4. 修复与优化清单

### 4.1 正确性修复(无需任何配置,自动生效)

| 项 | 原本 | 现在 |
|----|------|------|
| 存档加载 | TypeError 崩溃 | `algo_check` 前缀匹配 + `extract_state_dict` 返回键值字典;重建输出与训练时逐位一致 |
| kron_rank≥2 | 不被识别 | 后缀键正确检测、加载、续训 |
| 合并烘焙 | NotImplementedError | 实现 `get_merged_weight`(wd 解耦正确应用),`merge_to` 前后输出一致 |
| 续训 | 键名不匹配 | 参数属性名 == 存档键名,`load_state_dict`/`load_weights` 天然打通 |
| 强度滑条 | wd 不响应 multiplier | 按 `m·(scale−1)+1` 插值,`multiplier=0` 严格还原底模 |
| BoKR 误认 | 静默丢 BoRA 缩放 | `bokr.algo_check` 要求 `bora_r_scale` 键,`MODULE_LIST` 中排到 LoKr 前 |
| gsokr 串台 | —(新风险) | glokr 检测排除含 `sora_cp/sora_dp` 的存档 |

### 4.2 数值修复(对画风细节直接相关)

- **bf16 吸收**:非 wd 路径直接使用 `ΔW`(不再做 `+W₀−W₀` 往返);
  wd 路径在 fp32 中完成解耦与减法,只把小差值降精度。
  实测 wd 增量误差比旧算法小 3 倍以上(典型约一个数量级)。
  细节 = 小幅度权重增量,此修复让 loss 从第一步起就能"看见"它们。

### 4.3 语义与可复现性

- `g_norm`(‖W₀‖)持久化进存档;旧存档自动回退为"现算当前底模范数"
- `bora_iters` 持久化,加载后行为与训练时一致
- rank_dropout 两种前向模式行为统一(所有路径、两种模式一致)

### 4.4 画风增强(全部默认关闭,开关启用)

| 开关 | 默认 | 作用 |
|------|------|------|
| `train_gates` | false | 每路径 1 个可学习门控(G/B 配比逐层自适应);`kron_rank>1` 时附送逐项混合系数 `kron_mix` |
| `init_mode="nkp"` | "kaiming" | Van Loan 主成分对齐初始化,短周期训练收敛更快 |
| `bora_iters=N` | 1 | BoRA 行/列交替平衡(1 = 原行为) |
| `use_g_out` | false | 输出侧 C·W₀ 路径(约每层 6k 参数),色调/氛围向,建议消融 |
| `g_norm_mode="spectral"` | "frobenius" | G 路径按谱范数对齐,各层有效学习率更均匀 |

### 4.5 性能

- `apply_max_norm` 复用 forward 缓存的增量范数(滞后一步,软正则无碍),
  开 `scale_weight_norms` 时每步省一次完整 kron 重建 + W₀@A 大矩阵乘;
  未启用 max-norm 时零额外开销(缓存惰性开启)

### 4.6 代码结构

- B/A/C 三路径共用同一套参数构建、初始化、重建、bypass 逻辑(此前 B/A 两份复制粘贴)
- 阈值/缩放逻辑单一来源;净删约 100 行重复代码

**默认行为承诺**:不开启任何新开关时,前向数学与旧版逐位一致(数值修复除外);
旧版 glokr 存档可以直接被新代码加载。

---

## 5. 参数完整说明

`GLoKRModule.__init__` / `network_args` 可用参数(前 9 个位置参数顺序与 kohya 绑定,不可变):

| 参数 | 默认 | 说明 |
|------|------|------|
| `lora_dim` / `dim` | 4 | 低秩维度。决定 w1/w2 是否满秩:`dim ≥ max(因子)/2` 时该因子用完整矩阵 |
| `alpha` | 1 | 缩放分子;0 或 None 时取 dim(等效缩放 1) |
| `dropout` | 0.0 | 输入 dropout(仅 bypass 模式生效,LyCORIS 惯例) |
| `rank_dropout` | 0.0 | 行/输出元素 dropout。**画风训练建议 0** |
| `module_dropout` | 0.0 | 整模块跳过概率 |
| `use_tucker` | false | 占位(Linear-only 用不到),保持 kohya 位置兼容 |
| `use_scalar` | false | 全局可学习标量 0.01 起步(替代零初始化) |
| `decompose_both` | false | w1 也做低秩分解(默认只分解 w2) |
| `factor` | -1 | Kronecker 因子;-1 = 最均衡分解;可传 `(in_factor, out_factor)` |
| `rank_dropout_scale` | false | rank_dropout 后按均值重标定 |
| `weight_decompose` | false | 启用 DoRA |
| `use_bora` | false | 启用 BoRA(蕴含 weight_decompose) |
| `wd_on_out` | true | DoRA 范数按行(输出维)统计 |
| `full_matrix` | false | 全部用完整矩阵(不做低秩分解) |
| `bypass_mode` | None | 强制 bypass 前向;量化底模自动开启;与 wd 互斥 |
| `rs_lora` | false | 缩放分母用 √dim |
| `unbalanced_factorization` | false | 交换 out 侧因子(见 §10 限制) |
| `kron_rank` | 1 | Kron 项求和数;画风建议 2–4 |
| `use_g` | **true** | 输入侧 G 路径(W₀·A) |
| `use_g_out` | false | 输出侧 G 路径(C·W₀) |
| `train_gates` | false | 可学习门控 + kron 混合系数 |
| `init_mode` | "kaiming" | `"nkp"` = Van Loan 主成分对齐初始化 |
| `g_norm_mode` | "frobenius" | `"spectral"` = G 路径按谱范数对齐 |
| `bora_iters` | 1 | BoRA 交替平衡迭代次数(建议 2–3) |

**dim 选择速查(Flux,3072 维)**:注意力类层(out=3072)`dim=32` 即让 w2 满秩;
MLP 层(out=12288)需 `dim=64`。GLoKR 参数量远小于同 dim 的 LoRA,直接开 64 无压力。

---

## 6. 使用说明

### 6.1 基本训练(kohya sd-scripts)

```bash
--network_module=lycoris.kohya \
--network_dim=64 --network_alpha=64 \
--network_args "algo=glokr" "use_g=True"
```

### 6.2 画风训练推荐(一键 preset)

```bash
--network_module=lycoris.kohya \
--network_args "algo=glokr" "preset=<repo>/presets/flux_style_glokr.toml"
```

`presets/flux_style_glokr.toml` 内容概要:

- `single_blocks.*.linear*`:dim 64 + kron_rank 2(纹理/笔触主力)
- `double_blocks.*.{img,txt}_{attn,mlp}.*`:dim 32
- 排除 `*mod*` / `*modulation*`(调制层对全局风格影响过强,易带偏底模)
- 全线开启:`use_bora` + `bora_iters=2` + `train_gates` + `init_mode=nkp`,`rank_dropout=0`

手动等效(不用 preset):

```bash
--network_args "algo=glokr" "kron_rank=2" "use_bora=True" "bora_iters=2" \
               "train_gates=True" "init_mode=nkp" "rank_dropout=0"
```

### 6.3 训练器侧建议(杠杆比适配器更大)

- 时间步采样向低噪声段加权——纹理细节在去噪后段决定
- min-SNR / debiased 加权保持开启
- 数据集分辨率与裁剪质量优先于一切算法技巧

### 6.4 加载 / 合并 / 续训

```python
from lycoris.wrapper import create_lycoris_from_weights

net, _ = create_lycoris_from_weights(1.0, "style.safetensors", model)
net.apply_to()            # 挂载出图
net.merge_to(0.8)         # 或按 0.8 强度烘焙进底模(wd 解耦自动正确应用)
net.load_weights("ckpt.safetensors")   # 续训恢复
```

- `multiplier` 语义与 LoKr 家族一致:0 = 纯底模,插值对 DoRA/BoRA 同样生效
- 换底模挂载:G 路径强度锁定为训练时的 `g_norm`,不随新底模漂移

---

## 7. 存档格式

键名 = 参数属性名(`kron_rank>1` 时带 `_{i}` 后缀):

| 键 | 何时存在 | 说明 |
|----|----------|------|
| `b_w1[,_a,_b]` / `b_w2[,_a,_b]` | 恒有 | B 路 Kron 因子(w1 侧已折叠 `scalar`) |
| `a_w*` | `use_g` | A 路因子 |
| `c_w*` | `use_g_out` | C 路因子 |
| `alpha` | 恒有 | 缩放(rs_lora 已折算) |
| `dora_scale` / `bora_scale_r`+`bora_scale_c` | wd / BoRA | 解耦缩放 |
| `bora_iters` | BoRA | 平衡迭代数(旧档缺失 → 按 1) |
| `g_norm` | G 路径 | 训练时 ‖W₀‖(旧档缺失 → 现算回退) |
| `gate_b/a/c`、`kron_mix` | `train_gates` | 门控与混合系数 |

检测规则:存在 `{name}.b_w1*` 前缀键且无 `sora_cp/sora_dp` → GLoKR。
旧版(2026-07-23 前)glokr 存档完全兼容。

---

## 8. 测试与验证

测试位于 `tests/`,直接运行:

```bash
python tests/test_glokr.py   # 17 项单元/等价性测试
python tests/test_e2e.py     # 端到端 + preset 解析
```

覆盖(全部 PASS):

1. 合并前向 == 合并数学;bypass == 合并数学(含 kron_rank=2 + 门控/混合扰动、use_g_out)
2. wd 缩放扰动后 multiplier=0 严格还原底模(DoRA 与 BoRA×3 迭代)
3. wrapper 加载路径(kron_rank=1 / 2 + 门控 + BoRA),重建增量逐位一致
4. 续训 `load_state_dict` 覆盖加载
5. `merge_to` 非 wd 精确合并;wd 合并 == 训练前向
6. 全参数梯度回传(含门控/混合/BoRA/双 G 路)
7. bf16 吸收修复:wd 增量误差 < 旧算法 30%
8. `g_norm` 跨底模持久化 + 旧档回退
9. NKP 初始化:恒等起点 + 主成分方向对齐(|cos| > 0.9)
10. `apply_max_norm` 缓存路径 + 实际裁剪生效
11. Bokr/Lokr/gsokr 检测消歧;旧版 glokr 存档兼容

---

## 9. 性能特征

- **合并模式**(默认):每步重建 ΔW;A 路含一次 `W₀@A` 权重空间矩阵乘(O(out·in²))。
  当 `batch×seq > in_dim`(DiT 训练常态)时,这比在激活空间算更省——维持默认即可
- **bypass 模式**:分组算法 O(tokens·in·(in_m+in_n)),不重建大矩阵;
  适合小 batch、量化底模(自动强制)或显存紧张场景;`use_g_out` 在 bypass 下多一次全量 linear
- `scale_weight_norms` 开启时:范数缓存使每步正则开销从"完整重建"降为一次标量读取

---

## 10. 已知边界与注意事项

1. **`unbalanced_factorization` 存档的自动重建不支持**:`factorization` 返回排序因子,
   重建时形状对不上会报错(继承自 LoKr 同款模式)。训练可用,但请勿依赖从存档自动重建;
   如需使用请保留训练配置手动重建
2. **gsokr 键名重叠**:GloKrSora 与 GLoKR 共用 `b_w1` 命名,靠 `sora_cp/sora_dp` 区分;
   未启用 sora 的 gsokr 存档无法与 glokr 区分(gsokr 此前本就无检测能力,现状不劣化)
3. **bokr 未开 BoRA 的存档**按 LoKr 加载(键名与语义兼容,与修复前行为一致)
4. `lokr.py` 存在同款 bf16 往返吸收问题,属上游家规文件,本次未改动
5. 效果分级:§4.1–4.3 为机制保证(测试锁定);`train_gates`/`nkp` 有明确机制支撑、
   幅度依数据而定;`use_g_out` 为消融项——建议同数据集开/关对照一轮
