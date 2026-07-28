## Open Source Notices

### Upstream GUI / community packaging

This repository is maintained as **[storyAura/lora-scripts-story-next](https://github.com/storyAura/lora-scripts-story-next)**, forked from **[wochenlong/lora-scripts-next](https://github.com/wochenlong/lora-scripts-next)**, and traces its UX and packaging to **Akegarasu SD-Trainer** / **秋叶一键训练包**: **[Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts)** (training backend integration: **[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)**).

### Rectified Flow (SDXL LoRA)

This project includes Rectified Flow training support for SDXL LoRA inspired by and adapted from:

- `bluvoll/Akegarasu-lora-scripts-RF`: https://github.com/bluvoll/Akegarasu-lora-scripts-RF

The referenced repository is licensed under AGPL-3.0. Its RF training ideas include the Rectified Flow objective, sigma timestep sampling, optional cosine optimal transport pairing, and SDXL resolution-dependent timestep shift controls.

### Anima LoRA

Active Anima backend maintenance is based on:

- `kohya-ss/sd-scripts`: https://github.com/kohya-ss/sd-scripts

Earlier Anima integration work also referenced:

- `WhitecrowAurora/lora-rescripts` (**SD-reScripts** — historical fork / continuation of the LoRA-scripts line): https://github.com/WhitecrowAurora/lora-rescripts

The historical reference repository is licensed under AGPL-3.0. Current Anima training should be synchronized from `kohya-ss/sd-scripts`; local code is limited to the WebUI compatibility wrapper, config adapter, defaults, and launch orchestration.

### Anima LoRA Fast Mode（进阶插件）

The optional **Fast mode** (`model_train_type: anima-lora-fast`) integrates the optimized Anima LoRA training engine adapted from:

- `sorryhyun/anima_lora`: https://github.com/sorryhyun/anima_lora

Licensed under the **MIT License** (Copyright (c) 2026 Seunghyun Ji).

SD Trainer ships this engine as an **optional plugin** (`extensions/anima_lora/`). On install, the plugin snapshot includes upstream `LICENSE` / `NOTICE` / `README.md` (see `mikazuki/anima_fast_backend/installer.py`). User-facing docs: [`docs/anima-fast.md`](docs/anima-fast.md).

The referenced repository features per-block or full-model `torch.compile`, static-shape token bucketing, and compile-friendly forward paths. On consumer GPUs this yields substantially faster step times than the default Kohya/sd-scripts path at the cost of higher VRAM and a separate Python/CUDA runtime. See **Performance** in `docs/anima-fast.md` for measured comparisons in this repo.

### LyCORIS

LoKr, LoHa, and other advanced LoRA variants are powered by:

- `KohakuBlueleaf/LyCORIS`: https://github.com/KohakuBlueleaf/LyCORIS

LyCORIS (Lora beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion) is licensed under the **Apache License 2.0**. It provides the network modules for LoKr (Low-Rank Kronecker Product), LoHa (Low-Rank Hadamard Product), and other parameter-efficient fine-tuning methods used in this project.

**Paper**: Yeh et al. *Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning to Model Evaluation*. ICLR 2024.

### T-LoRA (Timestep-Dependent LoRA)

T-LoRA support is adapted from:

- `ControlGenAI/T-LoRA`: https://github.com/ControlGenAI/T-LoRA

The referenced repository is licensed under the **MIT License** (Copyright (c) 2025 AIRI, https://airi.net).

T-LoRA introduces dynamic rank adjustment based on diffusion timesteps and orthogonal initialization. Local files `networks/tlora.py` and `networks/tlora_anima.py` are adapted from the original implementation with modifications for Anima model integration within the sd-scripts training pipeline.

**Paper**: Nikita Balagansky, Daniil Gavrilov. *T-LoRA: Timestep-Dependent Low-Rank Adaptation for Diffusion Models*. 2025.

### DeLoRA and WaveFT

The local Anima implementations of DeLoRA and WaveFT follow the public
algorithm definitions and checkpoint structure documented by:

- `huggingface/peft`: https://github.com/huggingface/peft

PEFT is licensed under the **Apache License 2.0**. The local integrations are
limited to Linear layers and add sd-scripts network discovery, validation,
metadata, merge, and Anima training support.

**Papers**:

- Ling et al. *DeLoRA: Fine-Tuning-Free Low-Rank Adaptation for Subject-Driven
  Image Generation*. 2025.
- Guo et al. *WaveFT: Wavelet-domain Adaptation for Low-Rank Fine-Tuning*. 2025.

### DEFT

The local Anima DEFT implementation follows the public algorithm definition
and was cross-checked against:

- `MAXNORM8650/DEFT`: https://github.com/MAXNORM8650/DEFT
- `huggingface/peft` DEFT tuner:
  https://github.com/huggingface/peft/tree/main/src/peft/tuners/deft

Both referenced codebases are licensed under the **Apache License 2.0**. The
local integration supports the QR and ReLU projection variants for Linear
layers, strict configuration validation, identity-preserving initialization,
checkpoint metadata, and merge.

### MoSLoRA

The local MoSLoRA implementation is a clean-room implementation of the
published `B @ M @ A` equation from:

- Wu et al. *MoSLoRA: Extremely Parameter-Efficient Fine-Tuning via
  Mixture-of-Subspaces Low-Rank Adaptation*. 2024.
- Paper: https://arxiv.org/abs/2406.11909
- Author repository: https://github.com/wutaiqiang/MoSLoRA

The author repository did not contain a root license file when reviewed on
2026-07-28. No source code from that repository was copied. The local
implementation uses only the published mathematical description and supports
Linear layers, deterministic checkpoint identity, merge, and standard-LoRA
export by folding the mixer into the up projection.

### EmoSens Optimizer

The EmoSens optimizer is adapted from:

- `muooon/EmoSens`: https://github.com/muooon/EmoSens

Licensed under the **Apache License 2.0** (Copyright (c) muooon).

EmoSens is an emotion-driven optimizer that generates autonomous learning rates via the emoPulse mechanism, analyzing loss fluctuations through multi-scale EMA. The implementation is in `vendor/sd-scripts/library/optimizers/emosens.py`.

**Citation**: muooon. "emo series Optimizers: An emotion-driven optimizer that feels loss and navigates accordingly." DOI: 10.57967/hf/7738. https://github.com/muooon/EmoSens
