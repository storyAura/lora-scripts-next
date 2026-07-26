# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Next Trainer / SD Trainer Next** — a LoRA & finetune training WebUI for Windows, forked from
`Akegarasu/lora-scripts`. `python gui.py` starts a FastAPI backend (`mikazuki/`), serves a
**pre-compiled** frontend (`frontend/dist/`), and launches training subprocesses that run a
**locally modified** copy of kohya `sd-scripts`. Primary target is **Anima** (a DiT +
Rectified Flow model); SD 1.5 / SDXL / Flux are also supported.

## Commands

```bash
python gui.py                    # start everything (add --dev for dev mode)
run_gui.bat                      # Windows launcher (auto-installs deps on first run)
bash install.bash && bash run_gui.sh   # Linux

# Tests. Most are unittest; pytest is NOT installed in the venv by default.
venv\Scripts\python.exe -m unittest tests.test_anima_backend_adapter          # one module
venv\Scripts\python.exe -m unittest tests.test_tglokr.TGLoKRTests             # one class
venv\Scripts\python.exe -m unittest discover -s tests -t tests                # whole suite
# `discover -s tests -t .` fails — tests/ has no __init__.py, use `-t tests`.
# 4 of ~55 modules are pytest-style (test_china_hub, test_tokenizer_cache,
# test_portable_data_dir_links, test_portable_utils_flash_attn) and error out with
# "No module named 'pytest'" until you `pip install pytest`.

python scripts/sync_vendored_lycoris.py            # install vendor/lycoris over the pip one
python scripts/sync_vendored_lycoris.py --check    # report drift only (exit 1 if stale)
python scripts/bump_spa_asset_cache_key.py         # after editing frontend/dist
```

### Known-failing tests (pre-existing, not your change)

`test_anima_backend_upstream` (3, expects `vendor/sd-scripts` to be a git submodule),
`test_anima_fast_integration_static` (3, pins a stale cache key / VERSION 2.9.0 vs dist 2.8.35).
Verify against a clean checkout before assuming you broke something.

## Architecture

### Process layout (`gui.py`)

`gui.py` is a process orchestrator that also hosts uvicorn. Child services run as subprocesses:

| Service | Default port | Started at |
|---|---|---|
| Main WebUI (FastAPI) | 28000 | `gui.py` → `uvicorn.run("mikazuki.app:app")` (in-process) |
| Tag editor (Gradio) | 28001 | `run_tag_editor()`, submodule `mikazuki/dataset-tag-editor` |
| TensorBoard | 6006 | `python -m tensorboard.main` |
| Train monitor | 6008 | `train_monitor/server.py` (stdlib HTTP server) |

**Ports are not stable URLs.** `ensure_port_available()` reserves each service's default before
fallback scanning, then writes the results to `MIKAZUKI_PORT`, `MIKAZUKI_TENSORBOARD_PORT`,
`TRAIN_MONITOR_PORT`, `MIKAZUKI_TAGEDITOR_PORT`. Never hardcode `127.0.0.1:6008` — link to
`/train-monitor` (backend 302s) and `/proxy/tensorboard/`. See
`.cursor/rules/embedded-service-ports.mdc`.

### Config flow (the thing to understand first)

```
mikazuki/schema/*.ts          Schema DSL (schemastery), evaluated in the BROWSER via eval()
  → GET /api/schemas/all      api.py load_schemas(); /api/schemas/hashes drives hot reload
  → frontend builds the form, parseParams() flattens it
  → POST /api/run             api.py create_toml_file()
      fix_config_types → normalize_custom_args → validation → apply_*_defaults
      → config/autosave/<timestamp>.toml
  → process.py run_train() → build_accelerate_train_command()
      python mikazuki/accelerate_launch.py ... <trainer_file> --config_file <toml>
  → scripts/dev/anima_train_network.py   (thin wrapper: re-translates the TOML via
                                          adapt_anima_config → *-sd-scripts.toml)
  → vendor/sd-scripts/anima_train_network.py   (the real trainer)
```

The TOML is rewritten **twice** (GUI side, then wrapper side). `mixed_precision` is read back
out of the written TOML to feed accelerate, keeping launcher and trainer consistent.

Adding a UI parameter means touching both `mikazuki/schema/*.ts` (the form) **and** the backend
mapping — for LyCORIS algos that is `LYCORIS_NETWORK_ARG_MAP` in
`mikazuki/anima_backend/adapter.py`, whose reverse table lives in
`mikazuki/utils/config_import.py` (keep them symmetric; `tests/test_tglokr.py` asserts this).

### Editing schemas: three caches stand between you and the UI

1. `mikazuki/schema/*.ts` on disk.
2. **Backend memory** — `load_schemas()` reads the directory *once at startup*. Set
   `MIKAZUKI_SCHEMA_HOT_RELOAD=1` to re-read on every `/api/schemas/hashes` hit;
   `run_gui_source.bat`/`.ps1` already export it, `run_gui.bat` does not.
3. **Browser `localStorage["schemas"]`** — refreshed only when the served hash differs.

To tell layer 2 from layer 3 apart, compare the file's md5 against `GET /api/schemas/hashes`:
equal means the backend is current and the stale copy is in the browser (hard-reload).

Each schema is a `Schema.union([...])` discriminated by `lora_type`, every branch being a
`Schema.object` keyed `lora_type: Schema.const("<name>").required()`. **Do not use
`Schema.const()` for any other field in a branch.** The form model carries raw, unvalidated
values across branch switches, autosave and history restore — a leftover value that conflicts
with a branch const makes the union match nothing: the branch section vanishes, the TOML
preview goes blank, and submit throws *before* any request is sent (a misleading "network
error"). The implemented pattern: branch-stamped fields (`network_module`, `lycoris_algo`) are
tolerant `Schema.string().default(...)`, and `ANIMA_LORA_TYPE_BRANCH_CONSTS` in
`mikazuki/utils/config_import.py` is the single source of truth — the import path stamps the
right values on restore, `_apply_lora_type_overrides()` in the adapter forces them at train
time regardless of what the form carried. Changing a branch's module/algo means updating that
map in the same commit.

Two rendering rules: a union only gets a dropdown when it has **more than one** visible choice
(a single-option union renders no control at all), and a safe union that may legitimately not
match needs a trailing `Schema.object({})` fallback branch (see the optimizer unions).

### Saving / loading params ("保存参数 / 读取参数") and config import

Saving pushes a **raw form snapshot** into `localStorage["configs-<type>"]`; the autosave
(`configs-<type>-autosave`) is restored into the form verbatim on page load, no backend
involved. Applying a history entry and importing a TOML both POST to
`/api/config/validate-import` (`mikazuki/utils/config_import.py`: type detection, redirect
between pages, network_args → UI field hydration), after which the frontend merges *page
schema defaults + returned config*. Those defaults resolve every union to its **first**
branch, so any branch-dependent key the backend does not stamp explicitly silently falls back
to branch-1 values — that is why `validate-import` derives `network_module`/`lycoris_algo`
from `lora_type` instead of trusting the snapshot.

### Verifying a training change

Submit through the real path — `POST /api/run` with the flat config as JSON — instead of
invoking a trainer script by hand. `create_toml_file()` also writes the sample-prompts file
(`get_sample_prompts`) and applies per-type defaults; bypassing it produces failures that do not
exist in the product. Watch progress over `GET /api/train/log/stream/{task_id}` and stop with
`GET /api/tasks/terminate/{task_id}`.

### Training type routing

`trainer_mapping` in `mikazuki/app/api.py` maps `model_train_type` → trainer script.
`anima-lora-fast` is **not** in that table — `create_toml_file()` branches to the plugin backend
before reaching it. Note `sd3-lora` and `anima-lora` both point at the Anima trainer: the "SD3"
page *is* the Anima page (historical naming, see `frontend/VENDOR.md`).

### Standard vs Anima Fast backend

- `mikazuki/anima_backend/` — the standard path: `adapt_anima_config()` rewrites GUI keys into
  sd-scripts keys, `upstream.py` verifies the pinned `vendor/sd-scripts` commit
  (`ANIMA_ALLOW_COMMIT_DRIFT=1` downgrades to a warning), `lycoris_patch.py` applies runtime patches.
- `mikazuki/anima_fast_backend/` — an optional external trainer installed into
  `extensions/anima_lora/` with its **own venv**; launches `train.py` directly, bypassing
  accelerate. Kill switch: `LORA_ENABLE_ANIMA_FAST=0`.

### Vendored trees (do not confuse them)

| Path | What | Editable? |
|---|---|---|
| `vendor/sd-scripts/` | Modified kohya sd-scripts — the real Anima trainers | yes, this is where trainer fixes go |
| `vendor/lycoris/` | Modified LyCORIS (local algos: glokr / tglokr / bokr / bora / gsokr / glora_boft) | yes — then run the sync script |
| `scripts/stable/`, `scripts/dev/` | Vendored kohya stable/dev branches | no, except the two `anima_train*.py` wrappers |
| `frontend/dist/` | Pre-compiled frontend, built elsewhere | patch the built artifacts directly |

**`vendor/lycoris` gotcha:** `pip install lycoris-lora` overwrites it with upstream, which has
none of the local algos. Run `scripts/sync_vendored_lycoris.py` after any venv rebuild;
`tests/test_vendored_lycoris.py` fails if the installed copy drifts from the vendored one.
Never edit only the venv copy. Details in `vendor/lycoris/VENDOR.md`.

### Frontend: patch the build output

There is no frontend build in this repo. UI changes are string patches against
`frontend/dist/` (see `scripts/patch-*.py` for the established pattern). VuePress is SSR +
hydration, so **the same text usually has to be patched in both the JS chunk and the HTML**,
or the first paint flashes the old string.

After editing any file under `frontend/dist/`, bump the shared cache key: edit
`SPA_ASSET_CACHE_KEY` in `scripts/spa_asset_cache.py`, append the old value to
`LEGACY_SPA_ASSET_CACHE_KEYS`, then run `scripts/bump_spa_asset_cache_key.py`.
Partial bumps are worse than none — a stale `?v=` on one chunk makes the browser load two
copies of `app.js`, which breaks the whole SPA. `tests/test_frontend_dist_cache.py` enforces this.

### Runtime plumbing worth knowing

- **Tasks**: `mikazuki/tasks.py`, singleton `tm`, `max_concurrent=1` — one training at a time.
  In-memory only (lost on restart). `GET /api/tasks`, `GET /api/tasks/terminate/{id}`.
- **Log streaming**: `mikazuki/train_log_hub.py` keeps a ring buffer per task;
  `GET /api/train/log/stream/{task_id}` is SSE. The main API does **not** log every request,
  so an absent console line does not mean the request never arrived.
- **Subprocess env**: `process.py` sets `PYTORCH_CUDA_ALLOC_CONF` per platform
  (`expandable_segments` is unsupported on Windows), injects `PYTHONPATH`, disables color.
- **China mirror**: `mikazuki/china_hub.py` reroutes HF downloads to ModelScope; called both in
  `gui.py` and in every training subprocess.

## Repo conventions

- `docs/` (plural, tracked) = public docs. `doc/` (singular, **gitignored**) = local agent
  handover notes — `doc/local/AGENT_INTERNAL.md` is the entry point, indexed by
  `.cursor/rules/local-docs-index.mdc`. Same split for `scripts/` (tracked) vs `script/` (ignored).
  Never commit local-only material into the tracked directories.
- Contract paths that must not be renamed: `gui.py`, `run_gui.bat`, `start_autodl.sh`,
  `setup_environment.py`, `requirements.txt`, `VERSION`, and the portable layout under
  `scripts/portable/`. Full list in `docs/repo-layout.md`.
- Conventional-commit messages; Chinese subject lines are the norm in this fork.

## Anima-specific traps

- All Anima DiT `LayerNorm`s are `elementwise_affine=False` (`weight is None`), so LyCORIS
  `train_norm` has nothing to train there.
- Blocks run the AdaLN modulation inside `torch.autocast(..., enabled=use_fp32)`, which
  **disables** autocast on the bf16 path — tensors reaching it must already match the weight
  dtype. Changing timestep dtypes upstream has bitten this before.
- GLoKR/T-GLoKR default to merged mode, which reconstructs the full ΔW per module per step —
  heavy on VRAM. `bypass_mode=True` avoids it but is mutually exclusive with
  `use_bora`/`dora_wd`. `vendor/lycoris/GLOKR.md` has the full parameter reference.
- For LoKr-family algos `network_dim` is only a threshold ("stop decomposing the Kronecker
  factors"), not a capacity dial — huge values are idiomatic. Capacity comes from `factor`
  (`-1` = balanced = *fewest* parameters; smaller values inflate them quadratically) and, for
  GLoKR, from `kron_rank` (linear).
- Preview sampling: `sample_sampler` is decorative — the only implementation is the built-in
  rectified-flow Euler in `anima_train_utils.do_sample()`. `sample_scheduler` is real
  (simple/beta). Per-image sample params travel as prompt-line flags
  (`--w --h --s --l --d --ss --sch --fs`) baked by `get_sample_prompts()`, never as TOML keys
  (the adapter drops all `sample_*` UI fields). `--fs <flow_shift>` (default 3.0) is parsed by
  the trainer but not exposed in the UI.
- Timestep-aware adapters (T-LoRA rank masking, T-GLoKR time gates) read
  `network.set_current_timestep()`. The train loop injects per-batch timesteps in [0, 1000]
  and must **not** clear them before backward (gradient checkpointing re-runs the forward);
  preview sampling injects the per-step sigma in [0, 1] via `do_sample(timestep_callback=...)`.
  Consumers divide values > 1 by 1000, so both scales agree.
- On Windows an over-committed allocation does not raise: it silently spills into shared system
  memory and training keeps "running" at a crawl. GPU **power draw** tells the two apart —
  near the limit means real compute, roughly half means it is stalled on memory transfers.
