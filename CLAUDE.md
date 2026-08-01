# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**lora-scripts-story-next** (Next Story Trainer; upstream brand: Next Trainer / SD Trainer Next) — a LoRA & finetune training WebUI for Windows, forked from
`Akegarasu/lora-scripts`. `python gui.py` starts a FastAPI backend (`mikazuki/`), serves a
**pre-compiled** frontend (`frontend/dist/`), and launches training subprocesses that run a
**locally modified** copy of kohya `sd-scripts`. Primary target is **Anima** (a DiT +
Rectified Flow model); SD 1.5 / SDXL / Flux are also supported.

## Commands

```bash
python gui.py                    # start everything (add --dev for dev mode)
run_gui.bat                      # Windows launcher (auto-installs deps on first run)
bash install.bash && bash run_gui.sh   # Linux

# Tests: TWO gates, run both before calling a change green. pytest is installed
# in the venv (since 2026-07-28); unittest discover silently SKIPS bare-function
# pytest-style modules (test_china_hub, test_tagger_progress_api,
# test_base_model_quantization, ...), so a green unittest run alone proves nothing
# about them.
venv\Scripts\python.exe -m unittest tests.test_anima_backend_adapter          # one module
venv\Scripts\python.exe -m unittest tests.test_timestep_adapters.TrainerTimestepLifecycleTests  # one class
venv\Scripts\python.exe -m unittest discover -s tests -t tests                # whole suite
venv\Scripts\python.exe -m pytest -q tests\                                   # incl. pytest-style
# `discover -s tests -t .` fails — tests/ has no __init__.py, use `-t tests`.
# Never leave sys.modules stubs installed at test-module import time — one module's
# leftover stub breaks later modules that import the real package under pytest.

python scripts/sync_vendored_lycoris.py            # vendor/lycoris → venv (gui.py also self-heals this at boot)
python scripts/sync_vendored_lycoris.py --check    # report drift only (exit 1 if stale)
python scripts/bump_spa_asset_cache_key.py         # after editing frontend/dist
```

### Known-failing tests (pre-existing, not your change)

unittest baseline as of 2026-07-31 (462 tests): **4 failures + 1 skipped** —
`test_anima_backend_upstream` (3, expects `vendor/sd-scripts` to be a git submodule) and
`test_anima_fast_integration_static` (1, dist still pins `sd-trainer-brand.js?v=2.8.35`;
cosmetic — that file is served no-cache).
pytest baseline (`-m pytest -q tests\`, ~668 passed): the same 4 plus pytest-only
pre-existing reds — `test_china_hub` (2: one asserts modelscope-missing behavior, one
downloads from ModelScope), `test_cli_entrypoints` (1, README does not document
`train_anima_by_toml.sh`), `test_dataset_editor_api` (1, dist tageditor shell lacks the
经典标签编辑 embed) → **8 failures + 1 skipped**. `test_portable_updater_paths` hits the
live GitHub release API and flips with network health — rerun it in isolation before
blaming a change. Anything beyond these counts is your change.

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
      fix_config_types → normalize_custom_args → normalize_optimizer_configuration
      → validate_training_configuration → apply_anima_training_defaults (attn probe)
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
mapping in `mikazuki/anima_backend/adapter.py`. Two channels: standalone `networks.*` modules
register their fields in `ANIMA_NETWORK_MODULE_ARG_FIELDS` (the default for anything new — see
the placement rule below); the frozen LyCORIS algos use `LYCORIS_NETWORK_ARG_MAP`. Each has a
reverse/hydration table in `mikazuki/utils/config_import.py` (`_ANIMA_NETWORK_ARG_TO_UI` /
`_LYCORIS_NETWORK_ARG_TO_UI` plus the bool-coercion sets) — keep forward and reverse in sync;
`tests/test_timestep_adapters.py` asserts the LyCORIS pair.

### Pre-launch guards (run before any file/process work; audit 2026-07-28)

- `mikazuki/training_validation.py` — pure-Mapping validator (no torch import), called from
  both `create_toml_file()` and `adapt_anima_config()`. Removed algorithms go into
  `UNIMPLEMENTED_ANIMA_ADAPTER_TYPES` so a stale saved config fails loudly instead of silently
  training something else (precedents: tglokr 2026-07-28, glokr 2026-07-29).
- `mikazuki/attention_probe.py` — real forward/backward probes, cached per environment
  fingerprint. Auto-detect picks the first backend that passes; an explicitly requested
  xformers/flash whose probe fails raises `AttentionBackendUnavailableError`, which
  `_apply_anima_training_defaults_or_fail` converts to a structured `APIResponseFail` — every
  `apply_anima_training_defaults` call site in request handlers must go through that wrapper.
- `mikazuki/optimizer_configuration.py` — optimizer alias normalization (short names →
  full bitsandbytes class paths, structured AdEMAMix fields); runs before validation.
- Frozen-base quantization is a **two-layer white-list changed together**:
  lora_type {lora, lora_plus, lora_fa, vera} in `training_validation.py` ↔ the matching
  native modules in `SUPPORTED_QUANTIZED_NETWORK_MODULES`
  (`vendor/sd-scripts/library/base_model_quantization.py`). rsLoRA runs on lycoris.kohya and
  is rejected by design.

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

Three rendering rules: a union only gets a dropdown when it has **more than one** visible choice
(a single-option union renders no control at all); a safe union that may legitimately not
match needs a trailing `Schema.object({})` fallback branch (see the optimizer unions); and
number fields validate "(value − min) is an exact multiple of step" browser-side — a `min` off
the step grid makes the field's own default invalid and the whole branch silently vanishes
(`tests/test_schema_field_constraints.py` sweeps every schema for this and for required-const
branch markers).

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
  (`ANIMA_ALLOW_COMMIT_DRIFT=1` downgrades to a warning) and refuses lycoris.kohya training on
  an unsynced venv (`verify_vendored_lycoris`, `ANIMA_ALLOW_LYCORIS_DRIFT=1` bypasses).
  Never re-add runtime monkeypatches of LyCORIS forwards: the removed `lycoris_patch.py` used
  to clobber the fixed LoKr forward with a bf16-absorbing copy on every launch
  (`tests/test_anima_lycoris_guard.py` guards against a comeback).
- `mikazuki/anima_fast_backend/` — an optional external trainer installed into
  `extensions/anima_lora/` with its **own venv**; launches `train.py` directly, bypassing
  accelerate. Kill switch: `LORA_ENABLE_ANIMA_FAST=0`.

### Vendored trees (do not confuse them)

| Path | What | Editable? |
|---|---|---|
| `vendor/sd-scripts/` | Modified kohya sd-scripts — the real Anima trainers | yes, this is where trainer fixes go |
| `vendor/lycoris/` | Modified LyCORIS — **frozen 2026-07-29** (LoKr fp32-safe forward + bokr / bora / gsokr / glora_boft; the glokr & cdka copies are legacy-only) | numerical fixes to existing algos only, never new ones — then run the sync script |
| `scripts/stable/`, `scripts/dev/` | Vendored kohya stable/dev branches | no, except the two `anima_train*.py` wrappers |
| `frontend/dist/` | Pre-compiled frontend, built elsewhere | patch the built artifacts directly |

**`vendor/lycoris` gotcha:** `pip install lycoris-lora` overwrites it with upstream, which has
none of the local algos. `gui.py` auto-repairs the venv copy at startup and `install.bash`
runs the sync after installing requirements; for manual venv work run
`scripts/sync_vendored_lycoris.py` yourself. `tests/test_vendored_lycoris.py` fails if the
installed copy drifts from the vendored one, and the trainer refuses lycoris.kohya runs on an
unsynced venv. Never edit only the venv copy. Details in `vendor/lycoris/VENDOR.md`.

**LyCORIS 目标层圈选 (2026-07-31 定案):** the adapter auto-attaches
`config/lycoris_anima_preset.toml` to every `lycoris.kohya` run. Its target surface
deliberately mirrors `lora_anima.py`'s exclusion regex (`_modulation|_norm|_embedder|
final_layer`): only attn+mlp linears inside Block / LLMAdapterTransformerBlock are trained.
Never widen it back — training the `adaln_modulation` layers (per-block multiplicative tone
gates) progressively wrecks previews and products into saturation/contrast collapse; this was
the root cause of the 07-31 "LoKr 全灭而 LoRA 健康" incident, misattributed for weeks to the
bf16 forward and then to full_matrix. Preset mechanics: `match_fn` is regex by default, so
glob-style `exclude_name` patterns need `use_fnmatch = true`; and exclusion inside class-swept
children only works because the vendored `wrapper.py` threads `target_exclude_names` through
`create_modules_` (upstream ignores it there). `tests/test_lycoris_preset_exclusion.py`
guards the mechanism.

**Algorithm placement rule (2026-07-29, user decree):** new algorithms — paper-based or
experimental — are standalone `vendor/sd-scripts/networks/*_anima.py` modules (clone the
`moslora_anima.py` scaffolding: `_network_factory`/`_module_class` hooks over `lora_anima`,
branch fields forwarded via `ANIMA_NETWORK_MODULE_ARG_FIELDS` in the adapter).
`vendor/lycoris` is **frozen**: no new algorithms go in, ever; it only keeps what it already
ships. Precedent: CDKA moved to `networks/cdka_anima.py` (same archive keys; the lycoris copy
stays untouched for legacy `algo=cdka` configs, `networks.cdka_anima` is canonical).

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
  In-memory only (lost on restart); only the most recent 16 finished tasks are retained, older
  ones are pruned together with their log buffers (long sessions stay memory-bounded).
  `GET /api/tasks`, `GET /api/tasks/terminate/{id}`.
- **训练队列**: `mikazuki/train_queue.py` (singleton `train_queue`) + `mikazuki/app/queue_api.py`
  (`/api/queue*`). The asyncio runner starts at app lifespan and **arms** the `/api/run`
  intercept — unarmed (tests/scripts calling `create_toml_file` directly) keeps stock behavior.
  Armed rules: **every submit enqueues** (queue-first is the default; idle submits auto-start
  the conveyor unless `user_paused`); an entry in `editing` → the submit *saves into that
  entry* and the conveyor halts (`user_paused=True`) until manually restarted. done/failed
  entries stay as history (finish time + `duration_seconds` in the snapshot) until deleted or
  「归档清除」. Note: armed `/api/run` responses carry `queue_message`, **not** `task_id`
  (launch is async via the runner). Entries launch via
  `submit_training_config()` (the extracted `/api/run` pipeline), so validation/OOM failures
  land on the entry (`failed` + error) and the queue continues. State persists in
  `config/train_queue.json` (gitignored); on restart `active=False`, running→failed. Frontend
  is `frontend/dist/assets/sd-trainer-queue.js` (no-cache, dynamically loaded by
  sd-trainer-brand.js — no HTML references, so **no SPA cache-key bump needed** for either).
  Editing loads the entry config through `localStorage["configs-<pageTrainType>-autosave"]` +
  hard navigation (the form only reads that key at mount). layout.96d49288.js's success toast
  was string-patched to prefer `data.queue_message`
  (`tests/test_train_submit_loading_static.py` guards the exact string;
  `tests/test_train_queue.py` covers the conveyor).
- **Log streaming**: `mikazuki/train_log_hub.py` keeps a 15000-line ring buffer per task;
  `GET /api/train/log/stream/{task_id}` is SSE. Snapshot cursors are **absolute** appended-line
  counts so streaming survives ring wrap-around — keep that invariant
  (`tests/test_train_log_hub.py` guards it). The main API does **not** log every request,
  so an absent console line does not mean the request never arrived.
- **Subprocess env**: `process.py` sets `PYTORCH_CUDA_ALLOC_CONF` per platform
  (`expandable_segments` is unsupported on Windows), injects `PYTHONPATH`, disables color.
- **China mirror**: `mikazuki/china_hub.py` reroutes HF downloads to ModelScope; called both in
  `gui.py` and in every training subprocess.

## Repo conventions

- Remote: `origin` = `storyAura/lora-scripts-story-next` (this fork; commit to `main` and push
  directly, no PRs). Links to `wochenlong/lora-scripts-next` — issues, releases, `GITHUB_REPO`
  in `mikazuki/update_check.py`, portable updater URLs — point at the **upstream** on purpose:
  that is where the referenced issues and release packages live. Do not "fix" them.
- Cutting a version touches four places in one commit: `VERSION` (feeds the sidebar chip via
  `/api/version`), `CHANGELOG.md`, the changelog tables in **both** `README.md` and
  `README-zh.md` (bilingual pair — always edit both), and the WebUI 「其他 → 更新日志」 page.
  That page is SSR + hydration: insert *identical* HTML into
  `frontend/dist/other/changelog.html` **and** the template string in
  `frontend/dist/assets/changelog.html.e5f6a7b8.js`, then bump the SPA cache key. Never rerun
  `scripts/patch-home-changelog.py` — it regenerates the page from content baked in at v2.8.2
  and clobbers every newer entry.
- `docs/` (plural, tracked) = public docs. `doc/` (singular, **gitignored**) = local agent
  handover notes — `doc/local/AGENT_INTERNAL.md` is the entry point, indexed by
  `.cursor/rules/local-docs-index.mdc`. Same split for `scripts/` (tracked) vs `script/` (ignored).
  Never commit local-only material into the tracked directories.
- Contract paths that must not be renamed: `gui.py`, `run_gui.bat`, `start_autodl.sh`,
  `setup_environment.py`, `requirements.txt`, `VERSION`, and the portable layout under
  `scripts/portable/` (`launch_portable.bat`, bundle roots `python_embeded/`/`SD-Trainer/`;
  `build-scripts/` builds that bundle). Full list in `docs/repo-layout.md`.
- Conventional-commit messages; Chinese subject lines are the norm in this fork.

## Anima-specific traps

- All Anima DiT `LayerNorm`s are `elementwise_affine=False` (`weight is None`), so LyCORIS
  `train_norm` has nothing to train there.
- Blocks run the AdaLN modulation inside `torch.autocast(..., enabled=use_fp32)`, which
  **disables** autocast on the bf16 path — tensors reaching it must already match the weight
  dtype. Changing timestep dtypes upstream has bitten this before.
- GLoKR was removed from the GUI on 2026-07-29 (`lora_type=glokr` is rejected pre-launch;
  the vendored module stays for legacy archives and custom network_args). Where it still
  runs, its trap remains: merged mode reconstructs the full ΔW per module per step — heavy
  on VRAM; `bypass_mode=True` avoids it but is mutually exclusive with `use_bora`/`dora_wd`.
  `vendor/lycoris/GLOKR.md` has the full parameter reference.
- For LoKr-family algos `network_dim` is only a threshold ("stop decomposing the Kronecker
  factors"), not a capacity dial — huge values are idiomatic. Capacity comes from `factor`
  (`-1` = balanced = *fewest* parameters; smaller values inflate them quadratically). CDKA
  ignores `network_dim`/`network_alpha` entirely — capacity is `cdka_r1/r2/r`.
- LoKr `full_matrix=true` is a **supported** mode, not a forbidden one: with it (or with a
  huge `network_dim`, which is equivalent — dim is a threshold) `dim`/`alpha` are ignored
  (LyCORIS forces scale=1) and `adapt_anima_config` auto-fills `scale_weight_norms=1.0` when
  the field is left empty (explicit values, including `0` = off, are respected). Do not
  re-introduce the old warn-only behavior — the schema copy promises this auto-guardrail.
- Preview sampling: `sample_sampler` is decorative — the only implementation is the built-in
  rectified-flow Euler in `anima_train_utils.do_sample()`. `sample_scheduler` is real
  (simple/beta). Per-image sample params travel as prompt-line flags
  (`--w --h --s --l --d --ss --sch --fs`) baked by `get_sample_prompts()`, never as TOML keys
  (the adapter drops all `sample_*` UI fields). `--fs <flow_shift>` (default 3.0) is parsed by
  the trainer but not exposed in the UI.
- Timestep-aware adapters (T-LoRA rank masking) read
  `network.set_current_timestep()`. The train loop injects per-batch timesteps in [0, 1000]
  and must **not** clear them before backward (gradient checkpointing re-runs the forward);
  preview sampling injects the per-step sigma in [0, 1] via `do_sample(timestep_callback=...)`.
  Consumers divide values > 1 by 1000, so both scales agree.
- On Windows an over-committed allocation does not raise: it silently spills into shared system
  memory and training keeps "running" at a crawl. GPU **power draw** tells the two apart —
  near the limit means real compute, roughly half means it is stalled on memory transfers.
