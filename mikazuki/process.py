
import asyncio
import importlib.util
import os
import site
import sys
import webbrowser
import uuid
from pathlib import Path
from typing import Any, Optional

_VALID_ACCELERATE_MIXED_PRECISION = frozenset({"no", "fp16", "bf16"})

from mikazuki.app.models import APIResponse
from mikazuki.anima_fast_backend.launcher import build_launch_spec
from mikazuki.anima_fast_backend.service_resolver import default_resolver
from mikazuki.log import log
from mikazuki.tasks import tm
from mikazuki.launch_utils import base_dir_path
from mikazuki.portable_utils import train_env_overrides


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _loads_train_toml(text: str) -> Optional[dict]:
    for loader_name in ("toml", "tomllib"):
        try:
            loader = __import__(loader_name)
        except ModuleNotFoundError:
            continue
        try:
            data = loader.loads(text)
        except (ValueError, TypeError):
            return None
        return data if isinstance(data, dict) else None
    return None


def normalize_mixed_precision(value: Any) -> Optional[str]:
    """Return accelerate-compatible mixed_precision or None when unset/invalid."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized or normalized in {"none", "null"}:
        return None
    if normalized in _VALID_ACCELERATE_MIXED_PRECISION:
        return normalized
    return None


def read_mixed_precision_from_train_toml(toml_path: str) -> Optional[str]:
    path = Path(toml_path)
    if not path.is_file():
        return None
    try:
        data = _loads_train_toml(path.read_text(encoding="utf-8"))
    except OSError:
        return None
    if not data:
        return None
    return normalize_mixed_precision(data.get("mixed_precision"))


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _module_origin_under_user_site(module_name: str) -> bool:
    try:
        user_site = Path(site.getusersitepackages())
    except (AttributeError, TypeError):
        return False

    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False

    candidates: list[str] = []
    if spec.origin:
        candidates.append(spec.origin)
    if spec.submodule_search_locations:
        candidates.extend(str(location) for location in spec.submodule_search_locations)

    return any(_path_is_relative_to(Path(candidate), user_site) for candidate in candidates)


def _should_disable_user_site() -> bool:
    """Keep portable isolation unless training deps are installed in user site."""
    return not any(_module_origin_under_user_site(name) for name in ("torch", "accelerate"))


def build_accelerate_train_command(
    *,
    trainer_file: str,
    toml_path: str,
    cpu_threads: int = 2,
    gpu_ids: Optional[list] = None,
) -> tuple[list[str], dict[str, str], Optional[str]]:
    """Build accelerate launch argv and env for sd-scripts training."""
    launch_opts = [
        "--num_cpu_threads_per_process",
        str(cpu_threads),
        "--quiet",
    ]
    mixed_precision = read_mixed_precision_from_train_toml(toml_path)
    if mixed_precision:
        launch_opts.extend(["--mixed_precision", mixed_precision])

    launch_entry = Path(__file__).resolve().parent / "accelerate_launch.py"
    args = [
        sys.executable,
        str(launch_entry),
        *launch_opts,
        trainer_file,
        "--config_file",
        toml_path,
    ]

    customize_env = os.environ.copy()
    customize_env.update(train_env_overrides())
    # The training subprocess runs ``python mikazuki/accelerate_launch.py``,
    # whose entry imports ``mikazuki.china_hub``. Launching by script path puts
    # only the script's own directory (``mikazuki/``) on sys.path, not the
    # project root, so portable installs (no editable/site-packages mikazuki)
    # fail with ``ModuleNotFoundError: No module named 'mikazuki'`` (issue #158).
    # Inject the project root onto PYTHONPATH so the package is importable.
    project_root = str(base_dir_path())
    existing_pythonpath = customize_env.get("PYTHONPATH", "")
    path_parts = [p for p in existing_pythonpath.split(os.pathsep) if p]
    if project_root not in path_parts:
        path_parts.insert(0, project_root)
    customize_env["PYTHONPATH"] = os.pathsep.join(path_parts)
    # CUDA allocator: mitigate VRAM fragmentation OOM (bucketed resolutions and
    # block swap produce variable-size allocations). expandable_segments is not
    # supported on Windows (PyTorch warns and ignores it), so use GC threshold +
    # split cap there. A user-set PYTORCH_CUDA_ALLOC_CONF always wins.
    if "PYTORCH_CUDA_ALLOC_CONF" not in customize_env:
        if sys.platform == "win32":
            customize_env["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512"
        else:
            customize_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    customize_env["ACCELERATE_DISABLE_RICH"] = "1"
    customize_env["PYTHONUNBUFFERED"] = "1"
    customize_env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"
    if _should_disable_user_site():
        customize_env["PYTHONNOUSERSITE"] = "1"
    else:
        customize_env.pop("PYTHONNOUSERSITE", None)
    customize_env["NO_COLOR"] = "1"
    customize_env["FORCE_COLOR"] = "0"
    customize_env["TERM"] = "dumb"

    if gpu_ids:
        customize_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        if len(gpu_ids) > 1:
            multi_gpu_args = ["--multi_gpu", "--num_processes", str(len(gpu_ids))]
            if sys.platform == "win32":
                customize_env["USE_LIBUV"] = "0"
                multi_gpu_args = ["--rdzv_backend", "c10d", *multi_gpu_args]
            args[3:3] = multi_gpu_args

    return args, customize_env, mixed_precision


def build_train_log_urls(task_id: str) -> dict:
    """Construct full http(s) URLs for the train-log viewer + SSE stream.

    Reads host/port from ``MIKAZUKI_HOST`` / ``MIKAZUKI_PORT`` (set in
    ``mikazuki/app/application.py`` at boot). ``0.0.0.0`` is normalized to
    ``127.0.0.1`` so the printed URL is actually clickable from the host
    machine.
    """

    host = os.environ.get("MIKAZUKI_HOST", "127.0.0.1") or "127.0.0.1"
    port = os.environ.get("MIKAZUKI_PORT", "28000") or "28000"
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    base = f"http://{display_host}:{port}"
    return {
        "base": base,
        "viewer": f"{base}/train-log?task_id={task_id}",
        "stream": f"{base}/api/train/log/stream/{task_id}",
    }


def _announce_train_log(task_id: str, urls: dict) -> None:
    """Print a prominent, clickable banner pointing at the live log viewer."""

    viewer = urls["viewer"]
    stream = urls["stream"]
    banner = (
        "\n"
        "  Train log viewer (open in browser):\n"
        f"    {viewer}\n"
        f"    SSE stream: {stream}\n"
        f"    task_id   : {task_id}\n"
    )
    log.info(banner)

    if _truthy_env("MIKAZUKI_AUTO_OPEN_TRAIN_LOG"):
        try:
            webbrowser.open(viewer)
        except Exception as exc:  # noqa: BLE001 — best-effort UX nicety
            log.warning(f"Failed to auto-open train log in browser: {exc}")


def run_train(toml_path: str,
              trainer_file: str = "./scripts/train_network.py",
              gpu_ids: Optional[list] = None,
              cpu_threads: Optional[int] = 2,
              metadata: Optional[dict] = None):
    log.info(f"Training started with config file / 训练开始，使用配置文件: {toml_path}")
    cpu_threads = cpu_threads or 2
    args, customize_env, mixed_precision = build_accelerate_train_command(
        trainer_file=trainer_file,
        toml_path=toml_path,
        cpu_threads=cpu_threads,
        gpu_ids=gpu_ids,
    )

    if mixed_precision:
        log.info(
            "Accelerate launch mixed_precision=%s (from TOML); sd-scripts reads the same key "
            "from --config_file. / Accelerate 与训练脚本均使用 mixed_precision=%s",
            mixed_precision,
            mixed_precision,
        )
    else:
        log.warning(
            "No mixed_precision in %s; accelerate launch may default to 'no'. "
            "Set mixed_precision in the GUI (bf16/fp16). / 配置中未设置 mixed_precision，"
            "Accelerate 可能默认为 no",
            toml_path,
        )

    if gpu_ids:
        log.info(f"Using GPU(s) / 使用 GPU: {gpu_ids}")

    task_metadata = {
        "backend": "standard",
        "config_path": str(Path(toml_path).resolve()),
        "trainer_file": trainer_file,
        "cwd": str(Path.cwd()),
        "mixed_precision": mixed_precision,
        "cpu_threads": cpu_threads,
        "gpu_ids": list(gpu_ids or []),
        "command": [str(part) for part in args],
    }
    task_metadata.update(metadata or {})

    if not (task := tm.create_task(args, customize_env, metadata=task_metadata)):
        return APIResponse(
            status="error",
            message="Failed to create task / 无法创建训练任务",
            data=task_metadata,
        )

    urls = build_train_log_urls(task.task_id)
    _announce_train_log(task.task_id, urls)

    def _run():
        try:
            task.execute()
            task.wait()
            rc = task.process.returncode if task.process else -1
            if rc != 0:
                log.error(f"Training failed / 训练失败 (exit {rc})")
            else:
                log.info(f"Training finished / 训练完成")
        except Exception as e:
            log.error(f"An error occurred when training / 训练出现致命错误: {e}")

    coro = asyncio.to_thread(_run)
    asyncio.create_task(coro)

    return APIResponse(
        status="success",
        message=f"Training started / 训练开始 ID: {task.task_id}",
        data={
            "task_id": task.task_id,
            "train_log_path": "/train-log",
            "train_log_query": f"task_id={task.task_id}",
            "train_log_stream": f"/api/train/log/stream/{task.task_id}",
            # Full clickable URLs (new in this release).
            "train_log_url": urls["viewer"],
            "train_log_stream_url": urls["stream"],
            "metadata": task_metadata,
            "config_path": task_metadata["config_path"],
            "trainer_file": trainer_file,
        },
    )


def run_anima_fast_train(toml_path: str,
                         runtime,
                         gpu_ids: Optional[list] = None,
                         metadata: Optional[dict] = None):
    log.info(f"Anima Fast training started with config file / Anima Fast 训练开始，使用配置文件: {toml_path}")
    task_id = str(uuid.uuid4())
    spec = build_launch_spec(runtime, Path(toml_path), task_id, gpu_ids)
    log_file = Path(metadata.get("logging_dir") or runtime.logging_dir) / f"{Path(toml_path).stem}.launch.log" if metadata else runtime.logging_dir / f"{Path(toml_path).stem}.launch.log"
    task_metadata = {
        "backend": "anima-lora-fast",
        "config_path": str(Path(toml_path).resolve()),
        "anima_root": str(runtime.anima_root),
        "anima_python": str(runtime.python),
        "output_dir": str(runtime.output_dir),
        "logging_dir": str(runtime.logging_dir),
        "log_file": str(log_file),
    }
    task_metadata.update(metadata or {})

    if not (task := tm.create_task(spec.command, spec.env, metadata=task_metadata, cwd=str(spec.cwd), task_id=task_id)):
        return APIResponse(status="error", message="Failed to create Anima Fast task / 无法创建 Anima Fast 训练任务")

    resolver = default_resolver(Path.cwd())
    urls = {
        "viewer": resolver.train_log_viewer_url(task.task_id),
        "stream": resolver.public_base_url().rstrip("/") + resolver.train_log_stream_path(task.task_id),
        "base": resolver.public_base_url(),
    }
    _announce_train_log(task.task_id, urls)

    def _run():
        try:
            task.execute()
            task.wait()
            rc = task.process.returncode if task.process else -1
            if rc != 0:
                log.error(f"Anima Fast training failed / Anima Fast 训练失败 (exit {rc})")
            else:
                log.info("Anima Fast training finished / Anima Fast 训练完成")
        except Exception as e:
            log.error(f"An error occurred when Anima Fast training / Anima Fast 训练出现致命错误: {e}")

    coro = asyncio.to_thread(_run)
    asyncio.create_task(coro)

    return APIResponse(
        status="success",
        message=f"Anima Fast training started / Anima Fast 训练开始 ID: {task.task_id}",
        data={
            "task_id": task.task_id,
            "train_log_path": "/train-log",
            "train_log_query": f"task_id={task.task_id}",
            "train_log_stream": f"/api/train/log/stream/{task.task_id}",
            "train_log_url": urls["viewer"],
            "train_log_stream_url": urls["stream"],
            "metadata": task_metadata,
            "log_file": str(log_file),
        },
    )
