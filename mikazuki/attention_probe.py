from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from threading import Lock

from mikazuki.hardware_capabilities import evaluate_flash_attention_hardware


@dataclass(frozen=True, slots=True)
class AttentionEnvironmentKey:
    device_name: str
    compute_capability: str
    driver_version: str
    torch_version: str
    cuda_version: str
    flash_attn_version: str
    xformers_version: str


@dataclass(frozen=True, slots=True)
class AttentionProbeResult:
    backend: str
    usable: bool
    reason: str


class AttentionBackendUnavailableError(RuntimeError):
    pass


_SUPPORTED_BACKENDS = ("flash", "xformers", "torch")
_PROBE_CACHE: dict[
    tuple[AttentionEnvironmentKey, str],
    AttentionProbeResult,
] = {}
_PROBE_CACHE_LOCK = Lock()


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def _driver_version(torch_module: object) -> str:
    torch_c = getattr(torch_module, "_C", None)
    getter = getattr(torch_c, "_cuda_getDriverVersion", None)
    if getter is None:
        return "unavailable"
    try:
        return str(getter())
    except RuntimeError as exc:
        return f"unavailable:{type(exc).__name__}"


def _environment_key() -> AttentionEnvironmentKey:
    import torch

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        compute_capability = f"{major}.{minor}"
    else:
        device_name = "cpu"
        compute_capability = "none"
    return AttentionEnvironmentKey(
        device_name=device_name,
        compute_capability=compute_capability,
        driver_version=_driver_version(torch),
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda or "none"),
        flash_attn_version=_package_version("flash-attn"),
        xformers_version=_package_version("xformers"),
    )


def _probe_backend_uncached(backend: str) -> tuple[bool, str]:
    import torch

    if backend in {"flash", "xformers"} and not torch.cuda.is_available():
        return False, f"{backend} requires a CUDA device"
    if backend == "flash":
        compute_capability = torch.cuda.get_device_capability(0)
        hardware_decision = evaluate_flash_attention_hardware(
            torch.cuda.is_available(),
            compute_capability[0],
            compute_capability[1],
            str(torch.version.cuda or "none"),
            str(torch.__version__),
            _package_version("flash-attn"),
        )
        if not hardware_decision.usable:
            return False, hardware_decision.reason

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    try:
        shape = (1, 32, 4, 64)
        q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

        if backend == "flash":
            from flash_attn.flash_attn_interface import flash_attn_func

            output = flash_attn_func(q, k, v, 0.0)
        elif backend == "xformers":
            from xformers.ops import memory_efficient_attention

            output = memory_efficient_attention(q, k, v, p=0.0)
        elif backend == "torch":
            output = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=0.0,
            )
        else:
            raise ValueError(
                f"Unsupported attention backend {backend!r}; "
                f"expected one of {_SUPPORTED_BACKENDS}"
            )

        if not bool(torch.isfinite(output).all()):
            return False, f"{backend} forward produced non-finite values"
        output.float().square().mean().backward()
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if tensor.grad is None:
                return False, f"{backend} backward did not produce {name} gradients"
            if not bool(torch.isfinite(tensor.grad).all()):
                return False, f"{backend} backward produced non-finite {name} gradients"
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return True, ""
    except (
        AssertionError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return (
            False,
            f"{backend} forward/backward probe failed with "
            f"{type(exc).__name__}: {exc}",
        )


def probe_training_attention_backend(backend: str) -> AttentionProbeResult:
    normalized = backend.strip().lower()
    if normalized not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported attention backend {backend!r}; "
            f"expected one of {_SUPPORTED_BACKENDS}"
        )
    key = _environment_key()
    cache_key = (key, normalized)
    with _PROBE_CACHE_LOCK:
        cached = _PROBE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    usable, reason = _probe_backend_uncached(normalized)
    result = AttentionProbeResult(
        backend=normalized,
        usable=usable,
        reason=reason,
    )
    with _PROBE_CACHE_LOCK:
        _PROBE_CACHE[cache_key] = result
    return result


def detect_best_training_attention() -> str:
    failures: list[str] = []
    for backend in _SUPPORTED_BACKENDS:
        result = probe_training_attention_backend(backend)
        if result.usable:
            return backend
        failures.append(result.reason)
    raise AttentionBackendUnavailableError(
        "No training attention backend passed a forward/backward probe: "
        + "; ".join(failures)
    )


def clear_attention_probe_cache() -> None:
    with _PROBE_CACHE_LOCK:
        _PROBE_CACHE.clear()
