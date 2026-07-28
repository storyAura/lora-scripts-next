from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from threading import Lock

from packaging.version import InvalidVersion, Version


class HardwareCapabilityError(RuntimeError):
    """Raised when a requested training kernel is unsupported by the active stack."""


@dataclass(frozen=True, slots=True)
class HardwareCapabilityDecision:
    usable: bool
    reason: str


@dataclass(frozen=True, slots=True)
class FP8ProbeEnvironment:
    device_name: str
    compute_capability: tuple[int, int]
    torch_version: str
    cuda_version: str


_FP8_PROBE_CACHE: dict[FP8ProbeEnvironment, HardwareCapabilityDecision] = {}
_FP8_PROBE_LOCK = Lock()


def _parsed_version(version_text: str, component_name: str) -> Version | None:
    if not version_text or version_text in {"none", "not-installed"}:
        return None
    try:
        return Version(version_text)
    except InvalidVersion:
        return None


def evaluate_flash_attention_hardware(
    cuda_available: bool,
    compute_capability_major: int,
    compute_capability_minor: int,
    cuda_version: str,
    torch_version: str,
    flash_attn_version: str,
) -> HardwareCapabilityDecision:
    if not cuda_available:
        return HardwareCapabilityDecision(False, "FlashAttention training requires CUDA")
    if (compute_capability_major, compute_capability_minor) < (8, 0):
        return HardwareCapabilityDecision(
            False,
            "FlashAttention 2 training requires NVIDIA compute capability>=8.0, "
            f"got {compute_capability_major}.{compute_capability_minor}",
        )
    parsed_cuda = _parsed_version(cuda_version, "CUDA")
    if parsed_cuda is None or parsed_cuda < Version("12.0"):
        return HardwareCapabilityDecision(
            False,
            f"FlashAttention 2 training requires CUDA>=12.0, got {cuda_version!r}",
        )
    parsed_torch = _parsed_version(torch_version, "torch")
    if parsed_torch is None or parsed_torch < Version("2.2.0"):
        return HardwareCapabilityDecision(
            False,
            f"FlashAttention 2 training requires torch>=2.2, got {torch_version!r}",
        )
    parsed_flash = _parsed_version(flash_attn_version, "flash-attn")
    if parsed_flash is None or parsed_flash < Version("2.5.0"):
        return HardwareCapabilityDecision(
            False,
            f"FlashAttention training requires flash-attn>=2.5.0, got {flash_attn_version!r}",
        )
    return HardwareCapabilityDecision(True, "")


def evaluate_fp8_training_hardware(
    cuda_available: bool,
    compute_capability_major: int,
    compute_capability_minor: int,
    cuda_version: str,
    torch_version: str,
) -> HardwareCapabilityDecision:
    if not cuda_available:
        return HardwareCapabilityDecision(False, "FP8 frozen-base training requires CUDA")
    if (compute_capability_major, compute_capability_minor) < (8, 9):
        return HardwareCapabilityDecision(
            False,
            "FP8 frozen-base training requires native FP8 tensor cores with compute capability>=8.9, "
            f"got {compute_capability_major}.{compute_capability_minor}",
        )
    parsed_cuda = _parsed_version(cuda_version, "CUDA")
    if parsed_cuda is None or parsed_cuda < Version("12.0"):
        return HardwareCapabilityDecision(
            False,
            f"FP8 frozen-base training requires CUDA>=12.0, got {cuda_version!r}",
        )
    parsed_torch = _parsed_version(torch_version, "torch")
    if parsed_torch is None or parsed_torch < Version("2.1.0"):
        return HardwareCapabilityDecision(
            False,
            f"FP8 frozen-base training requires torch>=2.1, got {torch_version!r}",
        )
    return HardwareCapabilityDecision(True, "")


def _current_fp8_environment() -> FP8ProbeEnvironment:
    import torch

    if not torch.cuda.is_available():
        return FP8ProbeEnvironment(
            device_name="cpu",
            compute_capability=(0, 0),
            torch_version=str(torch.__version__),
            cuda_version=str(torch.version.cuda or "none"),
        )
    return FP8ProbeEnvironment(
        device_name=torch.cuda.get_device_name(0),
        compute_capability=torch.cuda.get_device_capability(0),
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda or "none"),
    )


def probe_fp8_frozen_linear_training() -> HardwareCapabilityDecision:
    import torch

    environment = _current_fp8_environment()
    with _FP8_PROBE_LOCK:
        cached = _FP8_PROBE_CACHE.get(environment)
    if cached is not None:
        return cached

    hardware_decision = evaluate_fp8_training_hardware(
        torch.cuda.is_available(),
        environment.compute_capability[0],
        environment.compute_capability[1],
        environment.cuda_version,
        environment.torch_version,
    )
    if not hardware_decision.usable:
        with _FP8_PROBE_LOCK:
            _FP8_PROBE_CACHE[environment] = hardware_decision
        return hardware_decision

    try:
        device = torch.device("cuda:0")
        linear = torch.nn.Linear(
            32,
            24,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        linear.requires_grad_(False)
        linear.to(dtype=torch.float8_e4m3fn)
        value = torch.randn(
            2,
            32,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = linear(value)
        if not bool(torch.isfinite(output).all()):
            result = HardwareCapabilityDecision(False, "FP8 probe produced non-finite output")
        else:
            output.float().square().mean().backward()
            if value.grad is None:
                result = HardwareCapabilityDecision(
                    False,
                    "FP8 probe did not produce input gradients",
                )
            elif not bool(torch.isfinite(value.grad).all()):
                result = HardwareCapabilityDecision(
                    False,
                    "FP8 probe produced non-finite input gradients",
                )
            else:
                torch.cuda.synchronize(device)
                result = HardwareCapabilityDecision(True, "")
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as error:
        result = HardwareCapabilityDecision(
            False,
            f"FP8 frozen Linear forward/backward probe failed with {type(error).__name__}: {error}",
        )

    with _FP8_PROBE_LOCK:
        _FP8_PROBE_CACHE[environment] = result
    return result


def require_fp8_frozen_base_training() -> None:
    result = probe_fp8_frozen_linear_training()
    if not result.usable:
        raise HardwareCapabilityError(
            "fp8_base was explicitly requested but the hardware/kernel probe failed: "
            + result.reason
        )


def installed_flash_attention_version() -> str:
    try:
        return version("flash-attn")
    except PackageNotFoundError:
        return "not-installed"
