from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


SD_SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "sd-scripts"
if str(SD_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SD_SCRIPTS_ROOT))


class Block(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(features, features, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.proj(inputs)


class FakeDiT(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(features)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](inputs)


def _build_network(
    module_name: str,
    *,
    features: int,
    rank: int,
    alpha: float,
    network_args: dict[str, object],
) -> tuple[nn.Module, FakeDiT, nn.Module]:
    module = importlib.import_module(f"networks.{module_name}")
    model = FakeDiT(features)
    network = module.create_network(
        1.0,
        rank,
        alpha,
        None,
        [],
        model,
        None,
        **network_args,
    )
    network.apply_to([], model, False, True)
    adapters = list(network.unet_loras)
    assert len(adapters) == 1
    return network, model, adapters[0]


def _clone_model(source: FakeDiT) -> FakeDiT:
    model = FakeDiT(source.blocks[0].proj.in_features)
    model.load_state_dict(source.state_dict(), strict=True)
    return model


def test_delora_matches_reference_formula_and_has_finite_gradients() -> None:
    torch.manual_seed(11)
    network, model, adapter = _build_network(
        "delora_anima",
        features=6,
        rank=3,
        alpha=3,
        network_args={"delora_lambda": 7.0},
    )
    inputs = torch.randn(2, 4, 6)
    base = adapter.org_forward(inputs)

    with torch.no_grad():
        adapter.lora_down.weight.copy_(torch.randn_like(adapter.lora_down.weight))
        adapter.lora_up.weight.copy_(torch.randn_like(adapter.lora_up.weight))
        adapter.delora_lambda.fill_(5.0)

    matrix_a = adapter.lora_down.weight
    matrix_b = adapter.lora_up.weight
    norm_a = torch.linalg.vector_norm(matrix_a.float(), dim=1).clamp_min(1e-4)
    norm_b = torch.linalg.vector_norm(matrix_b.float(), dim=0).clamp_min(1e-4)
    diagonal = adapter.delora_lambda.float() / adapter.lora_dim / (norm_a * norm_b)
    delta = (matrix_b.float() * diagonal.unsqueeze(0)) @ matrix_a.float()
    delta = delta * adapter.delora_w_norm.float().unsqueeze(0)
    expected = base + torch.nn.functional.linear(inputs, delta.to(inputs.dtype))
    actual = model(inputs)
    torch.testing.assert_close(actual, expected)

    loss = actual.square().mean()
    loss.backward()
    assert adapter.lora_down.weight.grad is not None
    assert adapter.lora_up.weight.grad is not None
    assert adapter.delora_lambda.grad is not None
    assert torch.isfinite(adapter.lora_down.weight.grad).all()
    assert torch.isfinite(adapter.lora_up.weight.grad).all()
    assert torch.isfinite(adapter.delora_lambda.grad).all()
    assert network.is_mergeable()


def test_delora_zero_initialization_and_merge_equivalence() -> None:
    torch.manual_seed(12)
    source_model = FakeDiT(5)
    pristine_state = {
        key: value.detach().clone()
        for key, value in source_model.state_dict().items()
    }
    module = importlib.import_module("networks.delora_anima")
    network = module.create_network(
        0.75,
        2,
        2,
        None,
        [],
        source_model,
        None,
        delora_lambda=4.0,
    )
    inputs = torch.randn(2, 5)
    baseline = source_model(inputs)
    network.apply_to([], source_model, False, True)
    torch.testing.assert_close(source_model(inputs), baseline)
    adapter = network.unet_loras[0]
    with torch.no_grad():
        adapter.lora_up.weight.normal_()
    expected = source_model(inputs)
    state = {
        key: value.detach().clone()
        for key, value in network.state_dict().items()
    }

    merged_model = FakeDiT(5)
    merged_model.load_state_dict(pristine_state, strict=True)
    merge_network, loaded_state = module.create_network_from_weights(
        0.75,
        None,
        None,
        [],
        merged_model,
        weights_sd=state,
        for_inference=True,
    )
    merge_network.merge_to([], merged_model, loaded_state)
    torch.testing.assert_close(merged_model(inputs), expected)
    assert any(key.endswith(".delora_w_norm") for key in state)


@pytest.mark.parametrize("invalid_value", (0, -1, float("nan"), float("inf")))
def test_delora_rejects_invalid_lambda(invalid_value: float) -> None:
    with pytest.raises(ValueError, match="delora_lambda"):
        _build_network(
            "delora_anima",
            features=4,
            rank=2,
            alpha=2,
            network_args={"delora_lambda": invalid_value},
        )


@pytest.mark.parametrize("mixer_init", ("kaiming", "identity", "orthogonal"))
def test_moslora_formula_initialization_and_standard_export(mixer_init: str) -> None:
    torch.manual_seed(21)
    network, model, adapter = _build_network(
        "moslora_anima",
        features=6,
        rank=3,
        alpha=6,
        network_args={"moslora_mixer_init": mixer_init},
    )
    inputs = torch.randn(2, 6)
    torch.testing.assert_close(model(inputs), adapter.org_forward(inputs))

    with torch.no_grad():
        adapter.lora_up.weight.normal_()
    expected_delta = (
        adapter.lora_up.weight.float()
        @ adapter.lora_mixer.weight.float()
        @ adapter.lora_down.weight.float()
        * adapter.scale
    )
    expected = adapter.org_forward(inputs) + torch.nn.functional.linear(
        inputs,
        expected_delta.to(inputs.dtype),
    )
    torch.testing.assert_close(model(inputs), expected)

    exported = network.to_standard_lora_state_dict()
    prefix = adapter.lora_name
    assert f"{prefix}.lora_mixer.weight" not in exported
    torch.testing.assert_close(
        exported[f"{prefix}.lora_up.weight"].float()
        @ exported[f"{prefix}.lora_down.weight"].float()
        * (exported[f"{prefix}.alpha"].float() / adapter.lora_dim),
        expected_delta,
    )


def test_moslora_merge_equivalence() -> None:
    torch.manual_seed(22)
    module = importlib.import_module("networks.moslora_anima")
    base = FakeDiT(5)
    pristine = {
        key: value.detach().clone()
        for key, value in base.state_dict().items()
    }
    network = module.create_network(
        1.0,
        2,
        2,
        None,
        [],
        base,
        None,
        moslora_mixer_init="identity",
    )
    network.apply_to([], base, False, True)
    with torch.no_grad():
        network.unet_loras[0].lora_up.weight.normal_()
    inputs = torch.randn(3, 5)
    expected = base(inputs)
    state = {
        key: value.detach().clone()
        for key, value in network.state_dict().items()
    }

    merged = FakeDiT(5)
    merged.load_state_dict(pristine, strict=True)
    merge_network, loaded = module.create_network_from_weights(
        1.0,
        None,
        None,
        [],
        merged,
        weights_sd=state,
        for_inference=True,
    )
    merge_network.merge_to([], merged, loaded)
    torch.testing.assert_close(merged(inputs), expected)


def test_deft_identity_initialization_formula_and_qr_projection() -> None:
    torch.manual_seed(31)
    network, model, adapter = _build_network(
        "deft_anima",
        features=6,
        rank=3,
        alpha=3,
        network_args={
            "deft_decomposition_method": "qr",
            "deft_alpha": 6,
            "deft_init_scale": 1.0,
        },
    )
    inputs = torch.randn(2, 4, 6)
    base = adapter.org_forward(inputs)
    torch.testing.assert_close(model(inputs), base, atol=2e-6, rtol=2e-6)

    with torch.no_grad():
        adapter.deft_R.add_(torch.randn_like(adapter.deft_R) * 0.1)
    q_matrix, right = adapter.projector_factors()
    torch.testing.assert_close(
        q_matrix.transpose(0, 1) @ q_matrix,
        torch.eye(adapter.lora_dim),
        atol=1e-5,
        rtol=1e-5,
    )
    base_product = base
    correction = (base_product.float() @ right) @ q_matrix.transpose(0, 1)
    injection = (
        torch.nn.functional.linear(inputs.float(), adapter.deft_R.float())
        @ q_matrix.transpose(0, 1)
        * adapter.deft_scaling
    )
    expected = base.float() - correction + injection
    torch.testing.assert_close(model(inputs).float(), expected, atol=2e-5, rtol=2e-5)

    model(inputs).square().mean().backward()
    assert adapter.deft_P.grad is not None
    assert adapter.deft_R.grad is not None
    assert torch.isfinite(adapter.deft_P.grad).all()
    assert torch.isfinite(adapter.deft_R.grad).all()
    assert network.is_mergeable()


@pytest.mark.parametrize("method", ("qr", "relu"))
def test_deft_merge_equivalence(method: str) -> None:
    torch.manual_seed(32)
    module = importlib.import_module("networks.deft_anima")
    base = FakeDiT(5)
    pristine = {
        key: value.detach().clone()
        for key, value in base.state_dict().items()
    }
    network = module.create_network(
        0.5,
        2,
        2,
        None,
        [],
        base,
        None,
        deft_decomposition_method=method,
        deft_alpha=4,
        deft_init_scale=1.0,
    )
    network.apply_to([], base, False, True)
    with torch.no_grad():
        network.unet_loras[0].deft_R.add_(0.2)
    inputs = torch.randn(4, 5)
    expected = base(inputs)
    state = {
        key: value.detach().clone()
        for key, value in network.state_dict().items()
    }

    merged = FakeDiT(5)
    merged.load_state_dict(pristine, strict=True)
    merge_network, loaded = module.create_network_from_weights(
        0.5,
        None,
        None,
        [],
        merged,
        weights_sd=state,
        for_inference=True,
    )
    merge_network.merge_to([], merged, loaded)
    torch.testing.assert_close(merged(inputs), expected, atol=2e-5, rtol=2e-5)


def test_deft_rejects_unknown_decomposition() -> None:
    with pytest.raises(ValueError, match="deft_decomposition_method"):
        _build_network(
            "deft_anima",
            features=4,
            rank=2,
            alpha=2,
            network_args={"deft_decomposition_method": "svd"},
        )


def test_deft_random_initialization_is_explicitly_non_identity() -> None:
    torch.manual_seed(33)
    _, model, adapter = _build_network(
        "deft_anima",
        features=5,
        rank=2,
        alpha=2,
        network_args={
            "deft_decomposition_method": "qr",
            "deft_init_weights": False,
            "deft_init_scale": 1.0,
        },
    )
    inputs = torch.randn(2, 5)
    assert not torch.allclose(model(inputs), adapter.org_forward(inputs))


@pytest.mark.parametrize("use_idwt", (False, True))
def test_waveft_sparse_spectrum_is_deterministic_and_differentiable(
    use_idwt: bool,
) -> None:
    torch.manual_seed(41)
    args = {
        "waveft_n_frequency": 7,
        "waveft_scaling": 3.0,
        "waveft_random_loc_seed": 777,
        "waveft_use_idwt": use_idwt,
        "waveft_wavelet_family": "db1",
    }
    network, model, adapter = _build_network(
        "waveft_anima",
        features=6,
        rank=4,
        alpha=4,
        network_args=args,
    )
    _, _, second = _build_network(
        "waveft_anima",
        features=6,
        rank=4,
        alpha=4,
        network_args=args,
    )
    assert adapter.waveft_spectrum.numel() == 7
    torch.testing.assert_close(adapter.waveft_indices, second.waveft_indices)

    inputs = torch.randn(3, 6)
    torch.testing.assert_close(model(inputs), adapter.org_forward(inputs))
    with torch.no_grad():
        adapter.waveft_spectrum.normal_()
    delta = adapter.get_delta_weight()
    assert delta.shape == (6, 6)
    assert torch.isfinite(delta).all()
    if not use_idwt:
        assert int(torch.count_nonzero(delta)) == 7
    output = model(inputs)
    output.square().mean().backward()
    assert adapter.waveft_spectrum.grad is not None
    assert torch.isfinite(adapter.waveft_spectrum.grad).all()
    assert network.is_mergeable()


def test_waveft_db1_inverse_matches_orthonormal_haar_reconstruction() -> None:
    module = importlib.import_module("networks.waveft_anima")
    reconstructed = module._haar_inverse_2d(
        torch.tensor([[1.0]]),
        torch.tensor([[2.0]]),
        torch.tensor([[3.0]]),
        torch.tensor([[4.0]]),
    )
    torch.testing.assert_close(
        reconstructed,
        torch.tensor([[5.0, -2.0], [-1.0, 0.0]]),
    )


@pytest.mark.parametrize("use_idwt", (False, True))
def test_waveft_checkpoint_is_self_contained_and_mergeable(use_idwt: bool) -> None:
    torch.manual_seed(42)
    module = importlib.import_module("networks.waveft_anima")
    base = FakeDiT(5)
    pristine = {
        key: value.detach().clone()
        for key, value in base.state_dict().items()
    }
    network = module.create_network(
        0.8,
        3,
        3,
        None,
        [],
        base,
        None,
        waveft_n_frequency=9,
        waveft_scaling=2.5,
        waveft_random_loc_seed=91,
        waveft_use_idwt=use_idwt,
        waveft_wavelet_family="db1",
    )
    network.apply_to([], base, False, True)
    with torch.no_grad():
        network.unet_loras[0].waveft_spectrum.normal_()
    inputs = torch.randn(2, 5)
    expected = base(inputs)
    state = {
        key: value.detach().clone()
        for key, value in network.state_dict().items()
    }
    assert any(key.endswith(".waveft_indices") for key in state)
    assert any(key.endswith(".waveft_scaling") for key in state)

    merged = FakeDiT(5)
    merged.load_state_dict(pristine, strict=True)
    merge_network, loaded = module.create_network_from_weights(
        0.8,
        None,
        None,
        [],
        merged,
        weights_sd=state,
        for_inference=True,
    )
    merge_network.merge_to([], merged, loaded)
    torch.testing.assert_close(merged(inputs), expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("invalid_frequency", (0, -1, 17))
def test_waveft_rejects_invalid_frequency(invalid_frequency: int) -> None:
    with pytest.raises(ValueError, match="waveft_n_frequency"):
        _build_network(
            "waveft_anima",
            features=4,
            rank=2,
            alpha=2,
            network_args={"waveft_n_frequency": invalid_frequency},
        )


@pytest.mark.parametrize(
    "invalid_seed",
    (-1, 1.5, True, "not-an-integer"),
)
def test_waveft_rejects_invalid_random_location_seed(
    invalid_seed: object,
) -> None:
    with pytest.raises(ValueError, match="waveft_random_loc_seed"):
        _build_network(
            "waveft_anima",
            features=4,
            rank=2,
            alpha=2,
            network_args={
                "waveft_n_frequency": 4,
                "waveft_random_loc_seed": invalid_seed,
            },
        )


def test_timestep_lora_requires_timestep_and_never_silently_uses_full_rank() -> None:
    torch.manual_seed(51)
    network, model, adapter = _build_network(
        "tlora_anima",
        features=6,
        rank=4,
        alpha=4,
        network_args={
            "tlora_min_rank": 1,
            "tlora_rank_schedule": "linear",
            "tlora_orthogonal_init": False,
        },
    )
    with torch.no_grad():
        adapter.lora_up.weight.normal_()
    inputs = torch.randn(2, 6)
    with pytest.raises(RuntimeError, match="set_current_timestep"):
        model(inputs)

    network.set_current_timestep(torch.tensor([1000.0, 0.0]))
    hidden = adapter.lora_down(inputs)
    mask, scale = adapter._get_tlora_rank_mask_and_scale(hidden)
    assert mask is not None
    assert scale is None
    assert mask[0].sum().item() == 1
    assert mask[1].sum().item() == 4


@pytest.mark.parametrize(
    "invalid_timestep",
    (
        torch.tensor([-1.0]),
        torch.tensor([1001.0]),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
    ),
)
def test_timestep_lora_rejects_invalid_timestep_values(
    invalid_timestep: torch.Tensor,
) -> None:
    network, model, _ = _build_network(
        "tlora_anima",
        features=4,
        rank=2,
        alpha=2,
        network_args={
            "tlora_min_rank": 1,
            "tlora_rank_schedule": "linear",
        },
    )
    network.set_current_timestep(invalid_timestep)
    with pytest.raises(ValueError, match="timestep"):
        model(torch.randn(1, 4))


def test_timestep_lora_can_be_disabled_for_inference() -> None:
    network, model, adapter = _build_network(
        "tlora_anima",
        features=4,
        rank=2,
        alpha=2,
        network_args={
            "tlora_min_rank": 1,
            "tlora_rank_schedule": "linear",
        },
    )
    with torch.no_grad():
        adapter.lora_up.weight.normal_()
    inputs = torch.randn(1, 4)
    network.set_current_timestep(torch.tensor([0.0]))
    assert not torch.allclose(model(inputs), adapter.org_forward(inputs))
    network.set_enabled(False)
    torch.testing.assert_close(model(inputs), adapter.org_forward(inputs))


@pytest.mark.parametrize("invalid_rank", (0, 1.5, True, "invalid"))
def test_timestep_lora_rejects_non_integer_min_rank(
    invalid_rank: object,
) -> None:
    with pytest.raises(ValueError, match="tlora_min_rank"):
        _build_network(
            "tlora_anima",
            features=4,
            rank=2,
            alpha=2,
            network_args={"tlora_min_rank": invalid_rank},
        )


def test_timestep_lora_inference_path_masks_rank_and_is_not_mergeable() -> None:
    torch.manual_seed(52)
    module = importlib.import_module("networks.tlora_anima")
    source = FakeDiT(5)
    network = module.create_network(
        1.0,
        3,
        3,
        None,
        [],
        source,
        None,
        tlora_min_rank=1,
        tlora_rank_schedule="linear",
        tlora_orthogonal_init=False,
    )
    network.apply_to([], source, False, True)
    adapter = network.unet_loras[0]
    with torch.no_grad():
        adapter.lora_up.weight.normal_()
    state = {
        key: value.detach().clone()
        for key, value in network.state_dict().items()
    }

    target = FakeDiT(5)
    loaded_network, loaded_state = module.create_network_from_weights(
        1.0,
        None,
        None,
        [],
        target,
        weights_sd=state,
        for_inference=True,
    )
    loaded_network.apply_to([], target, False, True)
    loaded_network.load_state_dict(loaded_state, strict=False)
    loaded_network.set_current_timestep(torch.tensor([1000.0]))
    inputs = torch.randn(1, 5)
    output_high_noise = target(inputs)
    loaded_network.set_current_timestep(torch.tensor([0.0]))
    output_low_noise = target(inputs)
    assert not torch.allclose(output_high_noise, output_low_noise)
    assert not loaded_network.is_mergeable()
    with pytest.raises(RuntimeError, match="cannot be merged"):
        loaded_network.merge_to([], target, loaded_state)


def test_timestep_lora_rejects_invalid_schedule() -> None:
    with pytest.raises(ValueError, match="tlora_rank_schedule"):
        _build_network(
            "tlora_anima",
            features=4,
            rank=2,
            alpha=2,
            network_args={
                "tlora_min_rank": 1,
                "tlora_rank_schedule": "mystery",
            },
        )


@pytest.mark.parametrize(
    ("module_name", "network_args"),
    (
        ("delora_anima", {"delora_lambda": 15}),
        (
            "waveft_anima",
            {
                "waveft_n_frequency": 16,
                "waveft_scaling": 25,
                "waveft_random_loc_seed": 777,
                "waveft_use_idwt": True,
                "waveft_wavelet_family": "db1",
            },
        ),
        (
            "deft_anima",
            {
                "deft_decomposition_method": "qr",
                "deft_alpha": 0,
                "deft_init_scale": 1,
                "deft_init_weights": True,
            },
        ),
        ("moslora_anima", {"moslora_mixer_init": "kaiming"}),
        (
            "tlora_anima",
            {
                "tlora_min_rank": 2,
                "tlora_rank_schedule": "linear",
                "tlora_orthogonal_init": False,
            },
        ),
    ),
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_advanced_adapter_cuda_bf16_forward_backward(
    module_name: str,
    network_args: dict[str, object],
) -> None:
    torch.manual_seed(61)
    network, model, _ = _build_network(
        module_name,
        features=8,
        rank=4,
        alpha=4,
        network_args=network_args,
    )
    model.to(device="cuda", dtype=torch.bfloat16)
    network.to(device="cuda", dtype=torch.bfloat16)
    set_timestep = getattr(network, "set_current_timestep", None)
    if callable(set_timestep):
        set_timestep(torch.tensor([700.0, 300.0], device="cuda"))
    inputs = torch.randn(
        2,
        8,
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = model(inputs)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in network.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
