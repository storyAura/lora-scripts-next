from __future__ import annotations

from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTHORITATIVE_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
MIRROR_REQUIREMENTS = (
    PROJECT_ROOT / "scripts" / "dev" / "requirements.txt",
    PROJECT_ROOT / "vendor" / "sd-scripts" / "requirements.txt",
)
SHARED_TRAINING_PACKAGES = {
    "accelerate",
    "bitsandbytes",
    "diffusers",
    "einops",
    "ftfy",
    "huggingface-hub",
    "imagesize",
    "lion-pytorch",
    "numpy",
    "opencv-python",
    "prodigy-plus-schedule-free",
    "prodigyopt",
    "pytorch-optimizer",
    "rich",
    "safetensors",
    "schedulefree",
    "sentencepiece",
    "tensorboard",
    "toml",
    "transformers",
    "voluptuous",
}


def _parse_requirements(path: Path) -> dict[str, Requirement]:
    parsed: dict[str, Requirement] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        content = raw_line.split("#", maxsplit=1)[0].strip()
        if not content or content.startswith("-"):
            continue
        requirement = Requirement(content)
        name = canonicalize_name(requirement.name)
        if name in parsed:
            raise AssertionError(
                f"{path}:{line_number} duplicates requirement {name!r}"
            )
        parsed[name] = requirement
    return parsed


def _signature(requirement: Requirement) -> tuple[object, ...]:
    return (
        tuple(sorted(requirement.extras)),
        str(requirement.specifier),
        str(requirement.marker) if requirement.marker is not None else None,
        requirement.url,
    )


def test_root_requirements_has_unique_project_names() -> None:
    _parse_requirements(AUTHORITATIVE_REQUIREMENTS)


def test_training_dependency_mirrors_match_root_pins() -> None:
    authoritative = _parse_requirements(AUTHORITATIVE_REQUIREMENTS)
    for package_name in SHARED_TRAINING_PACKAGES:
        assert package_name in authoritative, (
            f"{AUTHORITATIVE_REQUIREMENTS} is missing shared package "
            f"{package_name!r}"
        )

    for mirror_path in MIRROR_REQUIREMENTS:
        mirrored = _parse_requirements(mirror_path)
        for package_name in sorted(SHARED_TRAINING_PACKAGES):
            assert package_name in mirrored, (
                f"{mirror_path} is missing shared package {package_name!r}"
            )
            assert _signature(mirrored[package_name]) == _signature(
                authoritative[package_name]
            ), (
                f"{mirror_path} must mirror {package_name!r} from "
                f"{AUTHORITATIVE_REQUIREMENTS}: "
                f"{mirrored[package_name]!s} != "
                f"{authoritative[package_name]!s}"
            )
