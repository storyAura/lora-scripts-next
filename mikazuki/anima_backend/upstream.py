from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


_SUBMODULE_HINT = (
    "Anima backend requires the pinned sd-scripts submodule at {path}.\n"
    "Run `git submodule update --init --recursive` from the repo root to "
    "initialize it.\n"
    "To intentionally use a different commit, set ANIMA_ALLOW_COMMIT_DRIFT=1 "
    "to downgrade the version check to a warning."
)


def load_toml_file(path: Path) -> dict:
    try:
        import tomllib

        with path.open("rb") as config_file:
            return tomllib.load(config_file)
    except ModuleNotFoundError:
        import toml

        return toml.load(path)


def resolve_upstream_path(root: Path, config_path: Path | None = None) -> Path:
    config_path = config_path or root / "config" / "anima_backend.toml"
    data = load_toml_file(config_path)

    upstream_path = Path(data["backend"]["upstream_path"])
    if not upstream_path.is_absolute():
        upstream_path = root / upstream_path
    return upstream_path.resolve()


def pinned_commit(root: Path, config_path: Path | None = None) -> str:
    config_path = config_path or root / "config" / "anima_backend.toml"
    data = load_toml_file(config_path)
    return str(data["backend"]["pinned_commit"])


def _is_initialized_git_checkout(upstream_path: Path) -> bool:
    """Return True if upstream_path is itself a git work tree.

    A submodule that has not been initialized leaves an empty directory (or no
    directory at all) under ``vendor/sd-scripts``. Running ``git rev-parse`` in
    that location would walk up and resolve to the superproject, which silently
    produces a HEAD that has nothing to do with sd-scripts. Guard against that
    by verifying both ``.git`` exists in the directory and that git itself
    reports the directory as the top-level of its work tree.
    """

    if not upstream_path.exists() or not (upstream_path / ".git").exists():
        return False
    try:
        result = subprocess.run(
            ["git", "-C", str(upstream_path), "rev-parse", "--show-toplevel"],
            text=True,
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    try:
        toplevel = Path(result.stdout.strip()).resolve()
    except OSError:
        return False
    return toplevel == upstream_path.resolve()


def _auto_init_disabled() -> bool:
    return os.environ.get("ANIMA_SKIP_AUTO_INIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _try_init_submodule(root: Path, upstream_path: Path) -> bool:
    """Best-effort `git submodule update --init --recursive` for upstream_path.

    Returns True if the submodule appears initialized afterwards. The function
    swallows all errors; callers should still re-check with
    ``_is_initialized_git_checkout`` and raise a friendly error if False.
    """

    if _auto_init_disabled():
        return _is_initialized_git_checkout(upstream_path)
    if not (root / ".git").exists():
        # Not a git checkout (e.g. ZIP download) — nothing we can do.
        return _is_initialized_git_checkout(upstream_path)
    try:
        rel = upstream_path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = Path("vendor") / "sd-scripts"

    print(
        f"[Anima backend] vendor/sd-scripts submodule not initialized; "
        f"running `git submodule update --init --recursive -- {rel}` ...",
        file=sys.stderr,
    )
    try:
        subprocess.run(
            ["git", "-C", str(root), "submodule", "update",
             "--init", "--recursive", "--", str(rel)],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(
            f"[Anima backend] Auto-init failed: {exc}. "
            "Falling back to manual instructions.",
            file=sys.stderr,
        )
        return False
    return _is_initialized_git_checkout(upstream_path)


def current_upstream_commit(upstream_path: Path) -> str:
    if not _is_initialized_git_checkout(upstream_path):
        raise RuntimeError(_SUBMODULE_HINT.format(path=upstream_path))
    result = subprocess.run(
        ["git", "-C", str(upstream_path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _drift_allowed() -> bool:
    return os.environ.get("ANIMA_ALLOW_COMMIT_DRIFT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def verify_pinned_commit(root: Path, config_path: Path | None = None) -> str:
    upstream_path = resolve_upstream_path(root, config_path)
    expected = pinned_commit(root, config_path)

    if not _is_initialized_git_checkout(upstream_path):
        if upstream_path.exists() and any(upstream_path.iterdir()):
            # Not a git checkout but directory has files (e.g. vendored copy).
            return expected
        # Best-effort auto-init before giving up.
        _try_init_submodule(root, upstream_path)
    if not _is_initialized_git_checkout(upstream_path):
        if upstream_path.exists() and any(upstream_path.iterdir()):
            return expected
        raise RuntimeError(_SUBMODULE_HINT.format(path=upstream_path))

    actual = current_upstream_commit(upstream_path)
    if actual != expected:
        message = (
            "Pinned sd-scripts commit mismatch: "
            f"config expects {expected}, but {upstream_path} is at {actual}"
        )
        strict = os.environ.get("ANIMA_STRICT_COMMIT", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if strict and not _drift_allowed():
            raise RuntimeError(
                message
                + "\nSet ANIMA_ALLOW_COMMIT_DRIFT=1 to bypass this check."
            )
        print(
            f"[Anima backend] WARNING: {message} (continuing anyway)",
            file=sys.stderr,
        )
    return actual


def upstream_entrypoint(upstream_path: Path, entrypoint: str) -> Path:
    script = upstream_path / entrypoint
    if not script.exists():
        raise FileNotFoundError(f"Upstream Anima trainer not found: {script}")
    return script


def prefer_upstream_imports(upstream_path: Path) -> None:
    upstream = str(upstream_path.resolve())
    if upstream in sys.path:
        sys.path.remove(upstream)
    sys.path.insert(0, upstream)


_LYCORIS_SYNC_HINT = (
    "Installed LyCORIS is the upstream pip package, not this repo's vendored copy.\n"
    "The vendored copy carries the local algorithms (glokr / bokr / bora / gsokr / "
    "glora_boft / cdka) and the fp32-safe merged forward; upstream LoKr silently "
    "absorbs small bf16 updates during training.\n"
    "Fix: python scripts/sync_vendored_lycoris.py\n"
    "Set ANIMA_ALLOW_LYCORIS_DRIFT=1 to continue anyway."
)


def verify_vendored_lycoris() -> None:
    """Refuse to train through lycoris.kohya when upstream LyCORIS is installed.

    Only the vendored tree computes the LoKr merged delta in fp32
    (`compute_merged_delta`); its absence in the imported module means pip
    reinstalled upstream over the synced copy and training would silently run
    the numerically unsafe forward.
    """
    if os.environ.get("ANIMA_ALLOW_LYCORIS_DRIFT", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        return

    import importlib

    try:
        lokr = importlib.import_module("lycoris.modules.lokr")
    except Exception as exc:
        raise RuntimeError(
            f"LyCORIS is not importable ({exc}).\n{_LYCORIS_SYNC_HINT}"
        ) from exc
    if getattr(lokr, "compute_merged_delta", None) is None:
        raise RuntimeError(_LYCORIS_SYNC_HINT)
