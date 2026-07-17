from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download, try_to_load_from_cache


def resolve_checkpoint_path(
    *,
    checkpoint: str | None,
    hf_checkpoint: str | None,
    default_filename: str = "checkpoint.pth",
) -> str:
    if checkpoint is not None:
        path = Path(checkpoint).expanduser()
        return str(path)

    if hf_checkpoint is None:
        raise ValueError("Either checkpoint or hf_checkpoint must be provided.")

    repo_id, filename = _split_hf_checkpoint(hf_checkpoint)
    if filename is not None:
        return _download_hf_checkpoint(repo_id, filename)

    try:
        return _download_hf_checkpoint(repo_id, default_filename)
    except Exception:
        snapshot_root = Path(snapshot_download(repo_id=repo_id))
        candidates = sorted(snapshot_root.rglob("*.pth"))
        if len(candidates) == 1:
            print(f"[INFO] Using checkpoint from Hugging Face snapshot: {candidates[0]}")
            return str(candidates[0])
        if not candidates:
            raise FileNotFoundError(
                f"No .pth checkpoint found in Hugging Face repo '{repo_id}'. "
                f"Pass --hf_checkpoint as '{repo_id}/path/to/file.pth'."
            )
        formatted = "\n".join(f"  - {path.relative_to(snapshot_root)}" for path in candidates[:20])
        raise ValueError(
            f"Multiple .pth checkpoints found in Hugging Face repo '{repo_id}'. "
            f"Pass an explicit file path with --hf_checkpoint.\n{formatted}"
        )


def _split_hf_checkpoint(value: str) -> tuple[str, str | None]:
    parts = value.split("/")
    if len(parts) < 2:
        raise ValueError(
            "--hf_checkpoint must be a Hugging Face repo id like 'user/repo' "
            "or 'user/repo/path/to/checkpoint.pth'."
        )
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:]) if len(parts) > 2 else None
    return repo_id, filename


def _download_hf_checkpoint(repo_id: str, filename: str) -> str:
    cached_path = try_to_load_from_cache(repo_id=repo_id, filename=filename)
    if isinstance(cached_path, str):
        print(f"[INFO] Using cached Hugging Face checkpoint: {cached_path}")
        return cached_path

    if cached_path is not None:
        print(f"[INFO] Checkpoint is not cached and is marked missing on Hugging Face: repo={repo_id}, file={filename}")
    else:
        print(f"[INFO] Downloading Hugging Face checkpoint: repo={repo_id}, file={filename}")
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"[INFO] Downloaded Hugging Face checkpoint: {path}")
    return path
