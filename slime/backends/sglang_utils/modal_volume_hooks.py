import os
from typing import Any


def _distributed_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def commit_modal_delta_volume(args: Any, version_dir: str, rollout_engines: Any) -> None:
    """Commit a Modal Volume after slime writes a disk delta version directory."""
    if _distributed_rank() != 0:
        return

    volume_name = os.environ.get("SLIME_DELTA_VOLUME_NAME")
    if not volume_name:
        raise RuntimeError("SLIME_DELTA_VOLUME_NAME must be set to commit a Modal delta volume")

    import modal

    modal.Volume.from_name(volume_name, create_if_missing=True).commit()
    print(f"Committed Modal delta volume {volume_name} for {version_dir}", flush=True)
