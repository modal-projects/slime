"""Deprecated shim (restructure 2026-08-26): moved to agentic_rl/launch/modal_train.py.

Keeps W&B-recorded commands (`modal run agentic_rl/retro/modal_train.py::train`) working.
"""

from agentic_rl.launch.modal_train import (  # noqa: F401
    app,
    convert_hf_to_megatron_checkpoint,
    download_data,
    download_model,
    post_process_data,
    show_config,
    train,
)
