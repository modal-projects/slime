"""Materialize ``open-thoughts/OpenThoughts-Agent-v1-RL`` as slime prompt data.

Each HF row packs a harbor task dir as a gzipped tar (``task_binary``); this
unpacks each into a staging tree and hands off to the shared ``harbor.convert``.
Output matches ``harbor.convert`` (``<name>.jsonl`` + ``tasks/<instance_id>/``).
These tasks carry no ``[task].name``, so ``instance_id`` falls back to
``<dataset>__<path>``.

These tasks deposit their deliverable in ``/output`` (e.g. ``cp ...
/output/command_capture.txt``), but harbor only provisions the workdir +
``/logs/{agent,verifier,artifacts}``. Rather than special-case ``/output`` in
the generic env, we remap it onto harbor's ``/logs/artifacts`` here so the data
is fully ``/logs``-native (the tasks already write their reward to
``/logs/verifier``); see ``conform_output_paths``.

    python -m agentic_rl.environment.convert2slime.openthoughts_agent \
        --out-dir data/openthoughts_agent
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import re
import tarfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from agentic_rl.environment.convert2slime import harbor

logger = logging.getLogger(__name__)

HF_DATASET = "open-thoughts/OpenThoughts-Agent-v1-RL"
DATASET_NAME = "openthoughts_agent"  # instance_id prefix + JSONL stem

# These tasks hardcode a /output capture dir; harbor only provisions the workdir
# + /logs/{agent,verifier,artifacts}. Remap /output onto harbor's artifacts dir.
_HARBOR_ARTIFACTS_DIR = "/logs/artifacts"
# Match the /output dir prefix only -- not lookalikes like /outputs or /output_dir.
_OUTPUT_DIR_RE = re.compile(r"/output(?![A-Za-z0-9_])")
# Runtime + agent-facing files only; never touch test data (e.g. the .txt holding
# tests/expected_output.txt, which could legitimately contain the literal /output).
_REMAP_SUFFIXES = (".sh", ".md", ".py")


def _archive_bytes(task_binary: bytes | str) -> bytes:
    """The ``binary`` HF feature loads as bytes; JSON transports base64."""
    if isinstance(task_binary, str):
        return base64.b64decode(task_binary)
    return bytes(task_binary)


def unpack_tasks(rows: Iterable[dict[str, Any]], staging_dir: Path) -> list[Path]:
    """Unpack each row's ``task_binary`` into ``staging_dir/<path>/``; returns
    the sorted task dirs for ``harbor.convert``."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    task_dirs: list[Path] = []
    for row in rows:
        path = row.get("path")
        if not path:
            logger.warning("[openthoughts2slime] skip row with no path")
            continue
        dest = staging_dir / str(path)
        with tarfile.open(fileobj=io.BytesIO(_archive_bytes(row["task_binary"])), mode="r:gz") as tf:
            # filter="data" blocks path traversal/unsafe members (py3.12+).
            tf.extractall(dest, filter="data")
        task_dirs.append(dest)
    return sorted(task_dirs)


def conform_output_paths(task_dirs: Iterable[Path]) -> int:
    """Remap the legacy ``/output`` capture dir onto harbor's ``/logs/artifacts``
    in each task's scripts + instructions (solve.sh, test.sh, instruction.md).

    Rewrites in-place under the staging tree before ``harbor.convert`` copies it.
    Returns the count of files changed. Only ``_REMAP_SUFFIXES`` are touched, so
    test data (``tests/expected_output.txt``) is never rewritten.
    """
    rewritten = 0
    for task_dir in task_dirs:
        for path in task_dir.rglob("*"):
            if not path.is_file() or path.suffix not in _REMAP_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            new = _OUTPUT_DIR_RE.sub(_HARBOR_ARTIFACTS_DIR, text)
            if new != text:
                path.write_text(new, encoding="utf-8")
                rewritten += 1
    return rewritten


def load_hf_rows(repo: str, split: str, limit: int | None) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install `datasets` to pull OpenThoughts-Agent from HuggingFace.") from exc

    for index, row in enumerate(load_dataset(repo, split=split)):
        if limit is not None and index >= limit:
            break
        yield dict(row)


def materialize(
    out_dir: Path,
    *,
    name: str = DATASET_NAME,
    repo: str = HF_DATASET,
    split: str = "train",
    limit: int | None = None,
) -> tuple[int, int]:
    """Download the HF dataset, unpack tasks, and run the harbor converter.

    Returns ``(converted, skipped)``. Staging lives under ``out_dir`` (one
    filesystem for the converter's copytree) and is removed after.
    """
    import shutil

    staging = out_dir / "_staging"
    rows = load_hf_rows(repo, split, limit)
    task_dirs = unpack_tasks(rows, staging)
    logger.info("[openthoughts2slime] unpacked %d tasks -> %s", len(task_dirs), staging)
    remapped = conform_output_paths(task_dirs)
    logger.info("[openthoughts2slime] remapped /output -> %s in %d files", _HARBOR_ARTIFACTS_DIR, remapped)
    try:
        # Train-only dataset (eval is usaco / openthoughts_tblite): all rows ->
        # train.jsonl. key=<name> bakes task_path=<name>/tasks/<id>.
        n_train, n_eval, skipped = harbor.convert(task_dirs, out_dir, key=name, dataset=DATASET_NAME)
        return n_train + n_eval, skipped
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="output dir (JSONL + tasks/); use the slime-data volume"
    )
    parser.add_argument("--name", default=DATASET_NAME, help="JSONL filename stem")
    parser.add_argument("--repo", default=HF_DATASET, help="HuggingFace dataset repo id")
    parser.add_argument("--split", default="train", help="HuggingFace split")
    parser.add_argument("--limit", type=int, help="maximum tasks to convert")
    args = parser.parse_args(argv)

    converted, skipped = materialize(args.out_dir, name=args.name, repo=args.repo, split=args.split, limit=args.limit)
    out_dir = args.out_dir.resolve()
    print(f"converted {converted} tasks ({skipped} skipped) -> {out_dir}/train.jsonl + tasks/")
    print(f"next: publish out-dir to HF with path_in_repo={args.name} (see convert2slime/README.md).")
    return 0 if converted else 1


if __name__ == "__main__":
    raise SystemExit(main())
