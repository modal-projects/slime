"""Translate SWE-Gym / SWE-Gym-Lite rows into slime prompt data.

Schema pair of ``env/swe_gym.py`` (SWE rows carry no ``metadata.task_type``:
the default env). Output is one JSON object per line:

    {
      "prompt": "...",
      "label": "owner__repo-123",
      "metadata": {
        "instance_id": "...",
        "image": "...",
        "workdir": "/testbed",
        "problem_statement": "...",
        "eval_cmd": "echo ... | base64 -d > /tmp/swegym_eval.py && python /tmp/swegym_eval.py",
        "pre_commands": ["git checkout <base_commit> -f"]
      }
    }

SWE-Gym-specific: derive the prebuilt image name and build a pytest reward
command from ``test_patch`` + F2P/P2P tests.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any

HF_DATASET = "SWE-Gym/SWE-Gym"
HF_DATASET_LITE = "SWE-Gym/SWE-Gym-Lite"
IMAGE_PREFIX = os.environ.get("SWE_GYM_IMAGE_PREFIX", "docker.io/xingyaoww")
IMAGE_TAG = os.environ.get("SWE_GYM_IMAGE_TAG", "latest")
WORKDIR = "/testbed"


def image_for(instance_id: str) -> str:
    name = "sweb.eval.x86_64." + instance_id.replace("__", "_s_")
    return f"{IMAGE_PREFIX.rstrip('/')}/{name}:{IMAGE_TAG}"


def as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else []
    return [str(item) for item in (value or [])]


def build_eval_cmd(test_patch: str, tests: list[str]) -> str:
    patch_b64 = base64.b64encode((test_patch or "").encode()).decode("ascii")
    script = "\n".join(
        [
            "import base64",
            "import pathlib",
            "import subprocess",
            "import sys",
            "",
            "PATCH_B64 = " + repr(patch_b64),
            "TESTS = " + json.dumps(tests),
            "patch_path = pathlib.Path('/tmp/swegym_test.patch')",
            "patch_path.write_bytes(base64.b64decode(PATCH_B64))",
            "if patch_path.stat().st_size:",
            "    commands = [",
            "        ['git', 'apply', '-v', str(patch_path)],",
            "        ['git', 'apply', '--3way', str(patch_path)],",
            "        ['patch', '-p1', '--no-backup-if-mismatch', '-i', str(patch_path)],",
            "    ]",
            "    for command in commands:",
            "        result = subprocess.run(command)",
            "        if result.returncode == 0:",
            "            break",
            "    else:",
            "        sys.exit(result.returncode)",
            "",
            "import pytest",
            "sys.exit(pytest.main(['--no-header', '-rN', '-p', 'no:cacheprovider', *TESTS]))",
            "",
        ]
    )
    script_b64 = base64.b64encode(script.encode()).decode("ascii")
    return f"echo {script_b64} | base64 -d > /tmp/swegym_eval.py && python /tmp/swegym_eval.py"


def translate(raw: dict[str, Any]) -> dict[str, Any] | None:
    instance_id = raw.get("instance_id")
    if not instance_id:
        return None

    tests = as_list(raw.get("FAIL_TO_PASS")) + as_list(raw.get("PASS_TO_PASS"))
    if not tests:
        return None

    problem = raw.get("problem_statement") or ""
    metadata: dict[str, Any] = {
        "instance_id": instance_id,
        "image": image_for(instance_id),
        "workdir": WORKDIR,
        "problem_statement": problem,
        "eval_cmd": build_eval_cmd(raw.get("test_patch") or "", tests),
        "repo": raw.get("repo"),
        "base_commit": raw.get("base_commit"),
        "version": raw.get("version"),
    }
    if base_commit := raw.get("base_commit"):
        metadata["pre_commands"] = [f"git checkout {base_commit} -f"]

    # prompt as a single-message list, NOT a raw string: slime's Dataset asserts
    # a list when a HF processor loads for hf_checkpoint. Harmless: the agent
    # reads problem_statement from metadata, never sample.prompt.
    return {
        "prompt": [{"role": "user", "content": problem}],
        "label": instance_id,
        "metadata": metadata,
    }


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_hf(split: str, *, lite: bool, limit: int | None) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install `datasets` to pull SWE-Gym from HuggingFace.") from exc

    dataset = HF_DATASET_LITE if lite else HF_DATASET
    for index, row in enumerate(load_dataset(dataset, split=split)):
        if limit is not None and index >= limit:
            break
        yield dict(row)


def write_jsonl(rows: Iterable[dict[str, Any]], out_path: str | Path) -> int:
    count = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for raw in rows:
            row = translate(raw)
            if row is None:
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate SWE-Gym data to slime prompt JSONL.")
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument("--input", help="downloaded SWE-Gym JSONL to convert")
    parser.add_argument("--split", default="train", help="HuggingFace split when --input is omitted")
    parser.add_argument("--lite", action="store_true", help="use SWE-Gym-Lite when --input is omitted")
    parser.add_argument("--limit", type=int, help="maximum rows to read")
    args = parser.parse_args(argv)

    if args.input:
        raw_rows: Iterable[dict[str, Any]] = iter_jsonl(args.input)
        if args.limit is not None:
            raw_rows = islice(raw_rows, args.limit)
    else:
        raw_rows = load_hf(args.split, lite=args.lite, limit=args.limit)
    count = write_jsonl(raw_rows, args.out)
    print(f"wrote {count} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
