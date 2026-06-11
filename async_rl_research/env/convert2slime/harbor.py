"""Materialize a harbor dataset as slime prompt data + local task dirs.

Schema pair of ``env/harbor.py`` (rows carry ``metadata.task_type:
"harbor"``). This converter is the ONLY writer of that schema: it parses each
task's ``task.toml`` offline (plain ``tomllib`` -- the ``harbor`` package is
needed only for ``--registry`` downloads) and bakes everything the rollout
needs into ``metadata``, so the rollout runtime never reads harbor config.

Output layout (put ``--out-dir`` on the slime-data volume)::

    <out-dir>/<name>.jsonl          slime prompt data
    <out-dir>/tasks/<instance_id>/  copied harbor task dirs (environment/ for
                                    Dockerfile builds, tests/ + steps/ for
                                    verification, solution/ for oracle checks)

Rows reference tasks via ``metadata.task_path`` RELATIVE to the out dir;
export ``ASYNC_RL_TASK_ROOT=<out-dir>`` for the rollout / oracle.

Sources::

    # a directory whose subdirectories are harbor tasks (e.g. a
    # harbor-datasets checkout subtree, or an adapter's generated tasks)
    python -m async_rl_research.env.convert2slime.harbor \
        --tasks-dir ~/harbor-datasets/datasets/usaco --out-dir data/usaco

    # straight from a harbor registry (requires `pip install harbor`)
    python -m async_rl_research.env.convert2slime.harbor \
        --registry ~/harbor/registry.json --dataset usaco --out-dir data/usaco

v1 scope (anything else is skipped + logged): linux, single-container
Dockerfile or prebuilt docker_image, shared-environment verification, no
GPU/TPU, no MCP servers. Multi-step tasks ARE supported (per-step
instruction/tests/min_reward; see env/harbor.py for reward aggregation).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_COPY_IGNORE = shutil.ignore_patterns(".git", "__pycache__", ".DS_Store", ".venv", "node_modules")
_WORKDIR_RE = re.compile(r"^\s*WORKDIR\s+(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


class SkipTask(Exception):
    """Task is outside v1 scope; str(err) is the logged reason."""


def _parse_dockerfile_workdir(dockerfile: Path) -> str | None:
    matches = _WORKDIR_RE.findall(dockerfile.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        return None
    last = matches[-1].strip().strip('"').strip("'")
    # Variable or relative WORKDIRs can't be resolved statically; let the
    # rollout detect the cwd from the booted sandbox instead.
    return last if last.startswith("/") and "$" not in last else None


def _check_verifier_shared(verifier_cfg: dict[str, Any], where: str) -> None:
    if verifier_cfg.get("environment_mode") == "separate" or verifier_cfg.get("environment"):
        raise SkipTask(f"separate verifier environment ({where})")


def _verifier_md(verifier_cfg: dict[str, Any]) -> dict[str, Any]:
    md: dict[str, Any] = {}
    if verifier_cfg.get("timeout_sec"):
        md["timeout_sec"] = float(verifier_cfg["timeout_sec"])
    if verifier_cfg.get("env"):
        md["env"] = dict(verifier_cfg["env"])
    return md


def _instruction(path: Path, where: str) -> str:
    if not path.is_file():
        raise SkipTask(f"missing {where}")
    return path.read_text(encoding="utf-8")


def _steps_md(cfg: dict[str, Any], task_dir: Path) -> list[dict[str, Any]]:
    steps_md = []
    for step in cfg.get("steps") or []:
        name = step.get("name")
        if not name:
            raise SkipTask("unnamed step")
        step_dir = task_dir / "steps" / name
        # Harbor falls back to the shared top-level tests/ when a step ships
        # no tests of its own.
        tests_path = f"steps/{name}/tests" if (step_dir / "tests" / "test.sh").is_file() else "tests"
        if not (task_dir / tests_path / "test.sh").is_file():
            raise SkipTask(f"step {name!r} has no tests/test.sh (step or shared)")
        _check_verifier_shared(step.get("verifier") or {}, f"step {name!r}")
        steps_md.append(
            {
                "name": name,
                "instruction": _instruction(step_dir / "instruction.md", f"steps/{name}/instruction.md"),
                "tests_path": tests_path,
                "verifier": _verifier_md(step.get("verifier") or {}),
                "min_reward": step.get("min_reward"),
                "agent_timeout_sec": (step.get("agent") or {}).get("timeout_sec"),
            }
        )
    return steps_md


def translate_task(task_dir: Path, *, dataset: str | None = None) -> dict[str, Any]:
    """One harbor task dir -> one slime row (metadata.task_path filled by caller).

    ``dataset`` qualifies tasks whose task.toml carries no ``[task].name``
    (harbor-datasets tasks are often bare numeric dirs like ``usaco/84``).
    Raises ``SkipTask`` for tasks outside v1 scope.
    """
    config_path = task_dir / "task.toml"
    if not config_path.is_file():
        raise SkipTask("no task.toml")
    cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))

    env_cfg = cfg.get("environment") or {}
    if (env_cfg.get("os") or "linux") != "linux":
        raise SkipTask(f"os={env_cfg.get('os')}")
    if env_cfg.get("gpus") or env_cfg.get("gpu_types") or env_cfg.get("tpu"):
        raise SkipTask("requires GPU/TPU")
    if env_cfg.get("mcp_servers"):
        raise SkipTask("requires MCP servers")
    if env_cfg.get("network_mode") in ("no-network", "allowlist"):
        # The Modal backend doesn't enforce per-phase network policies yet;
        # running these would silently grant more network than the task allows.
        raise SkipTask(f"network_mode={env_cfg['network_mode']}")

    docker_image = env_cfg.get("docker_image")
    dockerfile = None
    workdir = env_cfg.get("workdir")
    if not docker_image:
        if (task_dir / "environment" / "docker-compose.yaml").is_file() or (
            task_dir / "environment" / "docker-compose.yml"
        ).is_file():
            raise SkipTask("docker-compose environment")
        if not (task_dir / "environment" / "Dockerfile").is_file():
            raise SkipTask("no environment/Dockerfile or docker_image")
        dockerfile = "environment/Dockerfile"
        if not workdir:
            workdir = _parse_dockerfile_workdir(task_dir / "environment" / "Dockerfile")

    verifier_cfg = cfg.get("verifier") or {}
    _check_verifier_shared(verifier_cfg, "task")

    steps_md = _steps_md(cfg, task_dir)
    if steps_md:
        instruction_path = task_dir / "instruction.md"
        instruction = instruction_path.read_text(encoding="utf-8") if instruction_path.is_file() else steps_md[0]["instruction"]
    else:
        instruction = _instruction(task_dir / "instruction.md", "instruction.md")
        if not (task_dir / "tests" / "test.sh").is_file():
            raise SkipTask("no tests/test.sh")

    fallback = f"{dataset}/{task_dir.name}" if dataset else task_dir.name
    task_name = ((cfg.get("task") or {}).get("name") or fallback).strip()
    instance_id = re.sub(r"[^A-Za-z0-9_.-]+", "__", task_name) or task_dir.name

    metadata: dict[str, Any] = {
        "task_type": "harbor",
        "instance_id": instance_id,
        "docker_image": docker_image,
        "dockerfile": dockerfile,
        "workdir": workdir,
        "problem_statement": instruction,
        "agent_timeout_sec": (cfg.get("agent") or {}).get("timeout_sec"),
        "verifier": _verifier_md(verifier_cfg),
        "steps": steps_md or None,
        "reward_strategy": cfg.get("multi_step_reward_strategy"),
        "cpus": env_cfg.get("cpus"),
        "memory_mb": env_cfg.get("memory_mb"),
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}

    return {"prompt": instruction, "label": task_name, "metadata": metadata}


def convert(
    task_dirs: list[Path], out_dir: Path, *, name: str, dataset: str | None = None, limit: int | None = None
) -> tuple[int, int]:
    """Copy tasks + write the JSONL. Returns (converted, skipped)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_out = out_dir / "tasks"
    rows: list[dict[str, Any]] = []
    skipped = 0

    for task_dir in task_dirs:
        if limit is not None and len(rows) >= limit:
            break
        try:
            row = translate_task(task_dir, dataset=dataset)
        except SkipTask as e:
            skipped += 1
            logger.warning("[harbor2slime] skip %s: %s", task_dir.name, e)
            continue
        instance_id = row["metadata"]["instance_id"]
        dest = tasks_out / instance_id
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(task_dir, dest, ignore=_COPY_IGNORE)
        row["metadata"]["task_path"] = f"tasks/{instance_id}"
        rows.append(row)

    jsonl_path = out_dir / f"{name}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("[harbor2slime] wrote %d rows -> %s (skipped %d)", len(rows), jsonl_path, skipped)
    return len(rows), skipped


def _discover_task_dirs(tasks_dir: Path) -> list[Path]:
    """Direct subdirectories holding a task.toml, sorted for determinism."""
    if (tasks_dir / "task.toml").is_file():
        return [tasks_dir]
    return sorted(p for p in tasks_dir.iterdir() if (p / "task.toml").is_file())


def _download_from_registry(registry_spec: str, dataset: str, version: str | None, download_dir: Path) -> list[Path]:
    """Fetch a dataset's tasks via the harbor package (optional dependency)."""
    try:
        from harbor.models.registry import Registry
        from harbor.tasks.client import TaskClient
    except ImportError as exc:
        raise SystemExit("--registry mode needs the `harbor` package: pip install harbor") from exc

    import asyncio

    if registry_spec.startswith(("http://", "https://")):
        registry = Registry.from_url(registry_spec)
    else:
        registry = Registry.from_path(Path(registry_spec))
    matches = [d for d in registry.datasets if d.name == dataset and (version is None or d.version == version)]
    if not matches:
        known = sorted({d.name for d in registry.datasets})
        raise SystemExit(f"dataset {dataset!r} not in registry; known: {known}")
    spec = matches[-1]
    task_ids = [t.to_source_task_id() for t in spec.tasks]
    logger.info("[harbor2slime] downloading %d tasks for %s==%s", len(task_ids), spec.name, spec.version)
    result = asyncio.run(TaskClient().download_tasks(task_ids, output_dir=download_dir))

    paths: list[Path] = []
    items = result.values() if hasattr(result, "values") else result
    for item in items:
        path = getattr(item, "path", None) or getattr(item, "downloaded_path", None)
        if path:
            paths.append(Path(path))
    if not paths:
        raise SystemExit(f"registry download produced no task paths (result type {type(result).__name__})")
    return sorted(paths)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Materialize a harbor dataset as slime prompt JSONL + task dirs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tasks-dir", type=Path, help="directory of harbor task dirs (or a single task dir)")
    source.add_argument("--registry", help="harbor registry.json path or URL (needs `pip install harbor`)")
    parser.add_argument("--dataset", help="dataset name in the registry (with --registry)")
    parser.add_argument("--dataset-version", help="dataset version in the registry (default: last match)")
    parser.add_argument("--out-dir", type=Path, required=True, help="output dir (JSONL + tasks/); use the slime-data volume")
    parser.add_argument("--name", help="JSONL filename stem (default: dataset or tasks-dir name)")
    parser.add_argument("--limit", type=int, help="maximum tasks to convert")
    args = parser.parse_args(argv)

    if args.registry:
        if not args.dataset:
            parser.error("--registry requires --dataset")
        task_dirs = _download_from_registry(args.registry, args.dataset, args.dataset_version, args.out_dir / "downloads")
        name = args.name or args.dataset
        dataset = args.dataset
    else:
        task_dirs = _discover_task_dirs(args.tasks_dir)
        name = args.name or args.tasks_dir.name
        dataset = args.tasks_dir.name
    if not task_dirs:
        raise SystemExit("no task dirs found")

    converted, skipped = convert(task_dirs, args.out_dir, name=name, dataset=dataset, limit=args.limit)
    out_dir = args.out_dir.resolve()
    print(f"converted {converted} tasks ({skipped} skipped) -> {out_dir / (name + '.jsonl')}")
    print("next steps:")
    print(f"  export ASYNC_RL_TASK_ROOT={out_dir}")
    print(f"  python -m async_rl_research.env.harbor {out_dir / (name + '.jsonl')} --limit 3   # oracle check")
    return 0 if converted else 1


if __name__ == "__main__":
    raise SystemExit(main())
