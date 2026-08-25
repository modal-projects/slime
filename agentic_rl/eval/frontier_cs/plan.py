"""Print, but never execute, Modal commands for Frontier-CS held-out evals."""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

from .protocol import CONFIG_MODULE, DEFAULT_REGISTRY, ArmSpec, EvalProtocol, load_registry


def _shell_assignment(key: str, value: str) -> str:
    return f"{key}={shlex.quote(value)}"


def _environment_prefix(environment: dict[str, str]) -> str:
    return " ".join(_shell_assignment(key, value) for key, value in environment.items())


def build_plan(
    arm: ArmSpec,
    protocol: EvalProtocol,
    *,
    eval_id: str,
    guide: str,
    modal_environment: str,
    wandb_project: str,
) -> dict[str, object]:
    environment = {
        **arm.environment(eval_id),
        "MODAL_ENVIRONMENT": modal_environment,
        "WANDB_PROJECT": wandb_project,
    }
    prefix = _environment_prefix(environment)
    modal = "uv run --no-dev modal run"
    dump = f"/checkpoints/swe_rollout_dumps/frontier_cs/heldout_avg3/{eval_id}/rollout_eval_0.pt"
    summary = str(Path(dump).with_name("summary.json"))
    return {
        "arm": arm.key,
        "label": arm.label,
        "source_run_tag": arm.source_run_tag,
        "checkpoint_path": arm.checkpoint_path,
        "checkpoint_step": arm.checkpoint_step,
        "eval_id": eval_id,
        "guide": guide,
        "protocol": protocol.__dict__,
        "commands": {
            "download_and_validate": f"{prefix} {modal} slime/modal_train.py::download_data",
            "evaluate": f"{prefix} {modal} -d slime/modal_train.py::train",
            "aggregate": f"{prefix} {modal} slime/modal_train.py::post_process_data",
        },
        "dump_path": dump,
        "summary_path": summary,
    }


def _print_shell(plan: dict[str, object]) -> None:
    print(f"# {plan['label']} ({plan['arm']})")
    print(f"# checkpoint: {plan['checkpoint_path']}")
    print(f"# dump:       {plan['dump_path']}")
    commands = plan["commands"]
    assert isinstance(commands, dict)
    print(f"cd {shlex.quote(str(plan['guide']))}")
    print(commands["evaluate"])
    print(commands["aggregate"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="*", help="Arm keys; defaults to every registered arm")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--format", choices=("shell", "json"), default="shell")
    parser.add_argument(
        "--guide",
        default=str(Path.home() / "Documents/Research/async-rl/multinode-training-guide"),
    )
    parser.add_argument("--modal-environment", default="junlin-dev")
    parser.add_argument("--wandb-project", default="Modal")
    parser.add_argument("--stamp", help="Shared UTC launch stamp; defaults to current time")
    args = parser.parse_args()

    protocol, registry = load_registry(args.registry)
    selected = args.arms or list(registry)
    unknown = sorted(set(selected) - set(registry))
    if unknown:
        parser.error(f"unknown arm(s) {unknown}; choose from {sorted(registry)}")
    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    plans = [
        build_plan(
            registry[key],
            protocol,
            eval_id=f"frontier-cs-heldout-avg3-{key}-{stamp}",
            guide=args.guide,
            modal_environment=args.modal_environment,
            wandb_project=args.wandb_project,
        )
        for key in selected
    ]

    if args.format == "json":
        print(json.dumps(plans, indent=2))
        return

    print("# Dry-run only: review and copy commands explicitly; this module never launches Modal.")
    print(f"# Config: {CONFIG_MODULE}")
    print("# Run download_and_validate once before the first eval:")
    first_commands = plans[0]["commands"]
    assert isinstance(first_commands, dict)
    print(first_commands["download_and_validate"])
    print()
    for index, plan in enumerate(plans):
        if index:
            print()
        _print_shell(plan)


if __name__ == "__main__":
    main()
