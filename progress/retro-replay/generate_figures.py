"""Regenerate Frontier-CS retro-replay progress figures from W&B."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import wandb
from matplotlib.lines import Line2D

PROJECT = "junlinwang/Modal"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
METRICS = {
    "final": "agentic/outcome/reward_final/mean",
    "submissions": "agentic/submissions/mean",
}


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    legend: str
    prefix: str
    color: str
    exclude: tuple[str, ...] = ()


O0 = Arm(
    "o0",
    "Baseline O0",
    "Baseline O0 — final reward",
    "qwen3.6-27b-frontier-cs-noncolocate-5n-baseline-20260710-125341",
    "#808080",
)
O1 = Arm(
    "o1",
    "O1 best",
    "O1 best — max(final, best)",
    "qwen3.6-27b-frontier-cs-o1-best-20260714-225603",
    "#2878b5",
)
O2 = Arm(
    "o2",
    "O2 bonus",
    "O2 bonus — final + 0.3 solved",
    "qwen3.6-27b-frontier-cs-o2-bonus-20260714-235055",
    "#d62728",
)
RETRO_A = Arm(
    "retro_a",
    "Retro A final",
    "Retro A final — 25% retro, final",
    "qwen3.6-27b-frontier-cs-retro-a-final-",
    "#d48a1f",
    ("-p25", "-p50", "-p75"),
)
RETRO_B = Arm(
    "retro_b",
    "Retro B best",
    "Retro B best — 25% retro, best",
    "qwen3.6-27b-frontier-cs-retro-b-best-",
    "#2f8f5b",
)
RETRO_A_REFERENCE = Arm(
    "retro_a",
    "Retro A final",
    "Retro A final — original arm",
    RETRO_A.prefix,
    "#606060",
    RETRO_A.exclude,
)
P25 = Arm(
    "p25",
    "P25",
    "P25 — branch at 25% of trajectory",
    "qwen3.6-27b-frontier-cs-retro-a-final-p25",
    "#2878b5",
)
P50 = Arm(
    "p50",
    "P50",
    "P50 — branch at 50% of trajectory",
    "qwen3.6-27b-frontier-cs-retro-a-final-p50",
    "#d48a1f",
)
P75 = Arm(
    "p75",
    "P75",
    "P75 — branch at 75% of trajectory",
    "qwen3.6-27b-frontier-cs-retro-a-final-p75",
    "#2f8f5b",
)


def load_lineage(api: wandb.Api, arm: Arm) -> dict[str, dict[int, float]]:
    """Merge one logical experiment's retries, preferring later overlaps."""
    curves = {key: {} for key in METRICS}
    runs = list(
        api.runs(
            PROJECT,
            filters={"display_name": {"$regex": f"^{arm.prefix}"}},
            order="+created_at",
            per_page=100,
        )
    )
    runs = [
        run
        for run in runs
        if not any(marker in (run.name or "") for marker in arm.exclude)
    ]
    history_keys = ["rollout/step", *METRICS.values()]
    used_runs = 0
    for run in runs:
        saw_value = False
        for row in run.scan_history(keys=history_keys, page_size=1000):
            step = row.get("rollout/step")
            if not isinstance(step, (int, float)):
                continue
            for key, metric in METRICS.items():
                value = row.get(metric)
                if isinstance(value, (int, float)):
                    curves[key][int(step)] = float(value)
                    saw_value = True
        used_runs += int(saw_value)

    if not all(curves.values()):
        raise RuntimeError(f"no complete metric lineage found for {arm.label}")
    coverage = ", ".join(
        f"{key}=0–{max(values)}" for key, values in curves.items()
    )
    print(f"{arm.label}: {used_runs}/{len(runs)} runs with data; {coverage}")
    return curves


def trailing_curve(
    curve: dict[int, float],
    *,
    window: int = 7,
    sample_every: int | None = None,
    cap: int | None = None,
) -> tuple[list[int], list[float]]:
    steps = [step for step in sorted(curve) if cap is None or step <= cap]
    smoothed: dict[int, float] = {}
    for index, step in enumerate(steps):
        start = max(0, index - window + 1)
        smoothed[step] = (
            sum(curve[item] for item in steps[start : index + 1])
            / (index - start + 1)
        )
    if sample_every is not None:
        sampled = [step for step in steps if step % sample_every == 0]
        if steps[-1] not in sampled:
            sampled.append(steps[-1])
        steps = sampled
    return steps, [smoothed[step] for step in steps]


def reward_axis_limit(
    data: dict[str, dict[str, dict[int, float]]],
    arms: list[Arm],
    *,
    cap: int | None,
    sample_every: int | None,
) -> float:
    maximum = max(
        max(
            trailing_curve(
                data[arm.key]["final"],
                sample_every=sample_every,
                cap=cap,
            )[1]
        )
        for arm in arms
    )
    return max(0.30, math.ceil((maximum + 0.01) / 0.05) * 0.05)


def legend_handles(arms: list[Arm], order: list[int] | None) -> list[Line2D]:
    ordered = arms if order is None else [arms[index] for index in order]
    return [
        Line2D([0], [0], color=arm.color, linewidth=2.5, label=arm.legend)
        for arm in ordered
    ]


def plot_stacked(
    filename: str,
    data: dict[str, dict[str, dict[int, float]]],
    arms: list[Arm],
    *,
    sample_every: int | None,
    cap: int | None,
    submissions_max: float,
    legend_columns: int,
    legend_order: list[int] | None = None,
) -> None:
    reward_max = reward_axis_limit(
        data,
        arms,
        cap=cap,
        sample_every=sample_every,
    )
    reward_ticks = []
    tick = 0.08
    while tick < reward_max:
        reward_ticks.append(round(tick, 2))
        tick += 0.05

    max_step = max(
        min(max(data[arm.key]["final"]), cap)
        if cap is not None
        else max(data[arm.key]["final"])
        for arm in arms
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#555555",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 7.2), dpi=180)
    bottom = 0.18 if legend_columns < len(arms) else 0.15
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.94,
        bottom=bottom,
        hspace=0.34,
    )

    panels = [
        ("final", "Final reward", (0.08, reward_max), reward_ticks),
        (
            "submissions",
            "Submissions / episode",
            (0, submissions_max),
            list(range(0, int(submissions_max) + 1, 4)),
        ),
    ]
    for axis, (metric, title, limits, ticks) in zip(axes, panels, strict=True):
        for arm in arms:
            x, y = trailing_curve(
                data[arm.key][metric],
                sample_every=sample_every,
                cap=cap,
            )
            axis.plot(x, y, color=arm.color, linewidth=2.1)
        axis.set_title(title, fontsize=13, pad=8, loc="left")
        axis.set_xlim(0, max_step)
        axis.set_ylim(*limits)
        x_ticks = [step for step in (0, 20, 40, 60, 80) if step <= max_step]
        if max_step not in x_ticks:
            x_ticks.append(max_step)
        axis.set_xticks(x_ticks)
        axis.set_yticks(ticks)
        axis.grid(True, color="#dedede", linewidth=0.8, alpha=0.9)
        axis.tick_params(labelsize=9, colors="#444444")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("optimizer step", fontsize=9)

    axes[1].axhline(10, color="#666666", linewidth=1.2, linestyle=(0, (3, 3)))
    axes[1].text(
        max_step - 0.5,
        10.25,
        "submission watchdog",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )
    fig.legend(
        handles=legend_handles(arms, legend_order),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=legend_columns,
        frameon=False,
        columnspacing=2.0,
        fontsize=10,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{filename}.png"
    svg = png.with_suffix(".svg")
    fig.savefig(png, facecolor="white")
    fig.savefig(svg, facecolor="white")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)
    print(png)


def main() -> None:
    api = wandb.Api(timeout=90)
    all_arms = [O0, O1, O2, RETRO_A, RETRO_B, P25, P50, P75]
    data = {arm.key: load_lineage(api, arm) for arm in all_arms}

    plot_stacked(
        "outcome-reward-ablation-final-submissions",
        data,
        [O0, O1, O2],
        sample_every=None,
        cap=None,
        submissions_max=14,
        legend_columns=3,
    )
    plot_stacked(
        "five-arm-final-submissions",
        data,
        [O0, O1, O2, RETRO_A, RETRO_B],
        sample_every=5,
        cap=84,
        submissions_max=20,
        legend_columns=3,
        legend_order=[0, 3, 1, 4, 2],
    )
    plot_stacked(
        "position-ablation-final-submissions",
        data,
        [P25, P50, P75, RETRO_A_REFERENCE],
        sample_every=5,
        cap=84,
        submissions_max=20,
        legend_columns=2,
        legend_order=[0, 2, 1, 3],
    )


if __name__ == "__main__":
    main()
