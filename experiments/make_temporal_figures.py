#!/usr/bin/env python3
"""Create the three manuscript figures from the frozen temporal-locus run.

This is a deterministic post-processing script. It loads only saved JSON/JSONL
artifacts, uses the same task-family bootstrap implementation as the frozen
analysis, and writes figure-specific TSV sidecars for every displayed value.
It does not load a model or run any forward passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image, ImageChops

from analyze_temporal_locus import BOOTSTRAPS, SEED, SURFACE_GROUPS, bootstrap, load_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "runs"
    / "temporal-welfare-locus-full-20260815T193114Z"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "manuscript" / "figures"

RUN_ID = "temporal-welfare-locus-full-20260815T193114Z"
INTERVAL = "long"

# Okabe-Ito-derived, color-vision-accessible palette. Shape and fill also carry
# every categorical distinction used in the forest plots.
BLUE = "#0072B2"
ORANGE = "#D55E00"
SKY = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#E69F00"
MAGENTA = "#CC79A7"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"
DARK = "#1F2937"
GRID = "#E5E7EB"

VECTOR_ORDER = [
    "welfare_raw",
    "welfare_sentiment_residualized",
    "path_residualized",
    "random_orthogonal_00",
    "random_orthogonal_01",
    "random_orthogonal_02",
    "random_orthogonal_03",
]

VECTOR_NAMES = {
    "welfare_raw": "Transferred direction",
    "welfare_sentiment_residualized": "Sentiment-residualized",
    "path_residualized": "Path residualized",
    "random_orthogonal_00": "Orthogonal 1",
    "random_orthogonal_01": "Orthogonal 2",
    "random_orthogonal_02": "Orthogonal 3",
    "random_orthogonal_03": "Orthogonal 4",
}

VECTOR_COLORS = {
    "welfare_raw": BLUE,
    "welfare_sentiment_residualized": ORANGE,
    "path_residualized": GREEN,
    "random_orthogonal_00": GRAY,
    "random_orthogonal_01": GRAY,
    "random_orthogonal_02": GRAY,
    "random_orthogonal_03": GRAY,
}

VECTOR_MARKERS = {
    "welfare_raw": "o",
    "welfare_sentiment_residualized": "s",
    "path_residualized": "D",
    "random_orthogonal_00": "^",
    "random_orthogonal_01": "v",
    "random_orthogonal_02": "<",
    "random_orthogonal_03": ">",
}

# Separate action-boundary validation on the same 64 adaptively
# clean-calibrated majority-rule cases. These integer counts are displayed in
# Figure 2C rather than inferred from the temporal-locus run loaded above.
# Both precision results are row-backed.  The BF16 hosted-log transport wrapped
# the seven long payload lines, but lossless reassembly recovered all 448 rows
# and reproduced the canonical records hash emitted by the job. ``None`` means
# that precision/direction combination was not run.
CHOICE_SWITCH_COUNTS = {
    "target_half": {
        "label": "0.5×",
        "direction": "functional_welfare_raw",
        "coefficient": 0.5,
        "nf4": 8,
        "bf16": 9,
    },
    "target_full": {
        "label": "1×",
        "direction": "functional_welfare_raw",
        "coefficient": 1.0,
        "nf4": 13,
        "bf16": 17,
    },
    "path": {
        "label": "Path",
        "direction": "path_direct",
        "coefficient": 1.1940978759847163,
        "nf4": 19,
        "bf16": 21,
    },
    "orthogonal_01": {
        "label": "Orthog.\n01",
        "direction": "random_orthogonal_01",
        "coefficient": 1.5272141178181842,
        "nf4": 13,
        "bf16": None,
    },
}
CHOICE_SWITCH_DENOMINATOR = 64
CHOICE_SWITCH_SOURCES = {
    "nf4": (
        "experiments/runs/repaired-semantic-choice-validation-v4c-"
        "20260816T174042Z/summary.json"
    ),
    "bf16": (
        "experiments/runs/qwen-bf16-semantic-choice-hf-job-"
        "6a81fdb8c97db76cbdf33362/records.jsonl"
    ),
}
CHOICE_SWITCH_CANONICAL_RECORDS_SHA256 = {
    "bf16": "e3f7f45e115e28f830c66a157b5c86b48fa3c3f233607c1dc871a626d1c3c52f",
}

SURFACE_LABELS = {
    "semantic": "Semantic labels",
    "opaque": "Reversed opaque labels",
    "numeric": "Reversed numeric scale",
}

MAPPING_LABELS = {
    "status_code_k_success": "Status\nK = success",
    "status_code_m_success": "Status\nM = success",
    "persistence_code_k_continue": "Persistence\nK = continue",
    "persistence_code_m_continue": "Persistence\nM = continue",
}

TSV_FIELDS = [
    "panel",
    "row_type",
    "measure",
    "surface_family",
    "surface",
    "direction",
    "direction_label",
    "coefficient",
    "interval",
    "estimate",
    "ci95_low",
    "ci95_high",
    "n_task_families",
    "bootstrap_draws",
    "numerator",
    "denominator",
    "units",
    "ci_method",
    "source",
    "run_id",
    "note",
]


def configure_style() -> None:
    """Use a restrained journal-style layout at the manuscript's final width.

    The figures are authored at approximately the full-width LaTeX display size,
    so the minimum 10.0-point text below remains at least 7.5 points after the
    small inclusion-scale adjustment in ``main.tex``.
    """

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 10.4,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "text.color": "#000000",
            "axes.labelcolor": "#000000",
            "axes.titlecolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "legend.labelcolor": "#000000",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10.4,
        fontweight="bold",
        va="top",
        ha="left",
    )


def finish_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.7, color=DARK)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)


def errorbar_point(
    ax: plt.Axes,
    x: float,
    y: float,
    low: float,
    high: float,
    *,
    color: str,
    marker: str,
    filled: bool = True,
    zorder: int = 3,
    markersize: float = 5.3,
) -> None:
    face = color if filled else "white"
    ax.errorbar(
        x,
        y,
        xerr=np.array([[x - low], [high - x]]),
        fmt=marker,
        markersize=markersize,
        markerfacecolor=face,
        markeredgecolor=color,
        markeredgewidth=1.0,
        color=color,
        ecolor=color,
        elinewidth=1.1,
        capsize=2.5,
        capthick=1.0,
        zorder=zorder,
    )


def clean_number(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".12g")
    return value


def write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty figure data: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in materialized:
            writer.writerow({field: clean_number(row.get(field, "")) for field in TSV_FIELDS})


def data_row(
    *,
    panel: str,
    row_type: str,
    measure: str,
    estimate: float,
    ci95_low: float,
    ci95_high: float,
    n_task_families: int,
    bootstrap_draws: int,
    units: str,
    source: str,
    surface_family: str = "",
    surface: str = "",
    direction: str = "",
    direction_label: str = "",
    coefficient: float | str = "",
    interval: str = INTERVAL,
    numerator: int | str = "",
    denominator: int | str = "",
    ci_method: str = "95% percentile bootstrap clustered on task family",
    run_id: str = RUN_ID,
    note: str = "",
) -> dict[str, Any]:
    return {
        "panel": panel,
        "row_type": row_type,
        "measure": measure,
        "surface_family": surface_family,
        "surface": surface,
        "direction": direction,
        "direction_label": direction_label,
        "coefficient": coefficient,
        "interval": interval,
        "estimate": estimate,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "n_task_families": n_task_families,
        "bootstrap_draws": bootstrap_draws,
        "numerator": numerator,
        "denominator": denominator,
        "units": units,
        "ci_method": ci_method,
        "source": source,
        "run_id": run_id,
        "note": note,
    }


def load_frozen(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    analysis_path = run_dir / "analysis-summary.json"
    summary_path = run_dir / "summary.json"
    records_path = run_dir / "records.jsonl"
    for path in (analysis_path, summary_path, records_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = load_records(records_path)
    if analysis.get("analysis_status") != "complete":
        raise ValueError("analysis-summary.json is not complete")
    if summary.get("status") != "complete":
        raise ValueError("summary.json is not complete")
    if analysis.get("bootstrap_seed") != SEED:
        raise ValueError("bootstrap seed differs from frozen analysis")
    if analysis.get("bootstrap_draws") != BOOTSTRAPS:
        raise ValueError("bootstrap count differs from frozen analysis")
    if summary.get("n_records") != len(records):
        raise ValueError("record count does not match summary.json")
    if summary.get("n_records") != 5_376:
        raise ValueError("unexpected frozen-run record count")
    return analysis, summary, records


def bootstrap_values(task_values: dict[int, list[float]]) -> dict[str, float]:
    rows = [
        {"value": mean(values), "task_index": float(task)}
        for task, values in sorted(task_values.items())
        if values
    ]
    if len(rows) != 16:
        raise ValueError(f"expected 16 task-family rows, got {len(rows)}")
    return bootstrap(rows, lambda sample: mean(row["value"] for row in sample))


def clean_margin_summaries(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    selected = [
        row
        for row in records
        if row["vector"] == "welfare_raw"
        and float(row["factor"]) == 0.5
        and int(row["prior_sign"]) == 0
        and int(row["readout_sign"]) == 0
        and row["surface"] in {"status_semantic", "persistence_semantic"}
    ]
    output: dict[str, dict[str, float]] = {}
    for name, surfaces in {
        "status": {"status_semantic"},
        "persistence": {"persistence_semantic"},
        "pooled": {"status_semantic", "persistence_semantic"},
    }.items():
        by_task: dict[int, list[float]] = {}
        for row in selected:
            if row["surface"] in surfaces:
                by_task.setdefault(int(row["task_index"]), []).append(float(row["endpoint"]))
        output[name] = bootstrap_values(by_task)
    return output


def readout_crossing_summaries(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    selected = [
        row
        for row in records
        if row["vector"] == "welfare_raw"
        and float(row["factor"]) == 0.5
        and int(row["prior_sign"]) == 0
        and int(row["readout_sign"]) in {-1, 1}
        and row["surface_family"] in {"semantic", "opaque"}
    ]
    lookup = {
        (
            row["surface_family"],
            row["interval"],
            int(row["task_index"]),
            row["surface"],
            int(row["readout_sign"]),
        ): float(row["endpoint"])
        for row in selected
    }
    output: dict[str, dict[str, float | int]] = {}
    for family in ("semantic", "opaque"):
        surfaces = SURFACE_GROUPS[family]
        by_task: dict[int, list[float]] = {}
        numerator = 0
        denominator = 0
        for task in range(16):
            for interval in ("short", "long"):
                for surface in surfaces:
                    negative = lookup[(family, interval, task, surface, -1)]
                    positive = lookup[(family, interval, task, surface, 1)]
                    # "Crossed" is a strict sign change. Touching exactly zero is
                    # not counted, matching the completed saturation audit.
                    crossed = float(negative * positive < 0)
                    by_task.setdefault(task, []).append(crossed)
                    numerator += int(crossed)
                    denominator += 1
        result = bootstrap_values(by_task)
        result["numerator"] = numerator
        result["denominator"] = denominator
        output[family] = result
    return output


def mapping_cell_summary(
    records: list[dict[str, Any]],
    *,
    vector: str,
    factor: float,
    surface: str,
) -> dict[str, float]:
    cells = {
        (
            int(row["task_index"]),
            int(row["readout_sign"]),
        ): float(row["endpoint"])
        for row in records
        if row["vector"] == vector
        and float(row["factor"]) == factor
        and row["interval"] == INTERVAL
        and row["surface"] == surface
        and int(row["prior_sign"]) == 0
        and int(row["readout_sign"]) in {-1, 1}
    }
    rows = []
    for task in range(16):
        rows.append(
            {
                "task_index": float(task),
                "value": cells[(task, 1)] - cells[(task, -1)],
            }
        )
    return bootstrap(rows, lambda sample: mean(row["value"] for row in sample))


def add_footer(
    fig: plt.Figure,
    *,
    extra: str = "",
    footer: str = (
        "Lexical-frame means; error bars are 95% percentile bootstrap intervals "
        "(16 frames, 10,000 draws)."
    ),
) -> None:
    lines = []
    if extra:
        lines.append(extra)
    lines.append(footer)
    fig.text(
        0.5,
        -0.075 if not extra else -0.105,
        "\n".join(lines),
        ha="center",
        va="top",
        fontsize=10.0,
        color="#000000",
    )


def save_bundle(fig: plt.Figure, output_dir: Path, stem: str, title: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fixed_date = datetime(2026, 8, 15, tzinfo=timezone.utc)
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        metadata={
            "Title": title,
            "Author": "Neal Krishna",
            "Subject": "Source runs recorded in the corresponding canonical TSV",
            "Creator": "make_temporal_figures.py",
            "CreationDate": fixed_date,
            "ModDate": fixed_date,
        },
    )
    fig.savefig(
        png_path,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        metadata={
            "Title": title,
            "Author": "Neal Krishna",
            "Run": "See corresponding canonical TSV for source runs",
            "Software": "make_temporal_figures.py",
        },
    )
    plt.close(fig)


def make_figure1(
    analysis: dict[str, Any],
    records: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> list[dict[str, Any]]:
    primary = analysis["primary"][INTERVAL]
    clean = clean_margin_summaries(records)
    crossings = readout_crossing_summaries(records)
    if crossings["semantic"]["numerator"] != 0 or crossings["semantic"]["denominator"] != 64:
        raise ValueError("semantic sign-crossing audit no longer matches frozen result")
    if crossings["opaque"]["numerator"] != 1 or crossings["opaque"]["denominator"] != 128:
        raise ValueError("opaque sign-crossing audit no longer matches frozen result")

    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.05, 2.72),
        gridspec_kw={"width_ratios": [1.12, 1.0, 1.05]},
        layout="constrained",
    )
    ax_a, ax_b, ax_c = axes

    # A: earlier-only versus contemporaneous readout effects.
    y_map = {"semantic": 1.0, "opaque": 0.0}
    for family in ("semantic", "opaque"):
        y = y_map[family]
        for timing, offset, color, marker, filled in (
            ("earlier", -0.12, GRAY, "o", False),
            ("readout", 0.12, BLUE, "s", True),
        ):
            item = primary[family][timing]
            errorbar_point(
                ax_a,
                item["estimate"],
                y + offset,
                item["ci95_low"],
                item["ci95_high"],
                color=color,
                marker=marker,
                filled=filled,
            )
            rows.append(
                data_row(
                    panel="1A",
                    row_type="temporal_contrast",
                    measure=f"{timing}_only_logit_margin_shift",
                    surface_family=family,
                    estimate=item["estimate"],
                    ci95_low=item["ci95_low"],
                    ci95_high=item["ci95_high"],
                    n_task_families=item["n_task_families"],
                    bootstrap_draws=item["bootstrap_draws"],
                    units="logits",
                    source="analysis-summary.json",
                    direction="welfare_raw",
                    direction_label=VECTOR_NAMES["welfare_raw"],
                    coefficient=0.5,
                )
            )
    ax_a.axvline(0, color=DARK, linewidth=0.8)
    ax_a.set_yticks([0, 1], ["Opaque codes", "Semantic labels"])
    ax_a.set_ylim(-0.45, 1.45)
    ax_a.set_xlim(-0.35, 4.55)
    ax_a.set_xlabel("Oriented margin shift")
    ax_a.set_title(r"$\bf{(A)}$ Output timing", loc="left", fontsize=9.8)
    ax_a.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color=GRAY,
                markerfacecolor="white",
                markeredgecolor=GRAY,
                linewidth=0,
                label="Earlier only",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color=BLUE,
                markerfacecolor=BLUE,
                linewidth=0,
                label="At readout",
            ),
        ],
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        loc="lower right",
    )
    finish_axis(ax_a)

    # B: same-axis projection at dose and after withdrawal.  A single clean
    # axis preserves the full scale; compact callouts report the near-zero
    # final-token estimates without a crowded inset.
    projection_colors = {"semantic": BLUE, "opaque": ORANGE}
    for family, jitter in (("semantic", -0.05), ("opaque", 0.05)):
        for position, field, measure in (
            (0.0, "earlier_dose_projection", "dose_site_projection_shift"),
            (1.0, "earlier_final_projection", "final_unsteered_projection_shift"),
        ):
            item = primary[family][field]
            ax_b.errorbar(
                position + jitter,
                item["estimate"],
                yerr=np.array(
                    [[item["estimate"] - item["ci95_low"]], [item["ci95_high"] - item["estimate"]]]
                ),
                fmt="o" if family == "semantic" else "s",
                color=projection_colors[family],
                markerfacecolor=projection_colors[family],
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=5.6,
                capsize=2.5,
                elinewidth=1.1,
                zorder=3,
            )
            rows.append(
                data_row(
                    panel="1B",
                    row_type="projection",
                    measure=measure,
                    surface_family=family,
                    estimate=item["estimate"],
                    ci95_low=item["ci95_low"],
                    ci95_high=item["ci95_high"],
                    n_task_families=item["n_task_families"],
                    bootstrap_draws=item["bootstrap_draws"],
                    units="unit-normalized layer-30 projection",
                    source="analysis-summary.json",
                    direction="welfare_raw",
                    direction_label=VECTOR_NAMES["welfare_raw"],
                    coefficient=0.5,
                )
            )
    ax_b.axhline(0, color=DARK, linewidth=0.8)
    ax_b.set_xticks([0, 1], ["Earlier\ndose", "Final\nreadout"])
    ax_b.set_ylabel("Projection shift")
    ax_b.set_xlim(-0.25, 1.28)
    ax_b.set_ylim(-1.25, 24.0)
    ax_b.set_title(r"$\bf{(B)}$ Internal projection", loc="left", fontsize=9.8)
    finish_axis(ax_b, grid_axis="y")
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="o", color=BLUE, linewidth=0, label="Semantic"),
            Line2D([0], [0], marker="s", color=ORANGE, linewidth=0, label="Opaque"),
        ],
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        loc="center right",
        bbox_to_anchor=(1.0, 0.57),
        handletextpad=0.35,
    )
    for family, x, label_y in (
        ("semantic", 0.95, 2.45),
        ("opaque", 1.05, 1.15),
    ):
        item = primary[family]["earlier_final_projection"]
        ax_b.annotate(
            f'{item["estimate"]:.3f}',
            xy=(x, item["estimate"]),
            xytext=(1.18, label_y),
            ha="right",
            va="center",
            fontsize=8.3,
            fontweight="semibold",
            color=projection_colors[family],
            arrowprops={
                "arrowstyle": "-",
                "color": projection_colors[family],
                "linewidth": 0.75,
                "shrinkA": 2,
                "shrinkB": 2,
            },
        )
    ax_b.text(
        1.18,
        3.75,
        "Final-token values",
        ha="right",
        va="center",
        fontsize=7.7,
        color=GRAY,
    )

    # C: baseline saturation and exact sign-crossing audit.
    labels = ["Status", "Persistence"]
    values = [clean["status"]["estimate"], clean["persistence"]["estimate"]]
    lows = [clean["status"]["ci95_low"], clean["persistence"]["ci95_low"]]
    highs = [clean["status"]["ci95_high"], clean["persistence"]["ci95_high"]]
    y = np.array([1, 0])
    ax_c.barh(y, values, height=0.48, color=[SKY, ORANGE], alpha=0.82, zorder=2)
    ax_c.errorbar(
        values,
        y,
        xerr=np.array([np.array(values) - np.array(lows), np.array(highs) - np.array(values)]),
        fmt="none",
        ecolor=DARK,
        elinewidth=1.0,
        capsize=2.5,
        zorder=3,
    )
    ax_c.axvline(
        clean["pooled"]["estimate"],
        color=DARK,
        linestyle="--",
        linewidth=0.9,
    )
    ax_c.set_ylim(-0.48, 1.48)
    ax_c.text(
        clean["pooled"]["estimate"] + 0.8,
        1.34,
        f'Pooled {clean["pooled"]["estimate"]:.2f}',
        ha="left",
        va="center",
        fontsize=8.2,
        color=DARK,
    )
    ax_c.set_yticks(y, labels)
    ax_c.set_xlabel("Clean margin (logits)")
    ax_c.set_title(r"$\bf{(C)}$ Endpoint margins", loc="left", fontsize=9.8)
    finish_axis(ax_c)

    for name, surface in (("status", "status_semantic"), ("persistence", "persistence_semantic"), ("pooled", "semantic_pooled")):
        item = clean[name]
        rows.append(
            data_row(
                panel="1C",
                row_type="clean_margin",
                measure="clean_semantic_margin",
                surface_family="semantic",
                surface=surface,
                estimate=item["estimate"],
                ci95_low=item["ci95_low"],
                ci95_high=item["ci95_high"],
                n_task_families=item["n_task_families"],
                bootstrap_draws=item["bootstrap_draws"],
                units="logits",
                source="records.jsonl",
                direction="welfare_raw",
                direction_label=VECTOR_NAMES["welfare_raw"],
                coefficient=0.0,
                interval="short_and_long",
                note="0/0 baseline: earlier coefficient 0 and readout coefficient 0.",
            )
        )
    for family in ("semantic", "opaque"):
        item = crossings[family]
        rows.append(
            data_row(
                panel="1C",
                row_type="strict_sign_crossing_rate",
                measure="readout_pair_strict_sign_crossing_rate",
                surface_family=family,
                estimate=float(item["estimate"]),
                ci95_low=float(item["ci95_low"]),
                ci95_high=float(item["ci95_high"]),
                n_task_families=int(item["n_task_families"]),
                bootstrap_draws=int(item["bootstrap_draws"]),
                numerator=int(item["numerator"]),
                denominator=int(item["denominator"]),
                units="proportion of task-by-interval-by-surface cells",
                source="records.jsonl",
                direction="welfare_raw",
                direction_label=VECTOR_NAMES["welfare_raw"],
                coefficient=0.5,
                interval="short_and_long",
                note="Strict sign change: negative- and positive-readout endpoints have product below zero.",
            )
        )

    add_footer(fig)
    save_bundle(
        fig,
        output_dir,
        "fig1_temporal_locus",
        "Temporal contrast, same-axis projection, and saturation",
        dpi,
    )
    return rows


def make_figure2(
    analysis: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.05, 2.62),
        gridspec_kw={"width_ratios": [1.00, 1.05, 1.22]},
        layout="constrained",
    )
    ax_a, ax_b, ax_c = axes

    # A: readout dose series.
    factors = [0.25, 0.5, 1.0]
    for family, color, marker in (
        ("semantic", BLUE, "o"),
        ("opaque", ORANGE, "s"),
    ):
        estimates = []
        lows = []
        highs = []
        for factor in factors:
            item = analysis["dose_response_long"][str(factor)][family]["readout"]
            estimates.append(item["estimate"])
            lows.append(item["ci95_low"])
            highs.append(item["ci95_high"])
            rows.append(
                data_row(
                    panel="2A",
                    row_type="dose_response",
                    measure="readout_only_logit_margin_shift",
                    surface_family=family,
                    direction="welfare_raw",
                    direction_label=VECTOR_NAMES["welfare_raw"],
                    coefficient=factor,
                    estimate=item["estimate"],
                    ci95_low=item["ci95_low"],
                    ci95_high=item["ci95_high"],
                    n_task_families=item["n_task_families"],
                    bootstrap_draws=item["bootstrap_draws"],
                    units="logits",
                    source="analysis-summary.json",
                )
            )
        estimates_a = np.array(estimates)
        ax_a.errorbar(
            factors,
            estimates_a,
            yerr=np.array([estimates_a - np.array(lows), np.array(highs) - estimates_a]),
            marker=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=5.5,
            capsize=2.5,
            label="Semantic" if family == "semantic" else "Opaque",
            zorder=3,
        )
    ax_a.set_xticks(factors, ["0.25", "0.50", "1.00"])
    ax_a.set_xlim(0.21, 1.22)
    ax_a.set_xlabel("Coefficient magnitude")
    ax_a.set_ylabel("Readout logit-margin shift")
    ax_a.set_title(r"$\bf{(A)}$ Dose response", loc="left", fontsize=9.8)
    for family, color in (("semantic", BLUE), ("opaque", ORANGE)):
        endpoint = analysis["dose_response_long"]["1.0"][family]["readout"]["estimate"]
        ax_a.text(
            1.035,
            endpoint,
            "Semantic" if family == "semantic" else "Opaque",
            ha="left",
            va="center",
            fontsize=8.0,
            fontweight="semibold",
            color=color,
        )
    finish_axis(ax_a, grid_axis="y")

    # B: reversed-code semantic binding and fixed-token bias.
    decomp = analysis["reversed_encoding_decomposition"][INTERVAL]
    domains = [
        ("opaque_status", "Status"),
        ("opaque_persistence", "Persistence"),
    ]
    for index, (domain, label) in enumerate(domains):
        y = 1 - index
        for component, offset, color, marker, legend_label in (
            ("semantic_binding", 0.12, BLUE, "o", "Mapping-aware component"),
            ("fixed_token_bias", -0.12, ORANGE, "s", "Fixed-token bias"),
        ):
            item = decomp[domain][f"readout_{component}"]
            errorbar_point(
                ax_b,
                item["estimate"],
                y + offset,
                item["ci95_low"],
                item["ci95_high"],
                color=color,
                marker=marker,
                filled=True,
            )
            rows.append(
                data_row(
                    panel="2B",
                    row_type="reversed_code_decomposition",
                    measure=f"readout_{component}",
                    surface_family="opaque",
                    surface=domain,
                    direction="welfare_raw",
                    direction_label=VECTOR_NAMES["welfare_raw"],
                    coefficient=0.5,
                    estimate=item["estimate"],
                    ci95_low=item["ci95_low"],
                    ci95_high=item["ci95_high"],
                    n_task_families=item["n_task_families"],
                    bootstrap_draws=item["bootstrap_draws"],
                    units="logits",
                    source="analysis-summary.json",
                )
            )
    ax_b.axvline(0, color=DARK, linewidth=0.8)
    ax_b.set_yticks([0, 1], ["Persistence", "Status"])
    ax_b.set_ylim(-0.45, 1.45)
    ax_b.set_xlabel("Decomposed readout shift (logits)")
    ax_b.set_title(r"$\bf{(B)}$ Code components", loc="left", fontsize=9.8)
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="o", color=BLUE, linewidth=0, label="Mapping-aware"),
            Line2D([0], [0], marker="s", color=ORANGE, linewidth=0, label="Token bias"),
        ],
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        loc="center right",
        bbox_to_anchor=(1.0, 0.50),
        fontsize=9.0,
    )
    finish_axis(ax_b)

    # C: unrestricted semantic winner switching in the separate calibrated
    # action-boundary assay.  Exact integer counts are plotted without a CI;
    # the comparison is descriptive and the matched controls are part of the
    # construct-specificity result.
    switch_keys = list(CHOICE_SWITCH_COUNTS)
    x_positions = np.arange(len(switch_keys), dtype=float)
    bar_width = 0.34
    precision_styles = (
        ("nf4", "NF4", BLUE, -bar_width / 2),
        ("bf16", "BF16", ORANGE, bar_width / 2),
    )
    for precision, legend_label, color, offset in precision_styles:
        for x_position, key in zip(x_positions, switch_keys):
            item = CHOICE_SWITCH_COUNTS[key]
            numerator = item[precision]
            if numerator is None:
                continue
            rate = numerator / CHOICE_SWITCH_DENOMINATOR
            direction_label = {
                "functional_welfare_raw": "Target",
                "path_direct": "Path",
                "random_orthogonal_01": "Orthogonal 01",
            }[item["direction"]]
            rows.append(
                data_row(
                    panel="2C",
                    row_type="unrestricted_winner_switching",
                    measure="minus_to_plus_unrestricted_winner_switch_rate",
                    surface_family="semantic",
                    surface="majority_rule_continue_stop",
                    direction=item["direction"],
                    direction_label=direction_label,
                    coefficient=item["coefficient"],
                    interval="active_answer_boundary",
                    estimate=rate,
                    ci95_low=float("nan"),
                    ci95_high=float("nan"),
                    n_task_families=8,
                    bootstrap_draws=0,
                    numerator=numerator,
                    denominator=CHOICE_SWITCH_DENOMINATOR,
                    units="fraction of cases",
                    ci_method="not shown; exact count over the fixed 64-case cohort",
                    source=CHOICE_SWITCH_SOURCES[precision],
                    run_id=(
                        "repaired-semantic-choice-validation-v4c-20260816T174042Z"
                        if precision == "nf4"
                        else "huggingface-job-6a81fdb8c97db76cbdf33362"
                    ),
                    note=(
                        (
                            "BF16 precision; row-backed by all 448 records losslessly "
                            "reassembled from wrapped hosted-log transport; canonical "
                            "records SHA-256 "
                            f"{CHOICE_SWITCH_CANONICAL_RECORDS_SHA256['bf16']}; "
                        )
                        if precision == "bf16"
                        else "NF4 precision; row-backed V4-C records; "
                    )
                    + (
                        "all cases have STOP ground truth. Switching is causal "
                        "actuation, not evidence of generally beneficial or rational "
                        "improvement; n_task_families=8 is the legacy-schema field for "
                        "eight permutation clusters, not lexical families."
                    ),
                )
            )
            bar = ax_c.bar(
                x_position + offset,
                rate,
                width=bar_width * 0.92,
                color=color,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.7,
                label=legend_label if x_position == 0 else None,
                zorder=3,
            )
            ax_c.text(
                bar[0].get_x() + bar[0].get_width() / 2,
                rate + 0.012,
                str(numerator),
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="semibold",
                color=DARK,
            )
    ax_c.set_xticks(
        x_positions,
        [CHOICE_SWITCH_COUNTS[key]["label"] for key in switch_keys],
    )
    ax_c.set_xlim(-0.45, len(switch_keys) - 0.55)
    # Reserve headroom so the top tick does not collide with the panel label.
    ax_c.set_ylim(0, 0.42)
    ax_c.set_yticks(
        [0, 0.1, 0.2, 0.3, 0.4],
        ["0", "10", "20", "30", "40"],
    )
    ax_c.set_xlabel("Target dose or control")
    ax_c.set_ylabel("Switch rate (%)")
    ax_c.set_title(r"$\bf{(C)}$ Answer switching", loc="left", fontsize=9.8)
    ax_c.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        loc="upper left",
        ncols=1,
        columnspacing=0.8,
        handletextpad=0.35,
        fontsize=8.5,
    )
    finish_axis(ax_c, grid_axis="y")

    add_footer(
        fig,
        footer=(
            "Panels A–B: lexical-frame means and 95% bootstrap intervals; "
            "panel C: exact 64-case rates (no interval)."
        ),
    )
    save_bundle(
        fig,
        output_dir,
        "fig2_dose_recoding",
        "Dose ordering, semantic recoding, and semantic winner switching",
        dpi,
    )
    return rows


def make_figure3(
    analysis: dict[str, Any],
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    factors = {name: float(value) for name, value in summary["vector_factors"].items()}
    control_data = analysis["kl_matched_controls_long"]
    mapping_surfaces = list(MAPPING_LABELS)

    fig = plt.figure(figsize=(7.05, 4.75), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.02, 1.42])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(grid[1, :])
    y_positions = np.arange(len(VECTOR_ORDER))[::-1]

    # A and B: aggregate semantic and opaque readout shifts.
    for ax, family, panel in ((ax_a, "semantic", "3A"), (ax_b, "opaque", "3B")):
        for y, vector in zip(y_positions, VECTOR_ORDER):
            item = control_data[vector]["groups"][family]["readout"]
            errorbar_point(
                ax,
                item["estimate"],
                float(y),
                item["ci95_low"],
                item["ci95_high"],
                color=VECTOR_COLORS[vector],
                marker=VECTOR_MARKERS[vector],
                filled=vector != "random_orthogonal_01",
                markersize=5.1,
            )
            rows.append(
                data_row(
                    panel=panel,
                    row_type="aggregate_control",
                    measure="readout_only_logit_margin_shift",
                    surface_family=family,
                    direction=vector,
                    direction_label=VECTOR_NAMES[vector],
                    coefficient=factors[vector],
                    estimate=item["estimate"],
                    ci95_low=item["ci95_low"],
                    ci95_high=item["ci95_high"],
                    n_task_families=item["n_task_families"],
                    bootstrap_draws=item["bootstrap_draws"],
                    units="logits",
                    source="analysis-summary.json",
                )
            )
        ax.axvline(0, color=DARK, linewidth=0.8)
        ax.set_ylim(-0.65, len(VECTOR_ORDER) - 0.35)
        ax.set_xlabel("Readout shift (logits)")
        ax.set_title(
            "Semantic aggregate" if family == "semantic" else "Opaque aggregate",
            loc="left",
        )
        panel_label(ax, "A" if panel == "3A" else "B")
        finish_axis(ax)

    y_labels = [
        f"{VECTOR_NAMES[vector]}  ({factors[vector]:.2f})" for vector in VECTOR_ORDER
    ]
    ax_a.set_yticks(y_positions, y_labels)
    ax_b.tick_params(axis="y", labelleft=False, left=False)
    ax_a.set_xlim(-0.75, 4.55)
    ax_b.set_xlim(-0.22, 2.58)

    # C: all mapping cells, not only their aggregate. Every tile prints its
    # estimate, and the exact interval is retained in the TSV.
    matrix = np.empty((len(VECTOR_ORDER), len(mapping_surfaces)), dtype=float)
    for row_index, vector in enumerate(VECTOR_ORDER):
        for col_index, surface in enumerate(mapping_surfaces):
            item = mapping_cell_summary(
                records,
                vector=vector,
                factor=factors[vector],
                surface=surface,
            )
            matrix[row_index, col_index] = item["estimate"]
            rows.append(
                data_row(
                    panel="3C",
                    row_type="opaque_mapping_cell",
                    measure="readout_only_logit_margin_shift",
                    surface_family="opaque",
                    surface=surface,
                    direction=vector,
                    direction_label=VECTOR_NAMES[vector],
                    coefficient=factors[vector],
                    estimate=item["estimate"],
                    ci95_low=item["ci95_low"],
                    ci95_high=item["ci95_high"],
                    n_task_families=item["n_task_families"],
                    bootstrap_draws=item["bootstrap_draws"],
                    units="logits",
                    source="records.jsonl",
                )
            )
        aggregate = control_data[vector]["groups"]["opaque"]["readout"]["estimate"]
        if not math.isclose(float(matrix[row_index].mean()), float(aggregate), abs_tol=1e-12):
            raise ValueError(f"mapping-cell aggregate mismatch for {vector}")

    vmax = float(np.max(np.abs(matrix)))
    image = ax_c.imshow(
        matrix,
        cmap="BrBG",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    ax_c.set_xticks(
        np.arange(len(mapping_surfaces)),
        [MAPPING_LABELS[surface] for surface in mapping_surfaces],
    )
    ax_c.set_yticks(np.arange(len(VECTOR_ORDER)), [VECTOR_NAMES[v] for v in VECTOR_ORDER])
    ax_c.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax_c.tick_params(axis="y", length=0)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax_c.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10.0,
                color="#000000",
                bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    ax_c.set_title("Opaque mapping cells", loc="left", pad=31)
    panel_label(ax_c, "C")
    colorbar = fig.colorbar(image, ax=ax_c, fraction=0.045, pad=0.03)
    colorbar.set_label("Readout shift (logits)")
    colorbar.ax.tick_params(labelsize=10.0)

    add_footer(
        fig,
        extra=(
            "Coefficient magnitudes are shown in parentheses;\n"
            "they were selected on a disjoint full-vocabulary divergence calibration, not on these endpoints."
        ),
    )
    save_bundle(
        fig,
        output_dir,
        "fig3_controls",
        "Raw direction and divergence-calibrated semantic and opaque controls",
        dpi,
    )
    return rows


def validate_outputs(output_dir: Path, dpi: int) -> dict[str, Any]:
    manifest: dict[str, Any] = {"output_dir": str(output_dir), "files": {}}
    for stem in ("fig1_temporal_locus", "fig2_dose_recoding", "fig3_controls"):
        pdf_path = output_dir / f"{stem}.pdf"
        png_path = output_dir / f"{stem}.png"
        tsv_path = output_dir / f"{stem}_data.tsv"
        for path in (pdf_path, png_path, tsv_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty output: {path}")
        if pdf_path.read_bytes()[:4] != b"%PDF":
            raise ValueError(f"invalid PDF header: {pdf_path}")
        with Image.open(png_path) as image:
            width, height = image.size
            # Figures are authored at their approximately 7-inch manuscript
            # display width; reject accidental thumbnail-sized outputs without
            # requiring the earlier oversized pre-layout canvas.
            if width < int(dpi * 6.8) or height < int(dpi * 2.2):
                raise ValueError(f"unexpectedly small PNG: {png_path} is {width}x{height}")
            rgb = image.convert("RGB")
            difference = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white"))
            if difference.getbbox() is None:
                raise ValueError(f"blank PNG: {png_path}")
        with tsv_path.open(encoding="utf-8", newline="") as handle:
            tsv_rows = list(csv.DictReader(handle, delimiter="\t"))
        if not tsv_rows:
            raise ValueError(f"empty TSV: {tsv_path}")
        manifest["files"][stem] = {
            "png_pixels": [width, height],
            "png_bytes": png_path.stat().st_size,
            "pdf_bytes": pdf_path.stat().st_size,
            "tsv_rows": len(tsv_rows),
            "tsv_bytes": tsv_path.stat().st_size,
        }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 200:
        raise ValueError("publication PNGs require at least 200 dpi")

    configure_style()
    analysis, summary, records = load_frozen(args.run_dir)
    figure1_rows = make_figure1(analysis, records, args.output_dir, args.dpi)
    write_tsv(args.output_dir / "fig1_temporal_locus_data.tsv", figure1_rows)
    figure2_rows = make_figure2(analysis, args.output_dir, args.dpi)
    write_tsv(args.output_dir / "fig2_dose_recoding_data.tsv", figure2_rows)
    figure3_rows = make_figure3(
        analysis,
        summary,
        records,
        args.output_dir,
        args.dpi,
    )
    write_tsv(args.output_dir / "fig3_controls_data.tsv", figure3_rows)
    manifest = validate_outputs(args.output_dir, args.dpi)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
