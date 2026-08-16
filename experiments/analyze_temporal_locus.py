#!/usr/bin/env python3
"""Family-level analysis for the frozen temporal-locus experiment.

The runner stores every response surface separately.  This script first forms
contrasts within each independent task family, then pairs reversed encodings,
and only then bootstraps task families.  Counterbalances are measurements, not
independent observations.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from collections import defaultdict
from statistics import mean
from typing import Any, Callable


SEED = 26081531
BOOTSTRAPS = 10_000

SURFACE_GROUPS = {
    "semantic": (
        "status_semantic",
        "persistence_semantic",
    ),
    "opaque": (
        "status_code_k_success",
        "status_code_m_success",
        "persistence_code_k_continue",
        "persistence_code_m_continue",
    ),
    "numeric": (
        "rating_9_good",
        "rating_0_good",
    ),
}

REVERSED_PAIRS = {
    "opaque_status": ("status_code_k_success", "status_code_m_success"),
    "opaque_persistence": (
        "persistence_code_k_continue",
        "persistence_code_m_continue",
    ),
    "numeric": ("rating_9_good", "rating_0_good"),
}


def percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def bootstrap(
    rows: list[dict[str, float]],
    statistic: Callable[[list[dict[str, float]]], float],
) -> dict[str, float]:
    rng = random.Random(SEED)
    estimate = statistic(rows)
    draws: list[float] = []
    for _ in range(BOOTSTRAPS):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        value = statistic(sample)
        if math.isfinite(value):
            draws.append(value)
    return {
        "estimate": estimate,
        "ci95_low": percentile(draws, 0.025),
        "ci95_high": percentile(draws, 0.975),
        "n_task_families": len(rows),
        "bootstrap_draws": len(draws),
    }


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cell_lookup(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (
            row["vector"],
            float(row["factor"]),
            row["interval"],
            int(row["task_index"]),
            row["surface"],
            int(row["prior_sign"]),
            int(row["readout_sign"]),
        ): row
        for row in records
    }


def surface_contrasts(
    records: list[dict[str, Any]], vector: str, factor: float, interval: str
) -> dict[int, dict[str, dict[str, float]]]:
    lookup = cell_lookup(records)
    task_ids = sorted(
        {
            int(row["task_index"])
            for row in records
            if row["vector"] == vector
            and float(row["factor"]) == factor
            and row["interval"] == interval
        }
    )
    surfaces = sorted(
        {
            row["surface"]
            for row in records
            if row["vector"] == vector
            and float(row["factor"]) == factor
            and row["interval"] == interval
        }
    )
    answer: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for task in task_ids:
        for surface in surfaces:
            key = (vector, factor, interval, task, surface)
            cells = {
                (p, r): lookup.get((*key, p, r))
                for p, r in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1))
            }
            if any(value is None for value in cells.values()):
                continue
            answer[task][surface] = {
                "earlier": cells[(1, 0)]["endpoint"] - cells[(-1, 0)]["endpoint"],
                "readout": cells[(0, 1)]["endpoint"] - cells[(0, -1)]["endpoint"],
                "congruent": cells[(1, 1)]["endpoint"] - cells[(-1, -1)]["endpoint"],
                "conflict": cells[(-1, 1)]["endpoint"] - cells[(1, -1)]["endpoint"],
                "earlier_final_projection": (
                    cells[(1, 0)]["final_projection"]
                    - cells[(-1, 0)]["final_projection"]
                ),
                "earlier_dose_projection": (
                    cells[(1, 0)]["dose_projection"]
                    - cells[(-1, 0)]["dose_projection"]
                ),
            }
    return answer


def summarize_group(
    contrasts: dict[int, dict[str, dict[str, float]]], surfaces: tuple[str, ...]
) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for task, task_surfaces in sorted(contrasts.items()):
        present = [task_surfaces[surface] for surface in surfaces if surface in task_surfaces]
        if len(present) != len(surfaces):
            continue
        rows.append(
            {
                key: mean(item[key] for item in present)
                for key in present[0]
            }
        )
    output: dict[str, Any] = {}
    for field in (
        "earlier",
        "readout",
        "congruent",
        "conflict",
        "earlier_final_projection",
        "earlier_dose_projection",
    ):
        output[field] = bootstrap(rows, lambda sample, f=field: mean(row[f] for row in sample))

    def ratio(sample: list[dict[str, float]]) -> float:
        denominator = mean(row["readout"] for row in sample)
        return mean(row["earlier"] for row in sample) / denominator if abs(denominator) > 1e-12 else math.nan

    output["retention_ratio_of_means"] = bootstrap(rows, ratio)
    output["readout_dominates_conflict"] = bootstrap(
        rows, lambda sample: mean(row["conflict"] for row in sample)
    )
    return output


def summarize_reversed_pair(
    contrasts: dict[int, dict[str, dict[str, float]]], pair: tuple[str, str]
) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    first, second = pair
    for task, task_surfaces in sorted(contrasts.items()):
        if first not in task_surfaces or second not in task_surfaces:
            continue
        row: dict[str, float] = {}
        for field in ("earlier", "readout", "congruent", "conflict"):
            a = task_surfaces[first][field]
            b = task_surfaces[second][field]
            row[f"{field}_semantic_binding"] = (a + b) / 2
            row[f"{field}_fixed_token_bias"] = (a - b) / 2
        rows.append(row)
    return {
        field: bootstrap(rows, lambda sample, f=field: mean(row[f] for row in sample))
        for field in rows[0]
    }


def available_group_rows(
    records: list[dict[str, Any]],
    vector: str,
    factor: float,
    interval: str,
    surfaces: tuple[str, ...],
) -> list[dict[str, float]]:
    """Form task-level earlier/readout contrasts when only four cells exist."""

    lookup = cell_lookup(records)
    task_ids = sorted(
        {
            int(row["task_index"])
            for row in records
            if row["vector"] == vector
            and float(row["factor"]) == factor
            and row["interval"] == interval
        }
    )
    rows: list[dict[str, float]] = []
    for task in task_ids:
        surface_rows: list[dict[str, float]] = []
        for surface in surfaces:
            base = (vector, factor, interval, task, surface)
            cells = {
                (p, r): lookup.get((*base, p, r))
                for p, r in ((1, 0), (-1, 0), (0, 1), (0, -1))
            }
            if any(value is None for value in cells.values()):
                break
            surface_rows.append(
                {
                    "earlier": cells[(1, 0)]["endpoint"] - cells[(-1, 0)]["endpoint"],
                    "readout": cells[(0, 1)]["endpoint"] - cells[(0, -1)]["endpoint"],
                }
            )
        if len(surface_rows) == len(surfaces):
            rows.append(
                {
                    "task_index": float(task),
                    "earlier": mean(row["earlier"] for row in surface_rows),
                    "readout": mean(row["readout"] for row in surface_rows),
                }
            )
    return rows


def summarize_available(rows: list[dict[str, float]]) -> dict[str, Any]:
    return {
        field: bootstrap(rows, lambda sample, f=field: mean(row[f] for row in sample))
        for field in ("earlier", "readout")
    }


def paired_vector_difference(
    primary: list[dict[str, float]], control: list[dict[str, float]], field: str
) -> dict[str, float]:
    control_by_task = {int(row["task_index"]): row for row in control}
    differences = [
        {
            "difference": row[field] - control_by_task[int(row["task_index"])][field]
        }
        for row in primary
        if int(row["task_index"]) in control_by_task
    ]
    return bootstrap(differences, lambda sample: mean(row["difference"] for row in sample))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=pathlib.Path)
    args = parser.parse_args()
    records = load_records(args.run_dir / "records.jsonl")
    primary = surface_contrasts(records, "welfare_raw", 0.5, "short")
    primary_long = surface_contrasts(records, "welfare_raw", 0.5, "long")
    result: dict[str, Any] = {
        "analysis_status": "complete",
        "bootstrap_seed": SEED,
        "bootstrap_draws": BOOTSTRAPS,
        "primary": {},
        "reversed_encoding_decomposition": {},
        "kl_matched_controls_long": {},
        "paired_primary_minus_control_long": {},
        "dose_response_long": {},
    }
    for interval, contrasts in (("short", primary), ("long", primary_long)):
        result["primary"][interval] = {
            group: summarize_group(contrasts, surfaces)
            for group, surfaces in SURFACE_GROUPS.items()
        }
        result["reversed_encoding_decomposition"][interval] = {
            name: summarize_reversed_pair(contrasts, pair)
            for name, pair in REVERSED_PAIRS.items()
        }

    vector_factors = {
        row["vector"]: float(row["factor"])
        for row in records
        if row["interval"] == "long"
    }
    vector_factors["welfare_raw"] = 0.5
    control_groups = {
        "semantic": SURFACE_GROUPS["semantic"],
        "opaque": SURFACE_GROUPS["opaque"],
    }
    available_rows: dict[tuple[str, str], list[dict[str, float]]] = {}
    for vector, factor in sorted(vector_factors.items()):
        if vector == "welfare_raw" and factor != 0.5:
            continue
        result["kl_matched_controls_long"][vector] = {
            "factor": factor,
            "groups": {},
        }
        for group, surfaces in control_groups.items():
            rows = available_group_rows(records, vector, factor, "long", surfaces)
            available_rows[(vector, group)] = rows
            result["kl_matched_controls_long"][vector]["groups"][group] = summarize_available(rows)

    for vector in sorted(vector_factors):
        if vector == "welfare_raw":
            continue
        result["paired_primary_minus_control_long"][vector] = {}
        for group in control_groups:
            result["paired_primary_minus_control_long"][vector][group] = {
                field: paired_vector_difference(
                    available_rows[("welfare_raw", group)],
                    available_rows[(vector, group)],
                    field,
                )
                for field in ("earlier", "readout")
            }

    for factor in (0.25, 0.5, 1.0):
        result["dose_response_long"][str(factor)] = {
            group: summarize_available(
                available_group_rows(records, "welfare_raw", factor, "long", surfaces)
            )
            for group, surfaces in control_groups.items()
        }
    output_path = args.run_dir / "analysis-summary.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
