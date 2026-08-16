#!/usr/bin/env python3
"""Independently recompute the Qwen BF16 semantic-choice HF Job result.

This auditor intentionally does not import the executed runner. It consumes the
captured raw Hugging Face Job log and the frozen NF4 source records, reconstructs
all pairings and summaries, and reports both outcome checks and known seal
deviations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
from collections import defaultdict
from statistics import mean, median
from typing import Any, Iterable


EXPECTED_CONDITIONS = (
    "clean",
    "fwa_minus_1",
    "fwa_minus_half",
    "fwa_plus_half",
    "fwa_plus_1",
    "path_minus",
    "path_plus",
)
VALID_TOKEN_IDS = {"continue": 9534, "stop": 9495}
RECORDED_BOOTSTRAP_SEED = 26081682
BOOTSTRAPS = 10_000


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: Iterable[float], probability: float) -> float:
    rows = sorted(float(value) for value in values)
    position = probability * (len(rows) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    fraction = position - left
    return rows[left] * (1.0 - fraction) + rows[right] * fraction


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("Pearson inputs differ or are empty")
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def parse_labeled_json(log_path: pathlib.Path) -> dict[str, list[Any]]:
    labels: dict[str, list[Any]] = defaultdict(list)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("QWEN_BF16_CHOICE_") or "=" not in line:
            index += 1
            continue
        label, payload = line.split("=", 1)
        # HF log transport hard-wraps long compact JSON lines at chunk
        # boundaries, including inside quoted strings. Join only until one
        # complete JSON value parses; unrelated runtime log lines are ignored.
        while True:
            try:
                value = json.loads(payload)
                break
            except json.JSONDecodeError:
                index += 1
                if index >= len(lines) or lines[index].startswith(
                    "QWEN_BF16_CHOICE_"
                ):
                    raise
                payload += lines[index]
        labels[label].append(value)
        index += 1
    return labels


def extract_records(labels: dict[str, list[Any]]) -> list[dict[str, Any]]:
    chunks = labels.get("QWEN_BF16_CHOICE_RECORDS", [])
    if len(chunks) != 7:
        raise ValueError(f"expected seven record chunks, found {len(chunks)}")
    if tuple(chunk["condition"] for chunk in chunks) != EXPECTED_CONDITIONS:
        raise ValueError("condition chunk order differs from frozen order")
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk["n"] != 64 or len(chunk["records"]) != 64:
            raise ValueError(f"condition {chunk['condition']} lacks 64 records")
        if any(row["condition"] != chunk["condition"] for row in chunk["records"]):
            raise ValueError(f"condition label mismatch in {chunk['condition']}")
        records.extend(chunk["records"])
    if len(records) != 448:
        raise ValueError(f"expected 448 records, found {len(records)}")
    return records


def validate_record_custody(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_condition[row["condition"]].append(row)
        choice = row["choice"]
        top_id = int(row["top_token_id"])
        expected_valid = choice in VALID_TOKEN_IDS and top_id == VALID_TOKEN_IDS[choice]
        if bool(row["valid"]) != expected_valid:
            raise ValueError(f"validity/token mismatch in {row['case_id']}")
    case_ids = [row["case_id"] for row in by_condition["clean"]]
    if len(case_ids) != 64 or len(set(case_ids)) != 64:
        raise ValueError("clean condition lacks 64 unique cases")
    reference = {
        row["case_id"]: (
            row["setting_index"],
            row["task_index"],
            row["order_id"],
            row["continue_votes"],
            row["expected_choice"],
            row["rendered_prompt_sha256"],
        )
        for row in by_condition["clean"]
    }
    for condition in EXPECTED_CONDITIONS:
        rows = by_condition[condition]
        if len(rows) != 64 or {row["case_id"] for row in rows} != set(reference):
            raise ValueError(f"case set differs for {condition}")
        for row in rows:
            observed = (
                row["setting_index"],
                row["task_index"],
                row["order_id"],
                row["continue_votes"],
                row["expected_choice"],
                row["rendered_prompt_sha256"],
            )
            if observed != reference[row["case_id"]]:
                raise ValueError(f"case metadata differs for {condition}/{row['case_id']}")
    return {
        "records": len(records),
        "conditions": len(by_condition),
        "cases_per_condition": sorted({len(rows) for rows in by_condition.values()}),
        "unique_case_ids": len(reference),
        "all_records_valid": all(row["valid"] for row in records),
        "all_clean_and_fwa_records_valid": all(
            row["valid"] for row in records if row["condition"] != "path_minus" and row["condition"] != "path_plus"
        ),
        "canonical_sha256": sha256_bytes(canonical_json(records).encode("utf-8")),
    }


def pair_rows(
    records: list[dict[str, Any]], minus_name: str, plus_name: str
) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        by_case[row["case_id"]][row["condition"]] = row
    paired: list[dict[str, Any]] = []
    for case_id in sorted(by_case):
        lookup = by_case[case_id]
        clean, minus, plus = lookup["clean"], lookup[minus_name], lookup[plus_name]
        paired.append(
            {
                "case_id": case_id,
                "order_id": clean["order_id"],
                "continue_votes": int(clean["continue_votes"]),
                "task_index": int(clean["task_index"]),
                "clean_choice": clean["choice"],
                "minus_choice": minus["choice"],
                "plus_choice": plus["choice"],
                "all_valid": clean["valid"] and minus["valid"] and plus["valid"],
                "margin_contrast": float(plus["continue_minus_stop_margin"])
                - float(minus["continue_minus_stop_margin"]),
                "oriented_switch": minus["choice"] == "stop"
                and plus["choice"] == "continue",
                "any_switch": minus["choice"] != plus["choice"],
                "clean_to_plus": clean["choice"] == "stop"
                and plus["choice"] == "continue",
                "clean_to_minus": clean["choice"] == "continue"
                and minus["choice"] == "stop",
            }
        )
    if len(paired) != 64:
        raise ValueError(f"expected 64 paired cases, found {len(paired)}")
    return paired


def bootstrap_ci(paired: list[dict[str, Any]], seed: int) -> list[float]:
    cluster_means = {
        task_index: mean(
            row["margin_contrast"]
            for row in paired
            if row["task_index"] == task_index
        )
        for task_index in range(8)
    }
    generator = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAPS):
        sample = [generator.randrange(8) for _ in range(8)]
        draws.append(mean(cluster_means[index] for index in sample))
    return [quantile(draws, 0.025), quantile(draws, 0.975)]


def summarize_pair(
    paired: list[dict[str, Any]], actual_seed: int
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for order_id in sorted({row["order_id"] for row in paired}):
        for continue_votes in (3, 4):
            selected = [
                row
                for row in paired
                if row["order_id"] == order_id
                and row["continue_votes"] == continue_votes
            ]
            label = f"{order_id}__c{continue_votes}"
            cells[label] = {
                "n": len(selected),
                "mean_margin_contrast": mean(
                    row["margin_contrast"] for row in selected
                ),
                "positive_margin_contrast_rate": mean(
                    float(row["margin_contrast"] > 0) for row in selected
                ),
                "oriented_switch_count": sum(
                    row["oriented_switch"] for row in selected
                ),
                "clean_to_plus_count": sum(row["clean_to_plus"] for row in selected),
                "clean_to_minus_count": sum(
                    row["clean_to_minus"] for row in selected
                ),
            }
    return {
        "n_cases": len(paired),
        "all_valid_rate": mean(float(row["all_valid"]) for row in paired),
        "mean_margin_contrast": mean(row["margin_contrast"] for row in paired),
        "median_margin_contrast": median(
            row["margin_contrast"] for row in paired
        ),
        "positive_margin_contrast_rate": mean(
            float(row["margin_contrast"] > 0) for row in paired
        ),
        "oriented_switch_count": sum(row["oriented_switch"] for row in paired),
        "any_switch_count": sum(row["any_switch"] for row in paired),
        "clean_to_plus_count": sum(row["clean_to_plus"] for row in paired),
        "clean_to_minus_count": sum(row["clean_to_minus"] for row in paired),
        "all_8_cell_means_positive": all(
            cell["mean_margin_contrast"] > 0 for cell in cells.values()
        ),
        "cells": cells,
        "bootstrap": {
            "draws": BOOTSTRAPS,
            "actual_code_seed": actual_seed,
            "actual_code_ci95": bootstrap_ci(paired, actual_seed),
            "recorded_seed_applied_to_each_pair": RECORDED_BOOTSTRAP_SEED,
            "recorded_seed_applied_to_each_pair_ci95": bootstrap_ci(
                paired, RECORDED_BOOTSTRAP_SEED
            ),
        },
        "oriented_switch_case_ids": sorted(
            row["case_id"] for row in paired if row["oriented_switch"]
        ),
        "any_switch_case_ids": sorted(
            row["case_id"] for row in paired if row["any_switch"]
        ),
    }


def load_nf4_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        vector, sign = row["vector"], int(row["sign"])
        if vector == "clean":
            condition = "clean"
        elif vector == "functional_welfare_raw":
            condition = "fwa_minus_1" if sign < 0 else "fwa_plus_1"
        elif vector == "functional_welfare_raw_half":
            condition = "fwa_minus_half" if sign < 0 else "fwa_plus_half"
        elif vector == "path_direct":
            condition = "path_minus" if sign < 0 else "path_plus"
        else:
            continue
        records.append({**row, "condition": condition})
    if len(records) != 448:
        raise ValueError(f"expected 448 NF4 comparison rows, found {len(records)}")
    return records


def precision_concordance(
    bf16_records: list[dict[str, Any]],
    nf4_records: list[dict[str, Any]],
    bf16_pairs: dict[str, list[dict[str, Any]]],
    nf4_pairs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    bf16_lookup = {
        (row["condition"], row["case_id"]): row for row in bf16_records
    }
    nf4_lookup = {
        (row["condition"], row["case_id"]): row for row in nf4_records
    }
    if set(bf16_lookup) != set(nf4_lookup):
        raise ValueError("BF16 and NF4 condition/case keys differ")
    choice_agreement = {}
    for condition in EXPECTED_CONDITIONS:
        keys = sorted(key for key in bf16_lookup if key[0] == condition)
        choice_agreement[condition] = {
            "n": len(keys),
            "same_choice_count": sum(
                bf16_lookup[key]["choice"] == nf4_lookup[key]["choice"]
                for key in keys
            ),
        }
        choice_agreement[condition]["same_choice_rate"] = (
            choice_agreement[condition]["same_choice_count"] / len(keys)
        )

    pair_concordance = {}
    for name in ("fwa_1", "fwa_half", "path"):
        bf_rows = {row["case_id"]: row for row in bf16_pairs[name]}
        nf_rows = {row["case_id"]: row for row in nf4_pairs[name]}
        bf_switches = {case_id for case_id, row in bf_rows.items() if row["any_switch"]}
        nf_switches = {case_id for case_id, row in nf_rows.items() if row["any_switch"]}
        intersection = bf_switches & nf_switches
        union = bf_switches | nf_switches
        case_ids = sorted(bf_rows)
        bf_margins = [bf_rows[case_id]["margin_contrast"] for case_id in case_ids]
        nf_margins = [nf_rows[case_id]["margin_contrast"] for case_id in case_ids]
        pair_concordance[name] = {
            "bf16_switch_count": len(bf_switches),
            "nf4_switch_count": len(nf_switches),
            "intersection_count": len(intersection),
            "union_count": len(union),
            "jaccard": len(intersection) / len(union) if union else 1.0,
            "nf4_switch_retained_rate": (
                len(intersection) / len(nf_switches) if nf_switches else None
            ),
            "bf16_switch_also_nf4_rate": (
                len(intersection) / len(bf_switches) if bf_switches else None
            ),
            "intersection_case_ids": sorted(intersection),
            "bf16_only_case_ids": sorted(bf_switches - nf_switches),
            "nf4_only_case_ids": sorted(nf_switches - bf_switches),
            "margin_contrast_pearson": pearson(bf_margins, nf_margins),
            "positive_sign_agreement_count": sum(
                (bf_rows[case_id]["margin_contrast"] > 0)
                == (nf_rows[case_id]["margin_contrast"] > 0)
                for case_id in case_ids
            ),
            "positive_in_both_count": sum(
                bf_rows[case_id]["margin_contrast"] > 0
                and nf_rows[case_id]["margin_contrast"] > 0
                for case_id in case_ids
            ),
            "cell_signs_positive_both_precisions": all(
                mean(
                    row["margin_contrast"]
                    for row in bf_rows.values()
                    if row["order_id"] == order_id
                    and row["continue_votes"] == continue_votes
                )
                > 0
                and mean(
                    row["margin_contrast"]
                    for row in nf_rows.values()
                    if row["order_id"] == order_id
                    and row["continue_votes"] == continue_votes
                )
                > 0
                for order_id in sorted({row["order_id"] for row in bf_rows.values()})
                for continue_votes in (3, 4)
            ),
        }
    return {
        "choice_agreement_by_condition": choice_agreement,
        "pair_concordance": pair_concordance,
    }


def compare_recomputed_to_emitted(
    recomputed: dict[str, Any], emitted: dict[str, Any]
) -> dict[str, Any]:
    field_map = {
        "all_valid_rate": "all_valid_rate",
        "mean_margin_contrast": "mean_margin_contrast",
        "median_margin_contrast": "median_margin_contrast",
        "positive_margin_contrast_rate": "positive_margin_contrast_rate",
        "oriented_switch_count": "oriented_sign_pair_switch_count",
        "any_switch_count": "any_sign_pair_switch_count",
        "clean_to_plus_count": "clean_to_plus_stop_to_continue_count",
        "clean_to_minus_count": "clean_to_minus_continue_to_stop_count",
        "all_8_cell_means_positive": "all_8_cell_mean_contrasts_positive",
    }
    differences = {}
    for local, remote in field_map.items():
        if recomputed[local] != emitted[remote]:
            differences[local] = {
                "recomputed": recomputed[local],
                "emitted": emitted[remote],
            }
    if recomputed["bootstrap"]["actual_code_ci95"] != emitted[
        "cluster_bootstrap_mean_margin_contrast_ci95"
    ]:
        differences["actual_bootstrap_ci95"] = {
            "recomputed": recomputed["bootstrap"]["actual_code_ci95"],
            "emitted": emitted["cluster_bootstrap_mean_margin_contrast_ci95"],
        }
    return {"exact": not differences, "differences": differences}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-log", type=pathlib.Path, required=True)
    parser.add_argument("--nf4-records", type=pathlib.Path, required=True)
    parser.add_argument(
        "--emit",
        choices=("audit", "result", "records-jsonl"),
        default="audit",
    )
    args = parser.parse_args()

    labels = parse_labeled_json(args.raw_log)
    records = extract_records(labels)
    custody = validate_record_custody(records)
    results = labels.get("QWEN_BF16_CHOICE_RESULT", [])
    if len(results) != 1:
        raise ValueError(f"expected one result line, found {len(results)}")
    emitted = results[0]
    if emitted.get("status") != "complete":
        raise ValueError(f"job result is not complete: {emitted.get('status')}")
    if args.emit == "result":
        print(json.dumps(emitted, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.emit == "records-jsonl":
        for row in records:
            print(canonical_json(row))
        return 0

    pair_specs = {
        "fwa_1": ("fwa_minus_1", "fwa_plus_1", RECORDED_BOOTSTRAP_SEED + 1),
        "fwa_half": (
            "fwa_minus_half",
            "fwa_plus_half",
            RECORDED_BOOTSTRAP_SEED + 2,
        ),
        "path": ("path_minus", "path_plus", RECORDED_BOOTSTRAP_SEED + 3),
    }
    bf16_pairs = {
        name: pair_rows(records, minus, plus)
        for name, (minus, plus, _seed) in pair_specs.items()
    }
    summaries = {
        name: summarize_pair(bf16_pairs[name], seed)
        for name, (_minus, _plus, seed) in pair_specs.items()
    }
    emitted_comparison = {
        name: compare_recomputed_to_emitted(summary, emitted["bf16_summary"][name])
        for name, summary in summaries.items()
    }
    if custody["canonical_sha256"] != emitted["records_canonical_sha256"]:
        raise ValueError("raw-record canonical hash differs from emitted result")

    nf4_records = load_nf4_records(args.nf4_records)
    nf4_pairs = {
        name: pair_rows(nf4_records, minus, plus)
        for name, (minus, plus, _seed) in pair_specs.items()
    }
    concordance = precision_concordance(
        records, nf4_records, bf16_pairs, nf4_pairs
    )

    scientific_checks = {
        "all_clean_and_fwa_outputs_valid": custody[
            "all_clean_and_fwa_records_valid"
        ],
        "fwa_1_mean_positive": summaries["fwa_1"]["mean_margin_contrast"] > 0,
        "fwa_half_mean_positive": summaries["fwa_half"]["mean_margin_contrast"] > 0,
        "fwa_1_all_8_cells_positive": summaries["fwa_1"][
            "all_8_cell_means_positive"
        ],
        "fwa_half_all_8_cells_positive": summaries["fwa_half"][
            "all_8_cell_means_positive"
        ],
        "factor_1_exceeds_half": summaries["fwa_1"]["mean_margin_contrast"]
        > summaries["fwa_half"]["mean_margin_contrast"],
        "fwa_1_oriented_switch_positive": summaries["fwa_1"][
            "oriented_switch_count"
        ]
        > 0,
        "fwa_half_oriented_switch_positive": summaries["fwa_half"][
            "oriented_switch_count"
        ]
        > 0,
    }
    path_vs_target = {
        "path_switches": summaries["path"]["any_switch_count"],
        "fwa_1_switches": summaries["fwa_1"]["any_switch_count"],
        "path_minus_fwa_1_switches": summaries["path"]["any_switch_count"]
        - summaries["fwa_1"]["any_switch_count"],
        "path_mean_contrast": summaries["path"]["mean_margin_contrast"],
        "fwa_1_mean_contrast": summaries["fwa_1"]["mean_margin_contrast"],
        "path_to_fwa_1_mean_contrast_ratio": summaries["path"][
            "mean_margin_contrast"
        ]
        / summaries["fwa_1"]["mean_margin_contrast"],
        "path_stronger_on_both_switches_and_mean_contrast": (
            summaries["path"]["any_switch_count"]
            > summaries["fwa_1"]["any_switch_count"]
            and summaries["path"]["mean_margin_contrast"]
            > summaries["fwa_1"]["mean_margin_contrast"]
        ),
    }
    audit = {
        "schema": "qwen-bf16-semantic-choice-independent-audit-v1",
        "raw_log": str(args.raw_log),
        "raw_log_sha256": sha256_file(args.raw_log),
        "terminal_result_status": emitted["status"],
        "reported_qualitative_pass": emitted[
            "qualitative_fwa_precision_replication_pass"
        ],
        "post_hoc_fwa_core_checks_excluding_path_validity_and_vram": scientific_checks,
        "post_hoc_fwa_core_pass_excluding_path_validity_and_vram": all(
            scientific_checks.values()
        ),
        "operational_observations": {
            "all_448_including_path_valid": custody["all_records_valid"],
            "vram_peak_reserved_bytes": emitted["vram"]["peak_reserved_bytes"],
            "vram_limit_bytes": 24 * 1024**3,
            "vram_pass": emitted["vram"]["peak_reserved_bytes"] <= 24 * 1024**3,
        },
        "record_custody": custody,
        "recomputed_bf16": summaries,
        "emitted_summary_exact_recomputation": emitted_comparison,
        "nf4_bf16_concordance": concordance,
        "path_comparison": path_vs_target,
        "reproducibility": {
            "retained_448_rows_exact_for_this_execution": (
                custody["records"] == 448
                and custody["canonical_sha256"]
                == emitted["records_canonical_sha256"]
            ),
            "bitwise_rerun_determinism_verified": False,
            "warning": (
                "The raw log records CuBLAS nondeterminism warnings because "
                "CUBLAS_WORKSPACE_CONFIG was unset. The retained rows are exact "
                "for this execution, but a bitwise-identical rerun is not established."
            ),
        },
        "seal_deviations": [
            {
                "id": "bootstrap_seed_semantics_not_fully_specified",
                "detail": (
                    "The frozen design records only seed 26081682, while the "
                    "preregistration binds that design hash but does not separately "
                    "state per-pair seed semantics. Runner code used effective seeds "
                    "26081683, 26081684, and 26081685 for FWA1, FWA-half, and path; "
                    "those effective seeds were not emitted in the job result. Both "
                    "actual-code intervals and recorded-seed-applied-to-each-pair "
                    "sensitivity intervals are recomputed above. Qualitative pass "
                    "does not depend on any bootstrap interval."
                ),
            },
            {
                "id": "path_gate_scope_contradiction",
                "detail": (
                    "The preregistered all-448-outputs-valid criterion explicitly "
                    "includes 128 path forwards and is conjunctive in the reported "
                    "pass, while other frozen prose says path is outside the FWA "
                    "gate. All path outputs were valid, so this gate-scope "
                    "contradiction did not change the observed pass."
                ),
            },
            {
                "id": "vram_gate_scope_contradiction",
                "detail": (
                    "Runner additionally conjuncts VRAM pass into the reported "
                    "qualitative pass although the frozen scientific gate list omits "
                    "it. Peak reserved memory was below the limit, so this gate-scope "
                    "contradiction did not change the observed pass."
                ),
            },
        ],
        "interpretation": (
            "The exact-cohort BF16 semantic action-boundary actuation result is "
            "positive and independently reproducible from raw records, but the run "
            "does not have a pristine frozen-specification seal. The three "
            "deviations are disclosed rather than retroactively repaired. Path is "
            "stronger than FWA1, so the result does not establish target specificity."
        ),
    }
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
