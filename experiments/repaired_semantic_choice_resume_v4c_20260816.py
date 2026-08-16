#!/usr/bin/env python3
"""One-shot semantic-choice V4-C target run with a conservative path control.

This runner binds the exact zero-target V4-B stop and reuses its already
measured neutral-control factors.  The sole change is to choose an already
evaluated path factor whose neutral KL is 1.2696x the target, ensuring the path
control is stronger rather than weaker.  It performs no clean rerun, neutral
recalibration, prompt change, case selection, or endpoint-dependent tuning.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time
from typing import Any

import torch


ROOT = pathlib.Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

import repaired_semantic_choice_20260816 as base  # noqa: E402
import repaired_semantic_choice_validation_20260816 as v4  # noqa: E402
import repaired_semantic_choice_resume_v4a_20260816 as v4a  # noqa: E402
import repaired_semantic_choice_resume_v4b_20260816 as v4b  # noqa: E402


EXPECTED_V4B_RUNNER_SHA256 = (
    "5721e91757a73dd9fe3f0665f6d88ba14a1b5c0b9714fcd70d86385dbfec687c"
)
EXPECTED_V4B_AMENDMENT_SHA256 = (
    "c5a3ef651b9016e8d2dc48afeca45c553809547ddc1f0ec3f1345e0f3b5162a6"
)
EXPECTED_V4C_AMENDMENT_SHA256 = (
    "12e516d449c450b09e65d22bbcf82a43ba122fe0fbc287acb5ab69e7a30442c1"
)
EXPECTED_V4B_STOPPED_RUN = (
    PROJECT
    / "experiments/runs/repaired-semantic-choice-validation-v4b-resume-20260816T173151Z"
)
EXPECTED_V4C_AMENDMENT = (
    ROOT / "repaired-semantic-choice-amendment-v4c-20260816.json"
)
EXPECTED_V4B_STOP_ARTIFACTS = {
    "adaptive-clean-gate.json": (
        "1df09bee981fdc4af3c19a8b51349ef3171dcc0ee3c57f8c04f9c0c3f345b119"
    ),
    "artifact-custody-preflight.json": (
        "d391013b1aab19d8e60255f5620912c4808a2ebd7d9920aacd4d909afe36d0d0"
    ),
    "custody.json": (
        "519ac21ffd61c6f3d01201f5cb8739e58bc92105779d9a736fe5f5c4d9e86423"
    ),
    "refined-kl-calibration.json": (
        "ce8b21c21b1cd9a48dbdd249a2678907f8caa937cf8cb81ec1b8606bf112e1d3"
    ),
    "regenerated-case-custody.json": (
        "d7b00cc0e5b033b4d4eabe64c5f6ec66895b3f6bbaedd665104312c22bf98c65"
    ),
    "summary.json": (
        "029ad54291326d8c547fcaa1f0d68425479e74c1f4bbe0abf0f48096a8110b9c"
    ),
    "vector-metadata-preintervention.json": (
        "5643d7a28172ec103c61c6238209001c42f6ba86c29cc5974f176d227616f520"
    ),
}
FORBIDDEN_V4B_TARGET_ARTIFACTS = (
    "vectors.pt",
    "vector-metadata.json",
    "validation-records.jsonl",
)
FIXED_FACTORS = {
    "functional_welfare_raw": 1.0,
    "path_direct": 1.1940978759847163,
    "random_orthogonal_00": 2.1579784664920996,
    "random_orthogonal_01": 1.5272141178181842,
    "random_orthogonal_02": 0.5128326982332162,
    "random_orthogonal_03": 1.9643610971105179,
    "sentiment": 1.425869737431211,
}
FIXED_NEUTRAL_KL_RATIOS = {
    "functional_welfare_raw": 1.0,
    "path_direct": 1.2696314866720828,
    "random_orthogonal_00": 0.857366832002642,
    "random_orthogonal_01": 0.9789662127882794,
    "random_orthogonal_02": 1.0233316284422416,
    "random_orthogonal_03": 0.9743369217720128,
    "sentiment": 0.9386691382839856,
}
RUN_ROOT = ROOT / "runs"


def sha256_file(path: pathlib.Path) -> str:
    return v4.sha256_file(path)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hash(path: pathlib.Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return observed


def verify_v4b_zero_target_and_fixed_factors(
    stopped_run: pathlib.Path, amendment_path: pathlib.Path
) -> dict[str, Any]:
    if stopped_run.resolve() != EXPECTED_V4B_STOPPED_RUN.resolve():
        raise ValueError("must consume exact stopped V4-B run")
    if amendment_path.resolve() != EXPECTED_V4C_AMENDMENT.resolve():
        raise ValueError("must consume exact V4-C amendment")
    hashes = {
        "v4b_runner": verify_hash(
            ROOT / "repaired_semantic_choice_resume_v4b_20260816.py",
            EXPECTED_V4B_RUNNER_SHA256,
            "V4-B runner",
        ),
        "v4b_amendment": verify_hash(
            ROOT / "repaired-semantic-choice-amendment-v4b-20260816.json",
            EXPECTED_V4B_AMENDMENT_SHA256,
            "V4-B amendment",
        ),
        "v4c_amendment": verify_hash(
            amendment_path, EXPECTED_V4C_AMENDMENT_SHA256, "V4-C amendment"
        ),
    }
    for name, expected in EXPECTED_V4B_STOP_ARTIFACTS.items():
        hashes[f"v4b_stop/{name}"] = verify_hash(
            stopped_run / name, expected, f"V4-B stop {name}"
        )
    forbidden = [
        name for name in FORBIDDEN_V4B_TARGET_ARTIFACTS if (stopped_run / name).exists()
    ]
    if forbidden:
        raise ValueError(f"V4-B stop contains target artifacts: {forbidden}")
    summary = read_json(stopped_run / "summary.json")
    if summary.get("status") != "stopped_v4b_fixed_grid_neutral_kl_gate":
        raise ValueError("V4-B summary does not have expected KL-stop status")
    if summary.get("target_or_control_validation_forwards") != 0:
        raise ValueError("V4-B reports nonzero held-out intervention forwards")
    calibration = read_json(stopped_run / "refined-kl-calibration.json")
    path_rows = calibration["controls"]["path_direct"]["evaluations"]
    exact_path = [
        row
        for row in path_rows
        if row["factor"] == FIXED_FACTORS["path_direct"]
    ]
    if len(exact_path) != 1:
        raise ValueError("fixed conservative path factor is not exact V4-B grid point")
    if exact_path[0]["ratio_to_target"] != FIXED_NEUTRAL_KL_RATIOS["path_direct"]:
        raise ValueError("fixed conservative path ratio differs from V4-B receipt")
    for name in v4a.MANDATORY_CONTROLS:
        if name == "path_direct":
            continue
        control = calibration["controls"][name]
        if control["chosen_factor"] != FIXED_FACTORS[name]:
            raise ValueError(f"fixed factor differs from V4-B chosen factor for {name}")
        if control["chosen_ratio_to_target"] != FIXED_NEUTRAL_KL_RATIOS[name]:
            raise ValueError(f"fixed ratio differs from V4-B receipt for {name}")
    path_ratio = FIXED_NEUTRAL_KL_RATIOS["path_direct"]
    revised_gate = {
        "path_control_is_not_weaker_than_target": path_ratio >= 1.0,
        "path_control_ratio_at_most_1p35": path_ratio <= 1.35,
        "all_other_controls_ratio_0p8_to_1p25": all(
            0.8 <= FIXED_NEUTRAL_KL_RATIOS[name] <= 1.25
            for name in v4a.MANDATORY_CONTROLS
            if name != "path_direct"
        ),
    }
    if not all(revised_gate.values()):
        raise ValueError("sealed V4-C conservative neutral gate does not pass")
    amendment = read_json(amendment_path)
    if amendment.get("schema") != (
        "repaired-semantic-choice-conservative-path-control-amendment-v4c-v1"
    ):
        raise ValueError("unexpected V4-C amendment schema")
    amendment_factors = amendment["fixed_factors_and_neutral_kl_ratios"]
    for name, factor in FIXED_FACTORS.items():
        if amendment_factors[name]["factor"] != factor:
            raise ValueError(f"V4-C amendment factor mismatch for {name}")
        if amendment_factors[name]["ratio"] != FIXED_NEUTRAL_KL_RATIOS[name]:
            raise ValueError(f"V4-C amendment ratio mismatch for {name}")
    return {
        "passed": True,
        "hashes": hashes,
        "v4b_status": summary["status"],
        "v4b_target_or_control_validation_forwards": 0,
        "v4b_target_artifacts_absent": True,
        "fixed_factors": FIXED_FACTORS,
        "fixed_neutral_kl_ratios": FIXED_NEUTRAL_KL_RATIOS,
        "revised_neutral_gate": revised_gate,
        "path_policy": "conservative_overshoot_control_not_weaker_than_target",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4b-stopped-run", default=str(EXPECTED_V4B_STOPPED_RUN))
    parser.add_argument("--amendment", default=str(EXPECTED_V4C_AMENDMENT))
    args = parser.parse_args()
    stopped_run = pathlib.Path(args.v4b_stopped_run).resolve()
    amendment_path = pathlib.Path(args.amendment).resolve()
    started = time.perf_counter()

    v4c_preflight = verify_v4b_zero_target_and_fixed_factors(
        stopped_run, amendment_path
    )
    # Re-verify the full original custody chain and exact clean gate without any
    # new clean or neutral model forward.
    v4b.verify_v4a_zero_target_stop(
        v4b.EXPECTED_V4A_STOPPED_RUN, v4b.EXPECTED_V4B_AMENDMENT
    )
    v4a.verify_artifact_custody(
        v4a.EXPECTED_DESIGN_PATH,
        v4a.EXPECTED_PRIOR_RUN,
        v4a.EXPECTED_AMENDMENT_PATH,
    )
    design = v4.load_design(v4a.EXPECTED_DESIGN_PATH)
    manifest = read_json(v4a.EXPECTED_PRIOR_RUN / "case-manifest.json")
    clean_rows = v4a.read_jsonl(v4a.EXPECTED_PRIOR_RUN / "clean-records.jsonl")
    adaptive_gate = v4a.recompute_adaptive_clean_gate(
        clean_rows, read_json(v4a.EXPECTED_AMENDMENT_PATH)
    )
    if not adaptive_gate["passed"]:
        raise ValueError("exact adaptive clean gate no longer passes")

    tokenizer, model = base.load_model()
    base.TOKENIZER = tokenizer
    ids = {name: base.one_token(tokenizer, name) for name in ("continue", "stop")}
    if ids != design["valid_answer_tokens"]:
        raise ValueError("answer token IDs differ from frozen design")
    cases, case_custody = v4a.verify_regenerated_cases(
        tokenizer, design, manifest, clean_rows
    )

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_ROOT / f"repaired-semantic-choice-validation-v4c-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    v4.write_json(run_dir / "artifact-custody-preflight.json", v4c_preflight)
    v4.write_json(run_dir / "adaptive-clean-gate.json", adaptive_gate)
    v4.write_json(run_dir / "regenerated-case-custody.json", case_custody)
    v4.write_json(
        run_dir / "fixed-neutral-control-receipt.json",
        {
            "source": str(stopped_run / "refined-kl-calibration.json"),
            "source_sha256": sha256_file(
                stopped_run / "refined-kl-calibration.json"
            ),
            "neutral_recalibration_forwards": 0,
            "factors": FIXED_FACTORS,
            "ratios": FIXED_NEUTRAL_KL_RATIOS,
            "path_policy": "conservative_overshoot_control_not_weaker_than_target",
            "revised_neutral_gate_passed": True,
        },
    )

    vectors, vector_metadata = base.build_vectors(tokenizer, model)
    vector_metadata["target_artifact_sha256"] = {
        name: sha256_file(pathlib.Path(path))
        for name, path in vector_metadata["target_artifacts"].items()
    }
    factors = dict(FIXED_FACTORS)
    vectors["functional_welfare_raw_half"] = vectors[
        "functional_welfare_raw"
    ].clone()
    factors["functional_welfare_raw_half"] = base.SECONDARY_TARGET_FACTOR
    vector_metadata["evaluation_factors"] = factors
    vector_metadata["neutral_kl_ratios_from_exact_v4b_receipt"] = (
        FIXED_NEUTRAL_KL_RATIOS
    )
    torch.save(vectors, run_dir / "vectors.pt")
    v4.write_json(run_dir / "vector-metadata.json", vector_metadata)

    combined_path = run_dir / "validation-records.jsonl"
    with combined_path.open("w", encoding="utf-8") as sink:
        for row in clean_rows:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
        sink.flush()
        rows = v4.append_interventions(
            tokenizer, model, cases, ids, clean_rows, vectors, factors, sink
        )
    base_summary = base.summarize_validation(rows)
    enhanced = v4.enhanced_analysis(rows)
    promotion = v4a.posttarget_analysis(rows, base_summary)
    summary = {
        "status": "complete_no_posttarget_retry_allowed",
        "run_dir": str(run_dir.relative_to(PROJECT)),
        "classification": "user_authorized_pretarget_conservative_control_repair",
        "claim_ceiling": promotion["claim_ceiling"],
        "adaptive_clean_gate": adaptive_gate,
        "fixed_neutral_control_receipt": v4c_preflight,
        "base_summary": base_summary,
        "enhanced_analysis": enhanced,
        "promotion_analysis": promotion,
        "no_further_retry_on_this_cohort": True,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    summary_path = run_dir / "summary.json"
    v4.write_json(summary_path, summary)
    v4.write_json(
        run_dir / "custody.json",
        {
            "v4b_runner_sha256": sha256_file(
                ROOT / "repaired_semantic_choice_resume_v4b_20260816.py"
            ),
            "v4c_runner_sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "v4c_amendment_sha256": sha256_file(amendment_path),
            "v4b_stopped_summary_sha256": sha256_file(
                stopped_run / "summary.json"
            ),
            "artifact_custody_preflight_sha256": sha256_file(
                run_dir / "artifact-custody-preflight.json"
            ),
            "adaptive_clean_gate_sha256": sha256_file(
                run_dir / "adaptive-clean-gate.json"
            ),
            "fixed_neutral_control_receipt_sha256": sha256_file(
                run_dir / "fixed-neutral-control-receipt.json"
            ),
            "vectors_sha256": sha256_file(run_dir / "vectors.pt"),
            "vector_metadata_sha256": sha256_file(
                run_dir / "vector-metadata.json"
            ),
            "combined_records_sha256": sha256_file(combined_path),
            "summary_sha256": sha256_file(summary_path),
        },
    )
    print(run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
