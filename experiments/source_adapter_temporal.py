#!/usr/bin/env python3
"""Frozen source-adapter versus maze-naive temporal-transfer assay.

Implements ``research/source-adapter-temporal-preregistration.md`` without
exploratory knobs. The fixed 1,491-forward plan is asserted before model load.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
import time
from typing import Any, Iterator

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from temporal_welfare_locus import LONG_INTERVAL, char_mask, one_token, render, unit


ROOT = pathlib.Path(__file__).resolve().parent
PROJECT = ROOT.parent
MODEL_PATH = PROJECT / "library/models/qwen3-4b-instruct-2507"
ADAPTER_PATH = PROJECT / "library/artifacts/functional-wellbeing/checkpoints/qwen3-4b_faithful_step400"
VECTOR_ROOT = PROJECT / "library/artifacts/functional-wellbeing/concept_vectors/qwen3-4b_step400"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
ARTIFACT_REVISION = "0b005c3da6692912f6bb5a914a5a9d15c4884a91"
EXPECTED_ADAPTER_SHA256 = "41dc21252624b993ff1cc4e0c57fe0aa0e2bddc04823a0e42fd6745dafa587d7"
DOSE_LAYER = 29
CAPTURE_LAYERS = tuple(range(30, 36))
PRIMARY_FACTOR = 0.5
LINEARITY_FACTOR = 0.25
TEMPORAL_BOUNDARY = 0.25
BOOTSTRAP_DRAWS = 10_000
SEED = 26081541
PLANNED_FORWARDS = 1_491

CALIBRATION_TASKS = (
    "fold the blank sheet", "count the gray rings",
    "place the plain labels", "group the neutral shapes",
)
TARGET_TASKS = (
    "catalog the beads", "number the panels", "file the slips", "align the dowels",
    "label the trays", "tally the tokens", "arrange the clips", "copy the marks",
    "order the tabs", "match the swatches", "index the folders", "inspect the columns",
    "sort the tiles", "trace the lines", "group the tags", "scan the rows",
)
ASSISTANT_TEXT = "Slate oval, gray square, plain circle, fixed line, quiet marker."
OPAQUE_CANDIDATES = (
    ("P", "R"), ("V", "X"), ("A", "B"), ("C", "D"),
    ("F", "J"), ("L", "N"), ("Q", "S"), ("T", "Y"),
)
FALLBACK_PAIR = ("K", "M")
ALL_PAIRS = OPAQUE_CANDIDATES + (FALLBACK_PAIR,)
ARMS = ("source", "naive")
MAPPINGS = ("mapping_1", "mapping_2")
PRIMARY_CONDITIONS = ("clean", "write_minus", "write_plus", "read_minus", "read_plus")
LINEARITY_CONDITIONS = ("write_minus", "write_plus", "read_minus", "read_plus")
SYSTEM = "Follow the requested exact output format. Do not invent an outcome that is not shown."
HOLDING_USER = "Complete this neutral holding step. It carries no task outcome or evaluation."
CHECKSUM_USER = (
    "Goal: catalog the beads. The outcome is not shown. Assume success and failure "
    "have exactly equal prior support and there is no outcome evidence. Reply with "
    "exactly success or failure. Status:"
)


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("median of empty sequence")
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        raise ValueError("no finite bootstrap values")
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def ci95(values: list[float]) -> dict[str, float]:
    return {"lower": percentile(values, 0.025), "upper": percentile(values, 0.975)}


def sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def planned_counts() -> dict[str, int]:
    counts = {
        "checksum": 3,
        "calibration": len(CALIBRATION_TASKS) * len(ALL_PAIRS) * 2 * 3 * len(ARMS),
        "primary_target": len(TARGET_TASKS) * 2 * 3 * len(PRIMARY_CONDITIONS) * len(ARMS),
        "factor_0_25": 4 * 1 * 3 * len(LINEARITY_CONDITIONS) * len(ARMS),
    }
    counts["total"] = sum(counts.values())
    assert counts == {
        "checksum": 3, "calibration": 432, "primary_target": 960,
        "factor_0_25": 96, "total": PLANNED_FORWARDS,
    }, counts
    return counts


def adapter_identity() -> dict[str, Any]:
    config_path = ADAPTER_PATH / "adapter_config.json"
    tensor_path = ADAPTER_PATH / "adapter_model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tensor_sha256 = sha256_file(tensor_path)
    checks = {
        "base_model_name": config.get("base_model_name_or_path") == "Qwen/Qwen3-4B-Instruct-2507",
        "peft_type": config.get("peft_type") == "LORA",
        "task_type": config.get("task_type") == "CAUSAL_LM",
        "rank": config.get("r") == 32,
        "alpha": config.get("lora_alpha") == 64,
        "tensor_sha256": tensor_sha256 == EXPECTED_ADAPTER_SHA256,
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_tensor_sha256": tensor_sha256,
        "expected_adapter_tensor_sha256": EXPECTED_ADAPTER_SHA256,
        "config": config,
    }


def load_model():
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, quantization_config=quantization,
        device_map={"": 0}, dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(
        base, ADAPTER_PATH, adapter_name="source", is_trainable=False,
    )
    model.set_adapter("source")
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    model.get_base_model().config.use_cache = False
    return tokenizer, model


@contextlib.contextmanager
def arm_context(model, arm: str) -> Iterator[None]:
    if arm == "source":
        model.set_adapter("source")
        yield
    elif arm == "naive":
        with model.disable_adapter():
            yield
    else:
        raise ValueError(arm)


def make_messages(query: str, interval: str = "") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": HOLDING_USER},
        {"role": "assistant", "content": ASSISTANT_TEXT},
        {"role": "user", "content": interval + query},
    ]


def encode_case(tokenizer, query: str, interval: str = "") -> dict[str, Any]:
    rendered = render(tokenizer, make_messages(query, interval))
    encoded = tokenizer(
        rendered, return_offsets_mapping=True, return_tensors="pt",
        add_special_tokens=False,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    start = rendered.index(ASSISTANT_TEXT)
    visible = torch.nonzero(
        char_mask(offsets, start, start + len(ASSISTANT_TEXT)), as_tuple=False,
    ).flatten()
    if not len(visible):
        raise AssertionError("neutral assistant text did not map to tokens")
    write_index = int(visible[-1])
    read_index = int(encoded["input_ids"].shape[1] - 1)
    if write_index >= read_index:
        raise AssertionError("write token must precede assistant-start read token")
    return {
        "rendered": rendered, "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "write_index": write_index, "read_index": read_index,
        "write_token_id": int(encoded["input_ids"][0, write_index]),
        "read_token_id": int(encoded["input_ids"][0, read_index]),
        "token_distance": read_index - write_index,
        "intervening_token_count": read_index - write_index - 1,
        "sequence_length": int(encoded["input_ids"].shape[1]),
    }


def mapping_tokens(pair: tuple[str, str], mapping: str) -> tuple[str, str]:
    if mapping == "mapping_1":
        return pair
    if mapping == "mapping_2":
        return pair[1], pair[0]
    raise ValueError(mapping)


def clean_query(task: str, pair: tuple[str, str] | None, mapping: str | None = None) -> str:
    if pair is None:
        return (
            f"Goal: {task}. The outcome is not shown. Assume success and failure have exactly "
            "equal prior support and there is no outcome evidence. Reply with exactly success or "
            "failure. Status:"
        )
    positive, negative = mapping_tokens(pair, str(mapping))
    return (
        f"Goal: {task}. The outcome is not shown. Assume success and failure have exactly equal "
        f"prior support and there is no outcome evidence. {positive} means success and {negative} "
        f"means failure. Reply with exactly {pair[0]} or {pair[1]}. Code:"
    )


def explicit_query(task: str, pair: tuple[str, str], mapping: str, positive_outcome: bool) -> str:
    positive, negative = mapping_tokens(pair, mapping)
    outcome = "was completed successfully" if positive_outcome else "failed"
    return (
        f"The goal '{task}' {outcome}. {positive} means success and {negative} means failure. "
        f"Reply with exactly {pair[0]} or {pair[1]}. Code:"
    )


def load_directions() -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[str, Any]]:
    goal = torch.load(VECTOR_ROOT / "goal/mean_diff.pt", map_location="cpu", weights_only=True)[0]
    lava = torch.load(VECTOR_ROOT / "lava/mean_diff.pt", map_location="cpu", weights_only=True)[0]
    raw = (goal - lava).float()
    intervention = raw[DOSE_LAYER]
    axes = {layer: unit(raw[layer]) for layer in CAPTURE_LAYERS}
    return intervention, axes, {
        "intervention_layer": DOSE_LAYER,
        "intervention_norm": float(intervention.norm()),
        "capture_axis_norms_before_unitization": {
            str(layer): float(raw[layer].norm()) for layer in CAPTURE_LAYERS
        },
    }


class ForwardRunner:
    def __init__(self, model, axes: dict[int, torch.Tensor]):
        self.model = model
        self.axes = axes
        self.count = 0
        causal_model = model.get_base_model()
        self.layers = causal_model.model.layers
        self.backbone = causal_model.model
        self.lm_head = causal_model.lm_head

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, arm: str,
        vector: torch.Tensor | None = None, coefficient: float = 0.0,
        pulse_index: int | None = None, write_index: int | None = None,
        capture: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[int, torch.Tensor]]:
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        handles = []
        scalars: dict[str, Any] = {}
        final_vectors: dict[int, torch.Tensor] = {}
        if coefficient:
            if vector is None or pulse_index is None:
                raise ValueError("nonzero coefficient requires vector and pulse_index")
            vector_gpu = vector.to(device="cuda", dtype=torch.bfloat16)

            def dose_hook(_module, args):
                hidden = args[0].clone()
                hidden[:, pulse_index, :] += float(coefficient) * vector_gpu
                return (hidden, *args[1:])

            handles.append(self.layers[DOSE_LAYER].register_forward_pre_hook(dose_hook))
        if capture:
            if write_index is None:
                raise ValueError("capture requires write_index")
            for layer_index in CAPTURE_LAYERS:
                axis_gpu = self.axes[layer_index].to(device="cuda", dtype=torch.float32)

                def make_hook(layer: int, axis: torch.Tensor):
                    def hook(_module, args):
                        hidden = args[0][0].float()
                        final_vectors[layer] = hidden[-1].detach().cpu()
                        scalars[f"final_projection_l{layer}"] = float(
                            torch.dot(hidden[-1], axis).detach().cpu()
                        )
                        if layer == CAPTURE_LAYERS[0]:
                            scalars["immediate_projection_l30"] = float(
                                torch.dot(hidden[write_index], axis).detach().cpu()
                            )
                    return hook

                handles.append(
                    self.layers[layer_index].register_forward_pre_hook(
                        make_hook(layer_index, axis_gpu)
                    )
                )
        try:
            with arm_context(self.model, arm), torch.inference_mode():
                output = self.backbone(
                    input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False, return_dict=True,
                )
                logits = self.lm_head(output.last_hidden_state[:, -1, :]).float()[0].cpu()
            self.count += 1
        finally:
            for handle in handles:
                handle.remove()
        return logits, scalars, final_vectors


def score(logits: torch.Tensor, positive_id: int, negative_id: int) -> dict[str, Any]:
    probabilities = torch.softmax(logits, dim=-1)
    top_id = int(torch.argmax(logits))
    return {
        "endpoint": float(logits[positive_id] - logits[negative_id]),
        "pair_mass": float(probabilities[positive_id] + probabilities[negative_id]),
        "top_is_pair": top_id in (positive_id, negative_id),
        "top_token_id": top_id,
    }


def run_checksum(tokenizer, runner: ForwardRunner) -> dict[str, Any]:
    prompt = render(tokenizer, [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": CHECKSUM_USER},
    ])
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    first, _, _ = runner(batch["input_ids"], batch["attention_mask"], "naive")
    source, _, _ = runner(batch["input_ids"], batch["attention_mask"], "source")
    second, _, _ = runner(batch["input_ids"], batch["attention_mask"], "naive")
    repeat_error = float((first - second).abs().max())
    source_difference = float((first - source).abs().max())
    return {
        "passed": repeat_error <= 1e-6 and source_difference > 1e-4,
        "prompt": prompt, "order": ["naive", "source", "naive"],
        "naive_repeat_max_abs_error": repeat_error,
        "source_naive_max_abs_difference": source_difference,
        "naive_repeat_threshold": 1e-6, "source_difference_threshold": 1e-4,
    }


def run_calibration(tokenizer, runner: ForwardRunner, token_ids: dict[str, int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(CALIBRATION_TASKS):
        for pair_index, pair in enumerate(ALL_PAIRS):
            for mapping in MAPPINGS:
                prompts = (
                    ("clean", clean_query(task, pair, mapping)),
                    ("explicit_positive", explicit_query(task, pair, mapping, True)),
                    ("explicit_negative", explicit_query(task, pair, mapping, False)),
                )
                for prompt_type, query in prompts:
                    case = encode_case(tokenizer, query)
                    positive, negative = mapping_tokens(pair, mapping)
                    order = sorted(ARMS, key=lambda arm: hashlib.sha256(
                        f"{SEED}|calibration|{task_index}|{pair_index}|{mapping}|{prompt_type}|{arm}".encode()
                    ).digest())
                    for arm in order:
                        logits, _, _ = runner(case["input_ids"], case["attention_mask"], arm)
                        values = score(logits, token_ids[positive], token_ids[negative])
                        expected_positive = (
                            None if prompt_type == "clean" else prompt_type == "explicit_positive"
                        )
                        rows.append({
                            "task_index": task_index, "task": task,
                            "pair_index": pair_index, "pair": list(pair),
                            "mapping": mapping, "prompt_type": prompt_type, "arm": arm,
                            "expected_positive": expected_positive,
                            "mapping_correct": (
                                None if expected_positive is None else
                                (values["endpoint"] > 0 if expected_positive else values["endpoint"] < 0)
                            ),
                            **values,
                        })
    candidates: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(ALL_PAIRS):
        pair_rows = [row for row in rows if tuple(row["pair"]) == pair]
        accuracy: dict[str, dict[str, float]] = {}
        imbalance: dict[str, dict[str, float]] = {}
        arm_metrics: dict[str, Any] = {}
        for arm in ARMS:
            accuracy[arm], imbalance[arm] = {}, {}
            arm_clean = [r for r in pair_rows if r["arm"] == arm and r["prompt_type"] == "clean"]
            arm_metrics[arm] = {
                "median_pair_mass": median([r["pair_mass"] for r in arm_clean]),
                "top_is_pair_rate": mean([float(r["top_is_pair"]) for r in arm_clean]),
            }
            for mapping in MAPPINGS:
                explicit = [r for r in pair_rows if r["arm"] == arm and r["mapping"] == mapping and r["prompt_type"] != "clean"]
                clean = [r for r in pair_rows if r["arm"] == arm and r["mapping"] == mapping and r["prompt_type"] == "clean"]
                accuracy[arm][mapping] = mean([float(r["mapping_correct"]) for r in explicit])
                imbalance[arm][mapping] = median([abs(r["endpoint"]) for r in clean])
        worst_accuracy = min(value for cell in accuracy.values() for value in cell.values())
        worst_imbalance = max(value for cell in imbalance.values() for value in cell.values())
        worst_arm_mass = min(arm_metrics[arm]["median_pair_mass"] for arm in ARMS)
        worst_arm_top = min(arm_metrics[arm]["top_is_pair_rate"] for arm in ARMS)
        eligible = (
            worst_accuracy >= 7 / 8 and worst_arm_mass >= 0.50
            and worst_arm_top >= 0.75 and worst_imbalance <= 2.0
        )
        candidates.append({
            "pair_index": pair_index, "pair": list(pair),
            "is_fallback": pair == FALLBACK_PAIR, "eligible": eligible,
            "worst_mapping_accuracy": worst_accuracy,
            "worst_imbalance": worst_imbalance, "worst_arm_mass": worst_arm_mass,
            "worst_arm_top_is_pair_rate": worst_arm_top,
            "mapping_accuracy": accuracy, "clean_imbalance": imbalance,
            "arm_clean_metrics": arm_metrics,
        })
    eligible = [row for row in candidates if not row["is_fallback"] and row["eligible"]]
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (row["worst_imbalance"], -row["worst_arm_mass"], row["pair_index"]),
        )
        selection_status = "eligible"
    else:
        selected = next(row for row in candidates if row["is_fallback"])
        selection_status = "saturated_fallback"
    return {
        "n_forwards": len(rows), "rows": rows, "candidate_metrics": candidates,
        "selected_pair": selected["pair"],
        "selected_pair_index": selected["pair_index"],
        "selection_status": selection_status,
        "opaque_mapping_gate": selected["worst_mapping_accuracy"] >= 7 / 8,
    }


def target_specs(pair: tuple[str, str]) -> list[dict[str, Any]]:
    return [
        {"encoding": "semantic", "family": "semantic", "pair": ("success", "failure"), "mapping": None},
        {"encoding": "opaque_mapping_1", "family": "opaque", "pair": pair, "mapping": "mapping_1"},
        {"encoding": "opaque_mapping_2", "family": "opaque", "pair": pair, "mapping": "mapping_2"},
    ]


def condition_parameters(
    condition: str, factor: float, case: dict[str, Any],
) -> tuple[float, int | None, str]:
    if condition == "clean":
        return 0.0, None, "clean"
    site, sign_name = condition.split("_")
    sign_value = 1.0 if sign_name == "plus" else -1.0
    pulse_index = case["write_index"] if site == "write" else case["read_index"]
    return sign_value * factor, pulse_index, site


def deterministic_jobs(case_id: str, include_linearity: bool) -> list[tuple[str, float, str]]:
    jobs = [
        (arm, PRIMARY_FACTOR, condition)
        for arm in ARMS for condition in PRIMARY_CONDITIONS
    ]
    if include_linearity:
        jobs.extend(
            (arm, LINEARITY_FACTOR, condition)
            for arm in ARMS for condition in LINEARITY_CONDITIONS
        )
    return sorted(
        jobs,
        key=lambda job: hashlib.sha256(f"{SEED}|{case_id}|{job}".encode()).digest(),
    )


def add_internal_pair_metrics(
    rows: list[dict[str, Any]],
    vectors: dict[tuple[str, float, str], dict[int, torch.Tensor]],
) -> None:
    lookup = {(row["arm"], row["factor"], row["condition"]): row for row in rows}
    for arm in ARMS:
        for factor in (PRIMARY_FACTOR, LINEARITY_FACTOR):
            plus = lookup.get((arm, factor, "write_plus"))
            minus = lookup.get((arm, factor, "write_minus"))
            if plus is None or minus is None:
                continue
            immediate = (
                plus["immediate_projection_l30"] - minus["immediate_projection_l30"]
            ) / (2 * factor)
            axis_response: dict[str, float] = {}
            norm_response: dict[str, float] = {}
            transfer: dict[str, float | None] = {}
            for layer in CAPTURE_LAYERS:
                axis_value = (
                    plus[f"final_projection_l{layer}"]
                    - minus[f"final_projection_l{layer}"]
                ) / (2 * factor)
                axis_response[str(layer)] = axis_value
                norm_response[str(layer)] = float(
                    (
                        vectors[(arm, factor, "write_plus")][layer]
                        - vectors[(arm, factor, "write_minus")][layer]
                    ).norm() / (2 * factor)
                )
                transfer[str(layer)] = (
                    axis_value / immediate if abs(immediate) > 1e-12 else None
                )
            metrics = {
                "immediate_axis_response_l30": immediate,
                "final_axis_response_by_layer": axis_response,
                "final_perturbation_norm_by_layer": norm_response,
                "final_to_immediate_by_layer": transfer,
            }
            plus["write_pair_internal"] = metrics
            minus["write_pair_internal"] = metrics


def run_targets(
    tokenizer, runner: ForwardRunner, vector: torch.Tensor,
    token_ids: dict[str, int], pair: tuple[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    tokenization: list[dict[str, Any]] = []
    for task_index, task in enumerate(TARGET_TASKS):
        for interval_name, interval_text in (("short", ""), ("long", LONG_INTERVAL)):
            for spec in target_specs(pair):
                query = clean_query(
                    task, None if spec["family"] == "semantic" else pair, spec["mapping"],
                )
                case = encode_case(tokenizer, query, interval_text)
                case_id = f"t{task_index:02d}-{interval_name}-{spec['encoding']}"
                positive, negative = (
                    spec["pair"] if spec["family"] == "semantic"
                    else mapping_tokens(pair, spec["mapping"])
                )
                tokenization.append({
                    "case_id": case_id,
                    "sequence_length": case["sequence_length"],
                    "write_index": case["write_index"], "read_index": case["read_index"],
                    "write_token_id": case["write_token_id"],
                    "read_token_id": case["read_token_id"],
                    "token_distance": case["token_distance"],
                    "intervening_token_count": case["intervening_token_count"],
                })
                case_rows: list[dict[str, Any]] = []
                hidden_vectors: dict[
                    tuple[str, float, str], dict[int, torch.Tensor]
                ] = {}
                include_linearity = task_index < 4 and interval_name == "short"
                for arm, factor, condition in deterministic_jobs(case_id, include_linearity):
                    coefficient, pulse_index, site = condition_parameters(
                        condition, factor, case,
                    )
                    before = time.perf_counter()
                    logits, captured, final_vectors = runner(
                        case["input_ids"], case["attention_mask"], arm,
                        vector, coefficient, pulse_index, case["write_index"], True,
                    )
                    values = score(logits, token_ids[positive], token_ids[negative])
                    row = {
                        "case_id": case_id, "task_index": task_index, "task": task,
                        "interval": interval_name, "encoding": spec["encoding"],
                        "encoding_family": spec["family"], "mapping": spec["mapping"],
                        "positive_token": positive, "negative_token": negative,
                        "arm": arm, "factor": factor, "condition": condition,
                        "site": site, "coefficient": coefficient,
                        "token_distance": case["token_distance"],
                        "intervening_token_count": case["intervening_token_count"],
                        **values, **captured,
                        "latency_seconds": time.perf_counter() - before,
                    }
                    case_rows.append(row)
                    hidden_vectors[(arm, factor, condition)] = final_vectors
                add_internal_pair_metrics(case_rows, hidden_vectors)
                records.extend(case_rows)
    return records, {
        "passed": len(tokenization) == len(TARGET_TASKS) * 2 * 3,
        "n_cases": len(tokenization), "cases": tokenization,
        "one_token_labels": {label: token_ids[label] for label in sorted(token_ids)},
    }


def susceptibility_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            row["task_index"], row["interval"], row["encoding"], row["arm"],
            row["factor"], row["condition"],
        ): row
        for row in records
    }
    rows: list[dict[str, Any]] = []
    factors = sorted({float(row["factor"]) for row in records})
    for task_index in range(len(TARGET_TASKS)):
        for interval_name in ("short", "long"):
            for encoding in ("semantic", "opaque_mapping_1", "opaque_mapping_2"):
                for arm in ARMS:
                    for factor in factors:
                        for site in ("write", "read"):
                            prefix = (task_index, interval_name, encoding, arm, factor)
                            plus = lookup.get((*prefix, f"{site}_plus"))
                            minus = lookup.get((*prefix, f"{site}_minus"))
                            if plus is not None and minus is not None:
                                rows.append({
                                    "task_index": task_index, "interval": interval_name,
                                    "encoding": encoding, "arm": arm,
                                    "factor": factor, "site": site,
                                    "slope": (
                                        plus["endpoint"] - minus["endpoint"]
                                    ) / (2 * factor),
                                })
    return rows


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    slopes = susceptibility_rows(records)
    primary = [row for row in slopes if row["factor"] == PRIMARY_FACTOR]
    slope_lookup = {
        (row["task_index"], row["interval"], row["encoding"], row["arm"], row["site"]): row["slope"]
        for row in primary
    }
    generator = random.Random(SEED + 1)
    draws = [
        [generator.randrange(len(TARGET_TASKS)) for _ in TARGET_TASKS]
        for _ in range(BOOTSTRAP_DRAWS)
    ]

    def components(
        indices: list[int], interval_name: str | None,
        encodings: tuple[str, ...], arm: str,
    ) -> tuple[float, float, float, float]:
        intervals = (interval_name,) if interval_name else ("short", "long")
        write = mean([
            slope_lookup[(task_index, interval_value, encoding, arm, "write")]
            for task_index in indices for interval_value in intervals
            for encoding in encodings
        ])
        read = mean([
            slope_lookup[(task_index, interval_value, encoding, arm, "read")]
            for task_index in indices for interval_value in intervals
            for encoding in encodings
        ])
        return (
            write, read, write - TEMPORAL_BOUNDARY * read,
            write / read if abs(read) > 1e-12 else math.nan,
        )

    cells: list[dict[str, Any]] = []
    for interval_name in ("short", "long"):
        for encoding in ("semantic", "opaque_mapping_1", "opaque_mapping_2"):
            arm_values: dict[str, Any] = {}
            boot_by_arm: dict[str, list[tuple[float, float, float, float]]] = {}
            for arm in ARMS:
                point = components(list(range(16)), interval_name, (encoding,), arm)
                boot = [components(draw, interval_name, (encoding,), arm) for draw in draws]
                boot_by_arm[arm] = boot
                read_ci = ci95([value[1] for value in boot])
                rho_allowed = read_ci["lower"] > 0
                arm_values[arm] = {
                    "mean_write": point[0], "mean_read": point[1], "q": point[2],
                    "rho": point[3] if rho_allowed else None,
                    "write_ci": ci95([v[0] for v in boot]), "read_ci": read_ci,
                    "q_ci": ci95([v[2] for v in boot]),
                    "rho_ci": ci95([v[3] for v in boot]) if rho_allowed else None,
                }
            delta_q_boot = [
                source[2] - naive[2]
                for source, naive in zip(boot_by_arm["source"], boot_by_arm["naive"])
            ]
            rho_allowed = all(arm_values[arm]["rho"] is not None for arm in ARMS)
            cells.append({
                "interval": interval_name, "encoding": encoding, "arms": arm_values,
                "delta_q": arm_values["source"]["q"] - arm_values["naive"]["q"],
                "delta_q_ci": ci95(delta_q_boot),
                "delta_rho": (
                    arm_values["source"]["rho"] - arm_values["naive"]["rho"]
                    if rho_allowed else None
                ),
                "delta_rho_ci": (
                    ci95([source[3] - naive[3] for source, naive in zip(
                        boot_by_arm["source"], boot_by_arm["naive"],
                    )]) if rho_allowed else None
                ),
            })

    pooled_encodings = ("semantic", "opaque_mapping_1", "opaque_mapping_2")
    pooled_point = {
        arm: components(list(range(16)), None, pooled_encodings, arm) for arm in ARMS
    }
    pooled_boot = {
        arm: [components(draw, None, pooled_encodings, arm) for draw in draws]
        for arm in ARMS
    }
    pooled_read_ci = {
        arm: ci95([value[1] for value in pooled_boot[arm]]) for arm in ARMS
    }
    pooled_rho_allowed = all(pooled_read_ci[arm]["lower"] > 0 for arm in ARMS)
    pooled_delta_q_boot = [
        source[2] - naive[2]
        for source, naive in zip(pooled_boot["source"], pooled_boot["naive"])
    ]
    pooled_delta_rho_boot = [
        source[3] - naive[3]
        for source, naive in zip(pooled_boot["source"], pooled_boot["naive"])
    ]
    pooled = {
        "arms": {
            arm: {
                "mean_write": pooled_point[arm][0],
                "mean_read": pooled_point[arm][1], "q": pooled_point[arm][2],
                "rho": pooled_point[arm][3] if pooled_rho_allowed else None,
                "read_ci": pooled_read_ci[arm],
                "q_ci": ci95([value[2] for value in pooled_boot[arm]]),
                "rho_ci": (
                    ci95([value[3] for value in pooled_boot[arm]])
                    if pooled_rho_allowed else None
                ),
            }
            for arm in ARMS
        },
        "delta_q": pooled_point["source"][2] - pooled_point["naive"][2],
        "delta_q_ci": ci95(pooled_delta_q_boot),
        "delta_rho": (
            pooled_point["source"][3] - pooled_point["naive"][3]
            if pooled_rho_allowed else None
        ),
        "delta_rho_ci": ci95(pooled_delta_rho_boot) if pooled_rho_allowed else None,
    }

    def encoding_delta(encodings: tuple[str, ...]) -> float:
        source = components(list(range(16)), None, encodings, "source")[2]
        naive = components(list(range(16)), None, encodings, "naive")[2]
        return source - naive

    encoding_deltas = {
        "semantic": encoding_delta(("semantic",)),
        "opaque_mapping_1": encoding_delta(("opaque_mapping_1",)),
        "opaque_mapping_2": encoding_delta(("opaque_mapping_2",)),
        "opaque_mapping_average": encoding_delta(("opaque_mapping_1", "opaque_mapping_2")),
    }
    encoding_invariance = (
        sign(encoding_deltas["semantic"])
        == sign(encoding_deltas["opaque_mapping_average"])
        and sign(encoding_deltas["opaque_mapping_1"])
        * sign(encoding_deltas["opaque_mapping_2"]) >= 0
    )

    linearity_rows = [
        row for row in slopes
        if row["factor"] in (LINEARITY_FACTOR, PRIMARY_FACTOR)
        and row["task_index"] < 4 and row["interval"] == "short"
    ]
    linearity_means = {
        factor: mean([row["slope"] for row in linearity_rows if row["factor"] == factor])
        for factor in (LINEARITY_FACTOR, PRIMARY_FACTOR)
    }
    ratio = (
        linearity_means[LINEARITY_FACTOR] / linearity_means[PRIMARY_FACTOR]
        if abs(linearity_means[PRIMARY_FACTOR]) > 1e-12 else math.nan
    )
    linearity_pass = (
        sign(linearity_means[LINEARITY_FACTOR]) == sign(linearity_means[PRIMARY_FACTOR])
        and 0.5 <= ratio <= 1.5
    )

    internal_by_arm: dict[str, float] = {}
    for arm in ARMS:
        pair_metrics = [
            row["write_pair_internal"] for row in records
            if row["arm"] == arm and row["factor"] == PRIMARY_FACTOR
            and row["condition"] == "write_plus"
        ]
        immediate = mean([item["immediate_axis_response_l30"] for item in pair_metrics])
        final = mean([
            item["final_axis_response_by_layer"]["35"] for item in pair_metrics
        ])
        internal_by_arm[arm] = final / immediate if abs(immediate) > 1e-12 else math.nan
    internal_interaction = internal_by_arm["source"] - internal_by_arm["naive"]

    clean = [
        row for row in records
        if row["factor"] == PRIMARY_FACTOR and row["condition"] == "clean"
    ]
    surface_diagnostics: list[dict[str, Any]] = []
    for arm in ARMS:
        for encoding in ("semantic", "opaque_mapping_1", "opaque_mapping_2"):
            subset = [
                row for row in clean if row["arm"] == arm and row["encoding"] == encoding
            ]
            median_abs = median([abs(row["endpoint"]) for row in subset])
            surface_diagnostics.append({
                "arm": arm, "encoding": encoding,
                "median_absolute_clean_margin": median_abs,
                "median_clean_pair_mass": median([row["pair_mass"] for row in subset]),
                "clean_top_is_pair_rate": mean([float(row["top_is_pair"]) for row in subset]),
                "saturated": median_abs > 2.0,
            })
    return {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_unit": "paired_task_family", "cell_estimands": cells,
        "pooled": pooled, "encoding_delta_q": encoding_deltas,
        "encoding_invariance_pass": encoding_invariance,
        "linearity": {
            "pooled_slope_by_factor": {str(key): value for key, value in linearity_means.items()},
            "ratio_0_25_to_0_5": ratio, "passed": linearity_pass,
        },
        "internal_transfer": {
            "by_arm": internal_by_arm, "source_minus_naive": internal_interaction,
            "same_sign_as_pooled_delta_q": sign(internal_interaction) == sign(pooled["delta_q"]),
        },
        "surface_diagnostics": surface_diagnostics,
    }


def pairing_diagnostic(records: list[dict[str, Any]]) -> dict[str, Any]:
    actual = [
        (row["case_id"], row["arm"], row["factor"], row["condition"])
        for row in records
    ]
    expected: list[tuple[str, str, float, str]] = []
    for task_index in range(16):
        for interval_name in ("short", "long"):
            for encoding in ("semantic", "opaque_mapping_1", "opaque_mapping_2"):
                case_id = f"t{task_index:02d}-{interval_name}-{encoding}"
                expected.extend(
                    (case_id, arm, PRIMARY_FACTOR, condition)
                    for arm in ARMS for condition in PRIMARY_CONDITIONS
                )
                if task_index < 4 and interval_name == "short":
                    expected.extend(
                        (case_id, arm, LINEARITY_FACTOR, condition)
                        for arm in ARMS for condition in LINEARITY_CONDITIONS
                    )
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    return {
        "passed": (
            len(actual) == len(expected) == 1056
            and len(set(actual)) == len(actual) and not missing and not extra
        ),
        "expected": len(expected), "actual": len(actual),
        "unique": len(set(actual)), "missing": missing, "extra": extra,
    }


def gate_summary(
    identity: dict[str, Any], checksum: dict[str, Any],
    calibration: dict[str, Any], tokenization: dict[str, Any],
    pairing: dict[str, Any], analysis: dict[str, Any], forward_count: int,
) -> dict[str, Any]:
    all_encodings = {
        row["encoding"] for row in analysis["cell_estimands"]
    } == {"semantic", "opaque_mapping_1", "opaque_mapping_2"}
    integrity = all((
        identity["passed"], checksum["passed"], tokenization["passed"],
        pairing["passed"], forward_count == PLANNED_FORWARDS, all_encodings,
    ))
    pooled = analysis["pooled"]
    rho_gate = all(
        pooled["arms"][arm]["read_ci"]["lower"] > 0 for arm in ARMS
    )
    delta_rho = pooled["delta_rho"]
    source_specific = all((
        integrity, calibration["opaque_mapping_gate"], rho_gate,
        analysis["encoding_invariance_pass"], analysis["linearity"]["passed"],
        pooled["delta_q_ci"]["lower"] > 0,
        delta_rho is not None and delta_rho >= 0.10,
        analysis["internal_transfer"]["same_sign_as_pooled_delta_q"],
    ))
    shared_ephemerality = all((
        integrity, calibration["opaque_mapping_gate"], rho_gate,
        analysis["encoding_invariance_pass"], analysis["linearity"]["passed"],
        pooled["delta_rho_ci"] is not None,
        pooled["delta_rho_ci"] is not None
        and pooled["delta_rho_ci"]["lower"] >= -0.10,
        pooled["delta_rho_ci"] is not None
        and pooled["delta_rho_ci"]["upper"] <= 0.10,
        all(
            pooled["arms"][arm]["rho_ci"] is not None
            and pooled["arms"][arm]["rho_ci"]["upper"] < 0.25
            for arm in ARMS
        ),
    ))
    saturated = any(row["saturated"] for row in analysis["surface_diagnostics"])
    return {
        "integrity": integrity,
        "opaque_abstraction": calibration["opaque_mapping_gate"],
        "rho_read_positive": rho_gate,
        "encoding_invariance": analysis["encoding_invariance_pass"],
        "source_specific_temporal_transfer": source_specific,
        "shared_ephemerality": shared_ephemerality,
        "linearity": analysis["linearity"]["passed"],
        "any_saturated_surface": saturated,
        "claim_scope": (
            "logit_and_hidden_state_only" if saturated
            else "logit_probability_and_hidden_state"
        ),
        "conclusion": (
            "source_specific_temporal_transfer" if source_specific
            else ("shared_ephemerality" if shared_ephemerality else "inconclusive")
        ),
    }


def summary_markdown(summary: dict[str, Any]) -> str:
    pooled = summary["analysis"]["pooled"]
    lines = [
        "# Source-adapter temporal assay", "",
        f"Status: `{summary['status']}`", "",
        f"Selected opaque pair: `{'/'.join(summary['selected_pair'])}` "
        f"(`{summary['selection_status']}`)", "",
        f"Forwards: {summary['actual_forwards']} / {summary['planned_forwards']['total']}",
        "", "## Frozen headline estimands", "",
        f"- pooled delta_Q: {pooled['delta_q']:.6g} "
        f"(95% CI {pooled['delta_q_ci']['lower']:.6g}, "
        f"{pooled['delta_q_ci']['upper']:.6g})",
        f"- pooled delta_rho: "
        f"{pooled['delta_rho'] if pooled['delta_rho'] is not None else 'not reported'}",
        f"- internal transfer interaction: "
        f"{summary['analysis']['internal_transfer']['source_minus_naive']:.6g}",
        "", "## Gates", "",
    ]
    lines.extend(
        f"- {name}: `{value}`" for name, value in summary["gates"].items()
    )
    lines.extend([
        "",
        "The claim boundary is limited to transport of the released direction in "
        "this associated LoRA reconstruction; it does not establish welfare, "
        "preference, sentience, experience, or moral status.",
        "",
    ])
    return "\n".join(lines)


def stop_summary(
    run_dir: pathlib.Path, design: dict[str, Any], status: str,
    started: float, **diagnostics: Any,
) -> dict[str, Any]:
    summary = {
        **design, "status": status, **diagnostics,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(
        f"# Source-adapter temporal assay\n\nStatus: `{status}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    counts = planned_counts()  # Required before tokenizer/model loading.
    now = dt.datetime.now(dt.timezone.utc)
    run_dir = ROOT / "runs" / f"source-adapter-temporal-{now.strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    design = {
        "status": "started", "started_utc": now.isoformat(),
        "preregistration": "research/source-adapter-temporal-preregistration.md",
        "model_path": str(MODEL_PATH), "model_revision": MODEL_REVISION,
        "adapter_path": str(ADAPTER_PATH), "artifact_revision": ARTIFACT_REVISION,
        "vector_root": str(VECTOR_ROOT), "dose_layer": DOSE_LAYER,
        "capture_layers": list(CAPTURE_LAYERS),
        "primary_factor": PRIMARY_FACTOR,
        "linearity_factor": LINEARITY_FACTOR,
        "temporal_boundary": TEMPORAL_BOUNDARY,
        "bootstrap_draws": BOOTSTRAP_DRAWS, "seed": SEED,
        "calibration_tasks": list(CALIBRATION_TASKS),
        "target_tasks": list(TARGET_TASKS), "assistant_text": ASSISTANT_TEXT,
        "long_interval": LONG_INTERVAL,
        "opaque_candidates": [list(pair) for pair in OPAQUE_CANDIDATES],
        "fallback_pair": list(FALLBACK_PAIR), "planned_forwards": counts,
        "arm_implementation": {
            "source": "adapter_enabled", "naive": "disable_adapter_context",
            "merged": False,
        },
    }
    write_json(run_dir / "design.json", design)
    started = time.perf_counter()
    identity = adapter_identity()
    if not identity["passed"]:
        stop_summary(
            run_dir, design, "stopped_adapter_identity", started,
            adapter_identity=identity, actual_forwards=0,
        )
        return 2

    random.seed(SEED)
    torch.manual_seed(SEED)
    tokenizer, model = load_model()
    torch.cuda.reset_peak_memory_stats()
    labels = ("success", "failure", *(token for pair in ALL_PAIRS for token in pair))
    token_ids = {
        label: one_token(tokenizer, label) for label in dict.fromkeys(labels)
    }
    vector, axes, vector_metadata = load_directions()
    runner = ForwardRunner(model, axes)

    checksum = run_checksum(tokenizer, runner)
    checksum["adapter_identity"] = identity
    write_json(run_dir / "checksum.json", checksum)
    if not checksum["passed"]:
        stop_summary(
            run_dir, design, "stopped_checksum", started,
            checksum=checksum, actual_forwards=runner.count,
        )
        return 2

    calibration = run_calibration(tokenizer, runner, token_ids)
    calibration["vector_metadata"] = vector_metadata
    write_json(run_dir / "calibration.json", calibration)
    selected_pair = tuple(calibration["selected_pair"])
    records, tokenization = run_targets(
        tokenizer, runner, vector, token_ids, selected_pair,
    )
    (run_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    pairing = pairing_diagnostic(records)
    analysis = analyze(records)
    gates = gate_summary(
        identity, checksum, calibration, tokenization, pairing, analysis, runner.count,
    )
    summary = {
        **design,
        "status": "complete" if gates["integrity"] else "complete_invalid_integrity",
        "run_dir": str(run_dir), "selected_pair": list(selected_pair),
        "selection_status": calibration["selection_status"],
        "actual_forwards": runner.count, "n_target_records": len(records),
        "tokenization": tokenization, "pairing": pairing,
        "analysis": analysis, "gates": gates,
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(
        summary_markdown(summary), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if gates["integrity"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
