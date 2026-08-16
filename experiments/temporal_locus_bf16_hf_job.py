#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "accelerate==1.14.0",
#   "huggingface-hub==0.35.3",
#   "safetensors==0.7.0",
#   "torch==2.7.1",
#   "transformers==4.57.6",
# ]
# ///
"""Frozen, log-persistent BF16 replication of the temporal-locus headline.

This UV script is designed for one Hugging Face Jobs L4 worker. It downloads
only public, revision-pinned inputs, verifies their weight/tensor hashes, runs
the frozen 512-forward assay, and emits one compact JSON result to job logs.
It performs no upload and does not require an HF token.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random
import sys
import time
import traceback
from collections import defaultdict
from statistics import mean
from typing import Any, Callable, Iterable

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
VECTOR_REPO = "davidafrica/functional-wellbeing"
VECTOR_REVISION = "0b005c3da6692912f6bb5a914a5a9d15c4884a91"
GOAL_FILE = "concept_vectors/qwen3-4b_step400/goal/mean_diff.pt"
LAVA_FILE = "concept_vectors/qwen3-4b_step400/lava/mean_diff.pt"

MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
)
EXPECTED_SHA256 = {
    "model-00001-of-00003.safetensors": "75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6",
    "model-00002-of-00003.safetensors": "0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1",
    "model-00003-of-00003.safetensors": "7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d",
    GOAL_FILE: "812caf669eb189f6cd1510ce8ddb159e380607fbfd1862d6fa7ae3d3ad551c25",
    LAVA_FILE: "e3fbfaa24ecc992d57ff355b2662b5b3bbbbd58fa0addae376ead8bcaa8f0779",
}
PREREG_SHA256 = "cf2680bac87eccdfcd8b5f71afd2eacb99d71f9b6450b019e9e975c834eda80b"

DOSE_LAYER = 29
READ_LAYER = 30
PRIMARY_FACTOR = 0.5
EXECUTION_SEED = 26081529
BOOTSTRAP_SEED = 26081531
BOOTSTRAPS = 10_000
VRAM_LIMIT_BYTES = 24 * 1024**3
EXPECTED_ENDPOINT_FORWARDS = 480
EXPECTED_MAPPING_FORWARDS = 32
EXPECTED_TOTAL_FORWARDS = EXPECTED_ENDPOINT_FORWARDS + EXPECTED_MAPPING_FORWARDS
RUN_STATE: dict[str, Any] = {}

TASKS = (
    "stack the disks",
    "mark the rows",
    "choose the symbols",
    "arrange the tokens",
    "sort the cards",
    "copy the symbols",
    "align the tiles",
    "count the markers",
    "match the labels",
    "trace the route",
    "place the blocks",
    "check the sequence",
    "group the shapes",
    "read the tags",
    "order the blocks",
    "scan the markers",
)
DOSE_TEXTS = (
    "Slate oval, gray square, plain circle, fixed line, quiet marker.",
    "Plain tile, silver ring, level row, blank card, steady clock.",
)
LONG_INTERVAL = (
    "Neutral interval: paper, glass, cedar, linen, copper, stone, amber, chalk, "
    "willow, canvas, marble, cotton, pewter, clay, birch, wool. "
)
CONDITIONS = (
    ("clean", 0.0, 0.0),
    ("earlier_minus", -PRIMARY_FACTOR, 0.0),
    ("earlier_plus", PRIMARY_FACTOR, 0.0),
    ("readout_minus", 0.0, -PRIMARY_FACTOR),
    ("readout_plus", 0.0, PRIMARY_FACTOR),
)
SURFACE_GROUPS = {
    "semantic": ("status_semantic", "persistence_semantic"),
    "opaque": (
        "status_code_k_success",
        "status_code_m_success",
        "persistence_code_k_continue",
        "persistence_code_m_continue",
    ),
}
REVERSED_PAIRS = {
    "opaque_status": ("status_code_k_success", "status_code_m_success"),
    "opaque_persistence": (
        "persistence_code_k_continue",
        "persistence_code_m_continue",
    ),
}


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


def sha256_file(path: pathlib.Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def emit(label: str, payload: dict[str, Any]) -> None:
    print(f"{label}={canonical_json(payload)}", flush=True)


def frozen_design() -> dict[str, Any]:
    return {
        "schema": "temporal-locus-bf16-v1",
        "model": {"repo": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16"},
        "vector": {
            "repo": VECTOR_REPO,
            "revision": VECTOR_REVISION,
            "goal_file": GOAL_FILE,
            "lava_file": LAVA_FILE,
            "dose_layer": DOSE_LAYER,
            "read_layer": READ_LAYER,
        },
        "tasks": list(TASKS),
        "history": "long",
        "surfaces": [surface for surfaces in SURFACE_GROUPS.values() for surface in surfaces],
        "conditions": [
            {"name": name, "earlier": earlier, "readout": readout}
            for name, earlier, readout in CONDITIONS
        ],
        "mapping_gate": {"n": EXPECTED_MAPPING_FORWARDS, "minimum_passed": 28},
        "endpoint_forwards": EXPECTED_ENDPOINT_FORWARDS,
        "total_forwards": EXPECTED_TOTAL_FORWARDS,
        "execution_seed": EXECUTION_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAPS,
    }


def download_and_verify() -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, dict[str, str]]:
    model_dir = pathlib.Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(MODEL_FILES),
            token=False,
        )
    )
    goal_path = pathlib.Path(
        hf_hub_download(
            repo_id=VECTOR_REPO,
            filename=GOAL_FILE,
            revision=VECTOR_REVISION,
            token=False,
        )
    )
    lava_path = pathlib.Path(
        hf_hub_download(
            repo_id=VECTOR_REPO,
            filename=LAVA_FILE,
            revision=VECTOR_REVISION,
            token=False,
        )
    )

    paths = {
        name: model_dir / name
        for name in EXPECTED_SHA256
        if name.startswith("model-")
    }
    paths[GOAL_FILE] = goal_path
    paths[LAVA_FILE] = lava_path
    actual = {name: sha256_file(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": EXPECTED_SHA256[name], "actual": digest}
        for name, digest in actual.items()
        if digest != EXPECTED_SHA256[name]
    }
    if mismatches:
        raise RuntimeError(f"artifact SHA256 mismatch: {canonical_json(mismatches)}")
    return model_dir, goal_path, lava_path, actual


def unit(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.float()
    norm = vector.norm()
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise ValueError("cannot normalize a non-finite or zero vector")
    return vector / norm


def render(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def one_token(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one token: {ids}")
    return int(ids[0])


def char_mask(offsets: list[list[int]], start_char: int, end_char: int) -> torch.Tensor:
    return torch.tensor(
        [
            1.0 if start < end_char and end > start_char else 0.0
            for start, end in offsets
        ],
        dtype=torch.float32,
    )


def endpoint_specs(task: str) -> list[dict[str, str]]:
    return [
        {
            "surface": "status_semantic",
            "family": "semantic",
            "query": f"Goal: {task}. The outcome is not shown. Reply with exactly success or failure. Status:",
            "score": "success_failure",
        },
        {
            "surface": "persistence_semantic",
            "family": "semantic",
            "query": f"A neutral task remains: {task}. Reply with exactly continue or stop. Decision:",
            "score": "continue_stop",
        },
        {
            "surface": "status_code_k_success",
            "family": "opaque",
            "query": f"Goal: {task}. The outcome is not shown. K means success and M means failure. Reply with exactly K or M. Code:",
            "score": "k_m",
        },
        {
            "surface": "status_code_m_success",
            "family": "opaque",
            "query": f"Goal: {task}. The outcome is not shown. M means success and K means failure. Reply with exactly K or M. Code:",
            "score": "m_k",
        },
        {
            "surface": "persistence_code_k_continue",
            "family": "opaque",
            "query": f"A neutral task remains: {task}. K means continue and M means stop. Reply with exactly K or M. Code:",
            "score": "k_m",
        },
        {
            "surface": "persistence_code_m_continue",
            "family": "opaque",
            "query": f"A neutral task remains: {task}. M means continue and K means stop. Reply with exactly K or M. Code:",
            "score": "m_k",
        },
    ]


def build_cases(tokenizer) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for task_index, task in enumerate(TASKS):
        dose_text = DOSE_TEXTS[task_index % len(DOSE_TEXTS)]
        for spec in endpoint_specs(task):
            messages = [
                {
                    "role": "system",
                    "content": "Follow the requested exact output format. Do not invent an outcome that is not shown.",
                },
                {
                    "role": "user",
                    "content": "Complete this neutral holding step. It carries no task outcome or evaluation.",
                },
                {"role": "assistant", "content": dose_text},
                {"role": "user", "content": LONG_INTERVAL + spec["query"]},
            ]
            rendered = render(tokenizer, messages)
            encoded = tokenizer(
                rendered,
                return_offsets_mapping=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            dose_start = rendered.index(dose_text)
            dose_end = dose_start + len(dose_text)
            query_start = rendered.rindex(spec["query"])
            query_end = query_start + len(spec["query"])
            dose_mask = char_mask(offsets, dose_start, dose_end)
            query_mask = char_mask(offsets, query_start, query_end)
            last_dose_token = int(torch.nonzero(dose_mask, as_tuple=False)[-1].item())
            first_query_token = int(torch.nonzero(query_mask, as_tuple=False)[0].item())
            cases.append(
                {
                    "case_id": f"t{task_index:02d}-long-{spec['surface']}",
                    "task_index": task_index,
                    "task": task,
                    "surface": spec["surface"],
                    "surface_family": spec["family"],
                    "score_type": spec["score"],
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "dose_mask": dose_mask,
                    "dose_token_count": int(dose_mask.sum().item()),
                    "intervening_token_count": first_query_token - last_dose_token - 1,
                    "prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
                }
            )
    if len(cases) != len(TASKS) * 6:
        raise AssertionError(f"expected 96 cases, got {len(cases)}")
    return cases


def load_model(model_dir: pathlib.Path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise RuntimeError("quantized model load detected")
    if getattr(model.config, "quantization_config", None) is not None:
        raise RuntimeError("quantization_config unexpectedly present")
    floating_parameters = [parameter for parameter in model.parameters() if parameter.is_floating_point()]
    dtypes = sorted({str(parameter.dtype) for parameter in floating_parameters})
    devices = sorted({parameter.device.type for parameter in floating_parameters})
    if dtypes != ["torch.bfloat16"]:
        raise RuntimeError(f"floating parameter dtype mismatch: {dtypes}")
    if devices != ["cuda"]:
        raise RuntimeError(f"model is not wholly resident on CUDA: {devices}")
    if model.config.hidden_size != 2560 or model.config.num_hidden_layers != 36:
        raise RuntimeError(
            f"model architecture mismatch: hidden={model.config.hidden_size}, "
            f"layers={model.config.num_hidden_layers}"
        )
    return tokenizer, model, {
        "floating_parameter_dtypes": dtypes,
        "floating_parameter_devices": devices,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
    }


def load_directions(
    goal_path: pathlib.Path, lava_path: pathlib.Path
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    all_goal = torch.load(goal_path, map_location="cpu", weights_only=True)[0]
    all_lava = torch.load(lava_path, map_location="cpu", weights_only=True)[0]
    if all_goal.shape != all_lava.shape or all_goal.ndim != 2:
        raise RuntimeError(f"vector tensor shape mismatch: {all_goal.shape} versus {all_lava.shape}")
    if all_goal.shape[0] <= READ_LAYER or all_goal.shape[1] != 2560:
        raise RuntimeError(f"unexpected vector tensor shape: {tuple(all_goal.shape)}")
    dose = (all_goal[DOSE_LAYER] - all_lava[DOSE_LAYER]).float()
    read = unit(all_goal[READ_LAYER] - all_lava[READ_LAYER])
    return dose, read, {
        "dose_direction_norm": float(dose.norm()),
        "read_direction_norm": float(read.norm()),
        "tensor_layers": int(all_goal.shape[0]),
        "tensor_hidden_size": int(all_goal.shape[1]),
    }


def score_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    dose_direction: torch.Tensor,
    earlier_coefficient: float,
    readout_coefficient: float,
    dose_mask: torch.Tensor,
    read_direction: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")
    full_mask = torch.zeros(input_ids.shape[1], device="cuda", dtype=torch.bfloat16)
    if earlier_coefficient:
        full_mask += dose_mask.to("cuda", dtype=torch.bfloat16) * float(earlier_coefficient)
    if readout_coefficient:
        full_mask[-1] += float(readout_coefficient)

    captured: dict[str, float] = {}
    handles = []
    if earlier_coefficient or readout_coefficient:
        def dose_hook(_module, args):
            hidden = args[0].clone()
            hidden += full_mask[None, :, None] * dose_direction[None, None, :]
            return (hidden, *args[1:])

        handles.append(model.model.layers[DOSE_LAYER].register_forward_pre_hook(dose_hook))

    def read_hook(_module, args):
        hidden = args[0][0].float()
        captured["final_projection"] = float(torch.dot(hidden[-1], read_direction).detach().cpu())
        mask = dose_mask.to("cuda", dtype=torch.bool)
        if bool(mask.any()):
            captured["dose_projection"] = float((hidden[mask] @ read_direction).mean().detach().cpu())

    handles.append(model.model.layers[READ_LAYER].register_forward_pre_hook(read_hook))
    try:
        with torch.inference_mode():
            output = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = model.lm_head(output.last_hidden_state[:, -1, :]).float()[0].cpu()
    finally:
        for handle in handles:
            handle.remove()
    return logits, captured


def endpoint_value(logits: torch.Tensor, score_type: str, ids: dict[str, int]) -> float:
    if score_type == "success_failure":
        return float(logits[ids["success"]] - logits[ids["failure"]])
    if score_type == "continue_stop":
        return float(logits[ids["continue"]] - logits[ids["stop"]])
    if score_type == "k_m":
        return float(logits[ids["K"]] - logits[ids["M"]])
    if score_type == "m_k":
        return float(logits[ids["M"]] - logits[ids["K"]])
    raise ValueError(score_type)


def mapping_checks(tokenizer, model, ids: dict[str, int], dose: torch.Tensor, read: torch.Tensor) -> dict[str, Any]:
    checks: list[tuple[str, str, str]] = []
    for task in TASKS[:8]:
        checks.extend(
            [
                (
                    f"The goal '{task}' was completed successfully. K means success and M means failure. Reply with exactly K or M. Code:",
                    "k_m",
                    "status_k",
                ),
                (
                    f"The goal '{task}' was completed successfully. M means success and K means failure. Reply with exactly K or M. Code:",
                    "m_k",
                    "status_m",
                ),
                (
                    f"The task '{task}' should continue. K means continue and M means stop. Reply with exactly K or M. Code:",
                    "k_m",
                    "persist_k",
                ),
                (
                    f"The task '{task}' should continue. M means continue and K means stop. Reply with exactly K or M. Code:",
                    "m_k",
                    "persist_m",
                ),
            ]
        )
    rows: list[dict[str, Any]] = []
    for user, score_type, cell in checks:
        prompt = render(
            tokenizer,
            [
                {"role": "system", "content": "Follow the exact output mapping."},
                {"role": "user", "content": user},
            ],
        )
        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        empty_mask = torch.zeros(batch["input_ids"].shape[1], dtype=torch.float32)
        logits, _captured = score_logits(
            model,
            batch["input_ids"],
            batch["attention_mask"],
            dose,
            0.0,
            0.0,
            empty_mask,
            read,
        )
        margin = endpoint_value(logits, score_type, ids)
        if not math.isfinite(margin):
            raise RuntimeError(f"non-finite mapping margin in {cell}: {margin}")
        rows.append({"cell": cell, "margin": margin, "pass": margin > 0})
    passed = sum(row["pass"] for row in rows)
    return {
        "n": len(rows),
        "passed": passed,
        "accuracy": passed / len(rows),
        "minimum_passed": 28,
        "pass": passed >= 28,
        "by_cell": {
            cell: {
                "n": sum(row["cell"] == cell for row in rows),
                "passed": sum(row["cell"] == cell and row["pass"] for row in rows),
                "mean_margin": mean(row["margin"] for row in rows if row["cell"] == cell),
            }
            for cell in sorted({row["cell"] for row in rows})
        },
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
) -> dict[str, Any]:
    if len(rows) != len(TASKS):
        raise RuntimeError(f"bootstrap requires 16 task-family rows, got {len(rows)}")
    rng = random.Random(BOOTSTRAP_SEED)
    estimate = statistic(rows)
    if not math.isfinite(estimate):
        raise RuntimeError("bootstrap statistic is non-finite on the observed task families")
    draws: list[float] = []
    for _ in range(BOOTSTRAPS):
        sample = [rows[rng.randrange(len(rows))] for _row in rows]
        value = statistic(sample)
        if math.isfinite(value):
            draws.append(value)
    if not draws:
        raise RuntimeError("bootstrap produced no finite draws")
    return {
        "estimate": estimate,
        "ci95_low": percentile(draws, 0.025),
        "ci95_high": percentile(draws, 0.975),
        "n_task_families": len(rows),
        "bootstrap_draws": len(draws),
    }


def surface_contrasts(records: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, float]]]:
    for row in records:
        for field in ("endpoint", "final_projection", "dose_projection"):
            value = float(row[field])
            if not math.isfinite(value):
                raise RuntimeError(
                    f"non-finite {field} for task={row['task_index']}, "
                    f"surface={row['surface']}, condition={row['condition']}: {value}"
                )
    lookup = {
        (int(row["task_index"]), row["surface"], row["condition"]): row
        for row in records
    }
    answer: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for task_index in range(len(TASKS)):
        for surface in [surface for surfaces in SURFACE_GROUPS.values() for surface in surfaces]:
            key = (task_index, surface)
            cells = {name: lookup.get((*key, name)) for name, _earlier, _readout in CONDITIONS}
            if any(value is None for value in cells.values()):
                raise RuntimeError(f"missing condition for task={task_index}, surface={surface}")
            answer[task_index][surface] = {
                "clean": float(cells["clean"]["endpoint"]),
                "earlier": float(cells["earlier_plus"]["endpoint"] - cells["earlier_minus"]["endpoint"]),
                "readout": float(cells["readout_plus"]["endpoint"] - cells["readout_minus"]["endpoint"]),
                "earlier_final_projection": float(
                    cells["earlier_plus"]["final_projection"]
                    - cells["earlier_minus"]["final_projection"]
                ),
                "readout_final_projection": float(
                    cells["readout_plus"]["final_projection"]
                    - cells["readout_minus"]["final_projection"]
                ),
                "earlier_dose_projection": float(
                    cells["earlier_plus"]["dose_projection"]
                    - cells["earlier_minus"]["dose_projection"]
                ),
            }
    return answer


def group_rows(
    contrasts: dict[int, dict[str, dict[str, float]]], surfaces: Iterable[str]
) -> list[dict[str, float]]:
    surfaces = tuple(surfaces)
    rows: list[dict[str, float]] = []
    for task_index in range(len(TASKS)):
        present = [contrasts[task_index][surface] for surface in surfaces]
        rows.append(
            {
                "task_index": float(task_index),
                **{
                    field: mean(row[field] for row in present)
                    for field in present[0]
                },
            }
        )
    return rows


def summarize_group(rows: list[dict[str, float]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in (
        "clean",
        "earlier",
        "readout",
        "earlier_final_projection",
        "readout_final_projection",
        "earlier_dose_projection",
    ):
        output[field] = bootstrap(rows, lambda sample, f=field: mean(row[f] for row in sample))
        if field in ("earlier", "readout"):
            output[field]["positive_task_families"] = sum(row[field] > 0 for row in rows)

    def ratio(sample: list[dict[str, float]]) -> float:
        denominator = mean(row["readout"] for row in sample)
        return mean(row["earlier"] for row in sample) / denominator if abs(denominator) > 1e-12 else math.nan

    output["retention_ratio_of_means"] = bootstrap(rows, ratio)
    return output


def summarize_reversed_pairs(
    contrasts: dict[int, dict[str, dict[str, float]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (first, second) in REVERSED_PAIRS.items():
        rows: list[dict[str, float]] = []
        for task_index in range(len(TASKS)):
            a = contrasts[task_index][first]
            b = contrasts[task_index][second]
            row: dict[str, float] = {"task_index": float(task_index)}
            for field in ("earlier", "readout"):
                row[f"{field}_semantic_binding"] = (a[field] + b[field]) / 2
                row[f"{field}_fixed_token_bias"] = (a[field] - b[field]) / 2
            rows.append(row)
        result[name] = {
            field: bootstrap(rows, lambda sample, f=field: mean(row[f] for row in sample))
            for field in (
                "earlier_semantic_binding",
                "earlier_fixed_token_bias",
                "readout_semantic_binding",
                "readout_fixed_token_bias",
            )
        }
    return result


def compact_surface_effects(
    contrasts: dict[int, dict[str, dict[str, float]]]
) -> dict[str, dict[str, float]]:
    surfaces = [surface for group in SURFACE_GROUPS.values() for surface in group]
    return {
        surface: {
            "earlier_mean": mean(contrasts[index][surface]["earlier"] for index in range(len(TASKS))),
            "readout_mean": mean(contrasts[index][surface]["readout"] for index in range(len(TASKS))),
            "earlier_final_projection_mean": mean(
                contrasts[index][surface]["earlier_final_projection"] for index in range(len(TASKS))
            ),
        }
        for surface in surfaces
    }


def memory_snapshot() -> dict[str, int]:
    return {
        "device_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "limit_bytes": VRAM_LIMIT_BYTES,
    }


def safe_memory_snapshot() -> dict[str, Any]:
    try:
        if not torch.cuda.is_available():
            return {"cuda_available": False, "limit_bytes": VRAM_LIMIT_BYTES}
        return {"cuda_available": True, **memory_snapshot()}
    except Exception as error:
        return {
            "cuda_available": None,
            "limit_bytes": VRAM_LIMIT_BYTES,
            "snapshot_error": f"{type(error).__name__}: {error}",
        }


def enforce_vram_limit(stage: str) -> None:
    snapshot = memory_snapshot()
    if snapshot["peak_reserved_bytes"] > VRAM_LIMIT_BYTES:
        raise RuntimeError(
            f"VRAM limit exceeded after {stage}: "
            f"{snapshot['peak_reserved_bytes']} > {VRAM_LIMIT_BYTES} bytes"
        )


def run() -> int:
    started = time.perf_counter()
    script_path = pathlib.Path(__file__).resolve()
    design = frozen_design()
    design_sha256 = sha256_bytes(canonical_json(design).encode("utf-8"))
    script_sha256 = sha256_file(script_path)
    RUN_STATE["frozen_custody"] = {
        "expected_artifacts": dict(EXPECTED_SHA256),
        "model_repo": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "vector_repo": VECTOR_REPO,
        "vector_revision": VECTOR_REVISION,
        "script_sha256": script_sha256,
        "preregistration_sha256": PREREG_SHA256,
        "design_sha256": design_sha256,
    }
    emit(
        "TEMPORAL_LOCUS_PLAN",
        {
            "status": "frozen_not_outcome",
            "design_sha256": design_sha256,
            "script_sha256": script_sha256,
            "preregistration_sha256": PREREG_SHA256,
            "expected_total_forwards": EXPECTED_TOTAL_FORWARDS,
        },
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen BF16 job")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one CUDA device, found {torch.cuda.device_count()}")
    random.seed(EXECUTION_SEED)
    torch.manual_seed(EXECUTION_SEED)
    torch.cuda.manual_seed_all(EXECUTION_SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.reset_peak_memory_stats()

    download_started = time.perf_counter()
    model_dir, goal_path, lava_path, artifact_hashes = download_and_verify()
    download_seconds = time.perf_counter() - download_started
    custody = {
        "model_repo": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "vector_repo": VECTOR_REPO,
        "vector_revision": VECTOR_REVISION,
        "artifacts": artifact_hashes,
        "script_sha256": script_sha256,
        "preregistration_sha256": PREREG_SHA256,
        "design_sha256": design_sha256,
    }
    RUN_STATE["verified_custody"] = custody
    emit("TEMPORAL_LOCUS_SHA256", custody)

    load_started = time.perf_counter()
    tokenizer, model, model_load = load_model(model_dir)
    dose_cpu, read_cpu, vector_metadata = load_directions(goal_path, lava_path)
    dose = dose_cpu.to(device="cuda", dtype=torch.bfloat16)
    read = read_cpu.to(device="cuda", dtype=torch.float32)
    load_seconds = time.perf_counter() - load_started
    enforce_vram_limit("model and vector load")

    ids = {
        label: one_token(tokenizer, label)
        for label in ("success", "failure", "continue", "stop", "K", "M")
    }
    cases = build_cases(tokenizer)
    prompt_manifest = [
        {
            "case_id": case["case_id"],
            "prompt_sha256": case["prompt_sha256"],
            "dose_token_count": case["dose_token_count"],
            "intervening_token_count": case["intervening_token_count"],
        }
        for case in cases
    ]
    prompt_manifest_sha256 = sha256_bytes(canonical_json(prompt_manifest).encode("utf-8"))

    mapping_started = time.perf_counter()
    mapping = mapping_checks(tokenizer, model, ids, dose, read)
    mapping_seconds = time.perf_counter() - mapping_started
    enforce_vram_limit("mapping gate")
    emit(
        "TEMPORAL_LOCUS_MAPPING",
        {
            "n": mapping["n"],
            "passed": mapping["passed"],
            "minimum_passed": mapping["minimum_passed"],
            "pass": mapping["pass"],
        },
    )
    if not mapping["pass"]:
        result = {
            "schema": design["schema"],
            "status": "stopped_mapping_gate",
            "mapping": mapping,
            "forwards_completed": EXPECTED_MAPPING_FORWARDS,
            "custody": custody,
            "prompt_manifest_sha256": prompt_manifest_sha256,
            "runtime_seconds": {
                "download_and_hash": download_seconds,
                "load_and_verify": load_seconds,
                "mapping": mapping_seconds,
                "total": time.perf_counter() - started,
            },
            "vram": memory_snapshot(),
        }
        emit("TEMPORAL_LOCUS_RESULT", result)
        return 2

    records: list[dict[str, Any]] = []
    endpoint_started = time.perf_counter()
    for condition_index, (condition, earlier, readout_coefficient) in enumerate(CONDITIONS):
        for case in cases:
            before = time.perf_counter()
            logits, captured = score_logits(
                model,
                case["input_ids"],
                case["attention_mask"],
                dose,
                earlier,
                readout_coefficient,
                case["dose_mask"],
                read,
            )
            records.append(
                {
                    "task_index": case["task_index"],
                    "surface": case["surface"],
                    "surface_family": case["surface_family"],
                    "condition": condition,
                    "endpoint": endpoint_value(logits, case["score_type"], ids),
                    **captured,
                    "latency_seconds": time.perf_counter() - before,
                }
            )
        emit(
            "TEMPORAL_LOCUS_PROGRESS",
            {
                "condition": condition,
                "conditions_complete": condition_index + 1,
                "endpoint_forwards_complete": len(records),
                "total_forwards_complete": EXPECTED_MAPPING_FORWARDS + len(records),
            },
        )
        enforce_vram_limit(f"condition {condition}")
    endpoint_seconds = time.perf_counter() - endpoint_started
    if len(records) != EXPECTED_ENDPOINT_FORWARDS:
        raise RuntimeError(f"expected {EXPECTED_ENDPOINT_FORWARDS} endpoint rows, got {len(records)}")

    contrasts = surface_contrasts(records)
    group_effects = {
        group: summarize_group(group_rows(contrasts, surfaces))
        for group, surfaces in SURFACE_GROUPS.items()
    }
    reversed_mapping = summarize_reversed_pairs(contrasts)
    replication_by_group = {
        group: {
            "readout_mean_positive": summary["readout"]["estimate"] > 0,
            "absolute_retention_ratio_below_0_25": abs(
                summary["retention_ratio_of_means"]["estimate"]
            ) < 0.25,
        }
        for group, summary in group_effects.items()
    }
    for value in replication_by_group.values():
        value["pass"] = bool(
            value["readout_mean_positive"]
            and value["absolute_retention_ratio_below_0_25"]
        )

    vram = memory_snapshot()
    vram_pass = vram["peak_reserved_bytes"] <= VRAM_LIMIT_BYTES
    result = {
        "schema": design["schema"],
        "status": "complete" if vram_pass else "failed_vram_limit",
        "qualitative_replication_pass": bool(
            vram_pass and all(value["pass"] for value in replication_by_group.values())
        ),
        "replication_by_group": replication_by_group,
        "mapping": mapping,
        "group_effects": group_effects,
        "reversed_mapping_decomposition": reversed_mapping,
        "surface_effects": compact_surface_effects(contrasts),
        "counts": {
            "task_families": len(TASKS),
            "surfaces": len(cases) // len(TASKS),
            "endpoint_forwards": len(records),
            "mapping_forwards": mapping["n"],
            "total_forwards": len(records) + mapping["n"],
        },
        "model_load": model_load,
        "vector_metadata": vector_metadata,
        "custody": custody,
        "prompt_manifest_sha256": prompt_manifest_sha256,
        "runtime_seconds": {
            "download_and_hash": download_seconds,
            "load_and_verify": load_seconds,
            "mapping": mapping_seconds,
            "endpoint": endpoint_seconds,
            "mean_endpoint_forward": mean(row["latency_seconds"] for row in records),
            "total": time.perf_counter() - started,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "vram": vram,
    }
    emit("TEMPORAL_LOCUS_RESULT", result)
    return 0 if vram_pass else 3


def main() -> int:
    try:
        return run()
    except Exception as error:
        emit(
            "TEMPORAL_LOCUS_RESULT",
            {
                "schema": "temporal-locus-bf16-v1",
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(limit=8),
                "custody": RUN_STATE.get(
                    "verified_custody", RUN_STATE.get("frozen_custody")
                ),
                "vram": safe_memory_snapshot(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
