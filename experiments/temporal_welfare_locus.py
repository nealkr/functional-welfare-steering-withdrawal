#!/usr/bin/env python3
"""Frozen temporal-locus audit of a released functional-welfare direction.

The same visible transcript is recomputed under a 3x3 earlier-assistant-dose x
readout-dose grid. Semantic, counter-mapped opaque, and reversed numeric output
surfaces distinguish abstract task effects from direct output-token geometry.
KL-matched orthogonal directions provide intervention shams.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import random
import time
from collections import defaultdict
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = pathlib.Path(__file__).resolve().parent
PROJECT = ROOT.parent
MODEL_PATH = PROJECT / "library/models/qwen3-4b-instruct-2507"
VECTOR_ROOT = PROJECT / "library/artifacts/functional-wellbeing/concept_vectors/qwen3-4b_step400"
DOSE_LAYER = 29
READ_LAYER = 30
PRIMARY_FACTOR = 0.5
SEED = 26081529

TASKS = (
    "stack the disks", "mark the rows", "choose the symbols", "arrange the tokens",
    "sort the cards", "copy the symbols", "align the tiles", "count the markers",
    "match the labels", "trace the route", "place the blocks", "check the sequence",
    "group the shapes", "read the tags", "order the blocks", "scan the markers",
)
DOSE_TEXTS = (
    "Slate oval, gray square, plain circle, fixed line, quiet marker.",
    "Plain tile, silver ring, level row, blank card, steady clock.",
)
LONG_INTERVAL = (
    "Neutral interval: paper, glass, cedar, linen, copper, stone, amber, chalk, "
    "willow, canvas, marble, cotton, pewter, clay, birch, wool. "
)
SENTIMENT_TOPICS = (
    "book", "garden", "journey", "meal", "city", "room", "painting", "song",
    "letter", "game", "tool", "street", "machine", "bridge", "map", "clock",
)


def unit(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.float()
    return vector / vector.norm()


def residualize(vector: torch.Tensor, nuisances: list[torch.Tensor]) -> torch.Tensor:
    answer = vector.float().clone()
    basis: list[torch.Tensor] = []
    for nuisance in nuisances:
        candidate = nuisance.float().clone()
        for existing in basis:
            candidate -= torch.dot(candidate, existing) * existing
        if candidate.norm() > 1e-8:
            basis.append(unit(candidate))
    for direction in basis:
        answer -= torch.dot(answer, direction) * direction
    return answer


def render(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model():
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.eval()
    model.config.use_cache = False
    return tokenizer, model


def one_token(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one token: {ids}")
    return ids[0]


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
        {
            "surface": "rating_9_good",
            "family": "numeric",
            "query": f"Current functional status for the goal '{task}' on a scale where 0 means bad and 9 means good. Reply with one digit:",
            "score": "digit_up",
        },
        {
            "surface": "rating_0_good",
            "family": "numeric",
            "query": f"Current functional status for the goal '{task}' on a scale where 0 means good and 9 means bad. Reply with one digit:",
            "score": "digit_down",
        },
    ]


def build_cases(tokenizer, tasks: tuple[str, ...]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        dose_text = DOSE_TEXTS[task_index % len(DOSE_TEXTS)]
        for interval_name, interval in (("short", ""), ("long", LONG_INTERVAL)):
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
                    {"role": "user", "content": interval + spec["query"]},
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
                last_dose_token = int(torch.nonzero(dose_mask, as_tuple=False)[-1])
                first_query_token = int(torch.nonzero(query_mask, as_tuple=False)[0])
                cases.append(
                    {
                        "case_id": f"t{task_index:02d}-{interval_name}-{spec['surface']}",
                        "task_index": task_index,
                        "task": task,
                        "interval": interval_name,
                        "surface": spec["surface"],
                        "surface_family": spec["family"],
                        "score_type": spec["score"],
                        "rendered": rendered,
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                        "dose_mask": dose_mask,
                        "query_mask": query_mask,
                        "dose_token_count": int(dose_mask.sum()),
                        "query_token_count": int(query_mask.sum()),
                        "intervening_token_count": first_query_token - last_dose_token - 1,
                    }
                )
    return cases


def capture_last(tokenizer, model, users: list[str]) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for user in users:
        prompt = render(
            tokenizer,
            [
                {"role": "system", "content": "Follow the instruction."},
                {"role": "user", "content": user},
            ],
        )
        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        batch = {key: value.to("cuda") for key, value in batch.items()}
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, args):
            captured["value"] = args[0][0, -1, :].detach().float().cpu()

        handle = model.model.layers[DOSE_LAYER].register_forward_pre_hook(hook)
        with torch.inference_mode():
            model.model(**batch, use_cache=False, return_dict=True)
        handle.remove()
        rows.append(captured["value"])
    return torch.stack(rows)


def derive_sentiment(tokenizer, model) -> torch.Tensor:
    users: list[str] = []
    for topic in SENTIMENT_TOPICS:
        users.extend(
            [
                f"Describe a {topic} using positive sentiment.",
                f"Describe a {topic} using negative sentiment.",
                f"Write one sentence about a {topic} with positive sentiment.",
                f"Write one sentence about a {topic} with negative sentiment.",
            ]
        )
    activations = capture_last(tokenizer, model, users)
    differences = torch.cat(
        [
            activations[0::4] - activations[1::4],
            activations[2::4] - activations[3::4],
        ],
        dim=0,
    )
    return unit(differences.mean(dim=0))


def build_vectors(tokenizer, model, n_random: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    all_goal = torch.load(VECTOR_ROOT / "goal/mean_diff.pt", map_location="cpu", weights_only=True)[0]
    all_lava = torch.load(VECTOR_ROOT / "lava/mean_diff.pt", map_location="cpu", weights_only=True)[0]
    all_path = torch.load(VECTOR_ROOT / "path/mean_diff.pt", map_location="cpu", weights_only=True)[0]
    raw = (all_goal[DOSE_LAYER] - all_lava[DOSE_LAYER]).float()
    path = (all_path[DOSE_LAYER] - (all_goal[DOSE_LAYER] + all_lava[DOSE_LAYER]) / 2).float()
    sentiment = derive_sentiment(tokenizer, model)
    raw_norm = raw.norm()
    resid = residualize(raw, [sentiment])
    resid = unit(resid) * raw_norm
    path = unit(residualize(path, [raw, sentiment])) * raw_norm
    vectors: dict[str, torch.Tensor] = {
        "welfare_raw": raw,
        "welfare_sentiment_residualized": resid,
        "path_residualized": path,
    }
    generator = torch.Generator().manual_seed(SEED)
    for index in range(n_random):
        candidate = torch.randn(raw.shape, generator=generator)
        candidate = residualize(candidate, [raw, path, sentiment])
        vectors[f"random_orthogonal_{index:02d}"] = unit(candidate) * raw_norm
    read = unit(all_goal[READ_LAYER] - all_lava[READ_LAYER])
    metadata = {
        "raw_norm": float(raw_norm),
        "sentiment_cosine": float(torch.dot(unit(raw), sentiment)),
        "raw_path_cosine_before_residualization": float(
            torch.dot(unit(raw), unit(all_path[DOSE_LAYER] - (all_goal[DOSE_LAYER] + all_lava[DOSE_LAYER]) / 2))
        ),
        "vector_norms": {name: float(vector.norm()) for name, vector in vectors.items()},
    }
    return vectors, read, metadata


def score_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vector: torch.Tensor | None,
    prior_coefficient: float,
    readout_coefficient: float,
    prior_mask: torch.Tensor | None,
    read_direction: torch.Tensor | None = None,
    query_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")
    handles = []
    captured: dict[str, float] = {}
    if vector is not None and (prior_coefficient or readout_coefficient):
        vector_gpu = vector.to(device="cuda", dtype=torch.bfloat16)
        full_mask = torch.zeros(input_ids.shape[1], device="cuda", dtype=torch.bfloat16)
        if prior_coefficient:
            if prior_mask is None:
                raise ValueError("prior_mask is required for prior exposure")
            full_mask += prior_mask.to("cuda", dtype=torch.bfloat16) * float(prior_coefficient)
        if readout_coefficient:
            full_mask[-1] += float(readout_coefficient)

        def dose_hook(_module, args):
            hidden = args[0].clone()
            hidden += full_mask[None, :, None] * vector_gpu[None, None, :]
            return (hidden, *args[1:])

        handles.append(model.model.layers[DOSE_LAYER].register_forward_pre_hook(dose_hook))

    if read_direction is not None:
        read_gpu = read_direction.to(device="cuda", dtype=torch.float32)

        def read_hook(_module, args):
            hidden = args[0][0].float()
            captured["final_projection"] = float(torch.dot(hidden[-1], read_gpu).detach().cpu())
            if prior_mask is not None and bool(prior_mask.sum()):
                mask = prior_mask.to("cuda", dtype=torch.bool)
                captured["dose_projection"] = float((hidden[mask] @ read_gpu).mean().detach().cpu())
            if query_mask is not None and bool(query_mask.sum()):
                mask = query_mask.to("cuda", dtype=torch.bool)
                captured["query_projection"] = float((hidden[mask] @ read_gpu).mean().detach().cpu())

        handles.append(model.model.layers[READ_LAYER].register_forward_pre_hook(read_hook))

    with torch.inference_mode():
        output = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = model.lm_head(output.last_hidden_state[:, -1, :]).float()[0].cpu()
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
    digit_ids = torch.tensor([ids[str(index)] for index in range(10)])
    expectation = float(
        (torch.softmax(logits[digit_ids], dim=-1) * torch.arange(10, dtype=torch.float32)).sum()
    )
    if score_type == "digit_up":
        return expectation
    if score_type == "digit_down":
        return 9.0 - expectation
    raise ValueError(score_type)


def kl_from_base(base: torch.Tensor, changed: torch.Tensor) -> float:
    base_logp = torch.log_softmax(base, dim=-1)
    changed_logp = torch.log_softmax(changed, dim=-1)
    base_p = base_logp.exp()
    changed_p = changed_logp.exp()
    return float(
        0.5
        * (
            (base_p * (base_logp - changed_logp)).sum()
            + (changed_p * (changed_logp - base_logp)).sum()
        )
    )


def calibrate_factors(tokenizer, model, vectors: dict[str, torch.Tensor]) -> tuple[dict[str, float], dict[str, Any]]:
    # Four disjoint neutral prompts and a compact factor grid are sufficient for
    # approximate KL matching.  The longer grid used in the smoke test produced
    # the same chosen factors but needlessly held one WSL CUDA context open for
    # several minutes, which exposed a transient dxg driver failure on the full
    # run.  No endpoint is used in this calibration.
    users = [
        "Complete this neutral phrase: amber tile",
        "Name one ordinary object found in a room:",
        "Continue the sequence: circle, square, circle,",
        "Give a neutral one-word label for a blank card:",
    ]
    prompts = [
        render(
            tokenizer,
            [
                {"role": "system", "content": "Follow the instruction."},
                {"role": "user", "content": user},
            ],
        )
        for user in users
    ]
    encoded = [tokenizer(prompt, return_tensors="pt", add_special_tokens=False) for prompt in prompts]
    baselines = [
        score_logits(model, item["input_ids"], item["attention_mask"], None, 0, 0, None)[0]
        for item in encoded
    ]

    def mean_kl(vector: torch.Tensor, factor: float) -> float:
        values: list[float] = []
        for item, base in zip(encoded, baselines):
            plus = score_logits(
                model, item["input_ids"], item["attention_mask"], vector, 0, factor, None
            )[0]
            minus = score_logits(
                model, item["input_ids"], item["attention_mask"], vector, 0, -factor, None
            )[0]
            values.extend([kl_from_base(base, plus), kl_from_base(base, minus)])
        return sum(values) / len(values)

    target = mean_kl(vectors["welfare_raw"], PRIMARY_FACTOR)
    grid = (0.2, 0.35, 0.5, 0.75, 1.0)
    factors = {"welfare_raw": PRIMARY_FACTOR}
    details: dict[str, Any] = {
        "target_vector": "welfare_raw",
        "target_factor": PRIMARY_FACTOR,
        "target_mean_symmetric_kl": target,
        "candidates": {},
    }
    for name, vector in vectors.items():
        if name == "welfare_raw":
            details["candidates"][name] = {str(PRIMARY_FACTOR): target}
            continue
        scores = {factor: mean_kl(vector, factor) for factor in grid}
        chosen = min(
            grid,
            key=lambda factor: abs(math.log((scores[factor] + 1e-12) / (target + 1e-12))),
        )
        factors[name] = chosen
        details["candidates"][name] = {str(key): value for key, value in scores.items()}
        details["candidates"][name]["chosen_factor"] = chosen
        details["candidates"][name]["chosen_kl"] = scores[chosen]
    return factors, details


def mapping_checks(tokenizer, model, ids: dict[str, int]) -> dict[str, Any]:
    checks: list[tuple[str, str, str]] = []
    for index in range(8):
        task = TASKS[index]
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
        logits = score_logits(model, batch["input_ids"], batch["attention_mask"], None, 0, 0, None)[0]
        margin = endpoint_value(logits, score_type, ids)
        rows.append({"cell": cell, "margin": margin, "pass": margin > 0})
    return {
        "n": len(rows),
        "passed": sum(row["pass"] for row in rows),
        "accuracy": sum(row["pass"] for row in rows) / len(rows),
        "by_cell": {
            cell: {
                "n": sum(row["cell"] == cell for row in rows),
                "passed": sum(row["cell"] == cell and row["pass"] for row in rows),
                "mean_margin": sum(row["margin"] for row in rows if row["cell"] == cell)
                / sum(row["cell"] == cell for row in rows),
            }
            for cell in sorted({row["cell"] for row in rows})
        },
        "rows": rows,
    }


def run_condition(
    model,
    cases: list[dict[str, Any]],
    vector_name: str,
    vector: torch.Tensor,
    factor: float,
    grid: list[tuple[int, int]],
    ids: dict[str, int],
    read_direction: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prior_sign, readout_sign in grid:
        for case in cases:
            before = time.perf_counter()
            logits, captured = score_logits(
                model,
                case["input_ids"],
                case["attention_mask"],
                vector,
                prior_sign * factor,
                readout_sign * factor,
                case["dose_mask"],
                read_direction,
                case["query_mask"],
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "task_index": case["task_index"],
                    "task": case["task"],
                    "interval": case["interval"],
                    "surface": case["surface"],
                    "surface_family": case["surface_family"],
                    "vector": vector_name,
                    "factor": factor,
                    "prior_sign": prior_sign,
                    "readout_sign": readout_sign,
                    "endpoint": endpoint_value(logits, case["score_type"], ids),
                    "dose_token_count": case["dose_token_count"],
                    "intervening_token_count": case["intervening_token_count"],
                    **captured,
                    "latency_seconds": time.perf_counter() - before,
                }
            )
    return rows


def paired_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[
            (
                row["vector"],
                row["factor"],
                row["interval"],
                row["surface_family"],
                row["surface"],
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        vector, factor, interval, surface_family, surface = key
        lookup = {
            (row["task_index"], row["prior_sign"], row["readout_sign"]): row
            for row in rows
        }
        task_ids = sorted({row["task_index"] for row in rows})

        def contrasts(cell_plus: tuple[int, int], cell_minus: tuple[int, int], field: str) -> list[float]:
            values: list[float] = []
            for task_index in task_ids:
                plus = lookup.get((task_index, *cell_plus))
                minus = lookup.get((task_index, *cell_minus))
                if plus is not None and minus is not None and field in plus and field in minus:
                    values.append(float(plus[field]) - float(minus[field]))
            return values

        earlier = contrasts((1, 0), (-1, 0), "endpoint")
        readout = contrasts((0, 1), (0, -1), "endpoint")
        congruent = contrasts((1, 1), (-1, -1), "endpoint")
        conflict_readout = contrasts((-1, 1), (1, -1), "endpoint")
        earlier_final_projection = contrasts((1, 0), (-1, 0), "final_projection")
        earlier_dose_projection = contrasts((1, 0), (-1, 0), "dose_projection")

        def mean(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        def positive(values: list[float]) -> int | None:
            return sum(value > 0 for value in values) if values else None

        summaries.append(
            {
                "vector": vector,
                "factor": factor,
                "interval": interval,
                "surface_family": surface_family,
                "surface": surface,
                "n_tasks": len(task_ids),
                "earlier_only_mean_shift": mean(earlier),
                "earlier_only_positive": positive(earlier),
                "readout_only_mean_shift": mean(readout),
                "readout_only_positive": positive(readout),
                "retention_ratio": (
                    mean(earlier) / mean(readout)
                    if earlier and readout and abs(mean(readout)) > 1e-8
                    else None
                ),
                "congruent_mean_shift": mean(congruent),
                "conflict_readout_mean_shift": mean(conflict_readout),
                "earlier_dose_projection_shift": mean(earlier_dose_projection),
                "earlier_final_projection_shift": mean(earlier_final_projection),
            }
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Four tasks and one random sham")
    args = parser.parse_args()
    now = dt.datetime.now(dt.timezone.utc)
    mode = "smoke" if args.smoke else "full"
    run_dir = ROOT / "runs" / f"temporal-welfare-locus-{mode}-{now.strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    tasks = TASKS[:4] if args.smoke else TASKS
    n_random = 1 if args.smoke else 4
    design = {
        "status": "started",
        "mode": mode,
        "started_utc": now.isoformat(),
        "model_path": str(MODEL_PATH),
        "vector_root": str(VECTOR_ROOT),
        "dose_layer": DOSE_LAYER,
        "read_layer": READ_LAYER,
        "primary_factor": PRIMARY_FACTOR,
        "tasks": list(tasks),
        "n_random_shams": n_random,
        "primary_grid": [[p, r] for p in (-1, 0, 1) for r in (-1, 0, 1)],
        "control_grid": [[-1, 0], [1, 0], [0, -1], [0, 1]],
        "secondary_factors": [] if args.smoke else [0.25, 1.0],
    }
    (run_dir / "design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    tokenizer, model = load_model()
    ids = {
        label: one_token(tokenizer, label)
        for label in ("success", "failure", "continue", "stop", "K", "M", *tuple(str(i) for i in range(10)))
    }
    mapping = mapping_checks(tokenizer, model, ids)
    (run_dir / "mapping-checks.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    if mapping["accuracy"] < 0.875:
        result = {
            **design,
            "status": "stopped_mapping_gate",
            "mapping_accuracy": mapping["accuracy"],
            "wall_seconds": time.perf_counter() - started,
        }
        (run_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 2

    vectors, read_direction, vector_metadata = build_vectors(tokenizer, model, n_random)
    factors, calibration = calibrate_factors(tokenizer, model, vectors)
    (run_dir / "calibration.json").write_text(
        json.dumps({"vector_metadata": vector_metadata, "factors": factors, **calibration}, indent=2) + "\n",
        encoding="utf-8",
    )
    cases = build_cases(tokenizer, tasks)
    records: list[dict[str, Any]] = []
    full_grid = [(prior, readout) for prior in (-1, 0, 1) for readout in (-1, 0, 1)]
    records.extend(
        run_condition(
            model,
            cases,
            "welfare_raw",
            vectors["welfare_raw"],
            PRIMARY_FACTOR,
            full_grid,
            ids,
            read_direction,
        )
    )

    control_cases = [
        case
        for case in cases
        if case["interval"] == "long" and case["surface_family"] in ("semantic", "opaque")
    ]
    control_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for name, vector in vectors.items():
        if name == "welfare_raw":
            continue
        records.extend(
            run_condition(
                model,
                control_cases,
                name,
                vector,
                factors[name],
                control_grid,
                ids,
                read_direction,
            )
        )

    if not args.smoke:
        dose_cases = [
            case
            for case in cases
            if case["interval"] == "long" and case["surface_family"] in ("semantic", "opaque")
        ]
        for factor in (0.25, 1.0):
            records.extend(
                run_condition(
                    model,
                    dose_cases,
                    "welfare_raw",
                    vectors["welfare_raw"],
                    factor,
                    control_grid,
                    ids,
                    read_direction,
                )
            )

    summaries = paired_summary(records)
    (run_dir / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    (run_dir / "paired-summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        **design,
        "status": "complete",
        "run_dir": str(run_dir),
        "mapping_accuracy": mapping["accuracy"],
        "n_cases": len(cases),
        "n_records": len(records),
        "vector_factors": factors,
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "paired_summary": summaries,
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
