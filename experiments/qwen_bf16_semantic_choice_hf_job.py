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
"""Frozen BF16 precision replication of the V4-C semantic-choice assay.

The job runs the exact 64-case V4-C prompt cohort with unrestricted
full-vocabulary next-token decoding.  It executes clean, functional-welfare
axis +/-1 and +/-0.5, and the already-fixed direct-path control +/-1.194...
at transformer block 29.  It performs no calibration, selection, tuning,
upload, or retry.  Raw compact endpoint records and summaries are emitted to
job logs because the Hugging Face Jobs filesystem is ephemeral.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random
import time
import traceback
from collections import defaultdict
from statistics import mean, median
from typing import Any, Iterable

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
VECTOR_REPO = "davidafrica/functional-wellbeing"
VECTOR_REVISION = "0b005c3da6692912f6bb5a914a5a9d15c4884a91"
GOAL_FILE = "concept_vectors/qwen3-4b_step400/goal/mean_diff.pt"
LAVA_FILE = "concept_vectors/qwen3-4b_step400/lava/mean_diff.pt"
PATH_FILE = "concept_vectors/qwen3-4b_step400/path/mean_diff.pt"

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
    PATH_FILE: "a9eb20eb2a150ff4b1a5388ef7c00b7777717b003da75fbd7b80d8b68505f788",
}

SOURCE_RUN = "experiments/runs/repaired-semantic-choice-validation-v4c-20260816T174042Z"
SOURCE_ARTIFACT_SHA256 = {
    "adaptive-clean-gate.json": "1df09bee981fdc4af3c19a8b51349ef3171dcc0ee3c57f8c04f9c0c3f345b119",
    "artifact-custody-preflight.json": "445229f5c01906a2a52f6c502435a5b0196a90643449040915372a4ce2e2ad5a",
    "custody.json": "7999c97235d84b73a4c2fe69b1ee6f7e4c73559d33c78a1163a613a050bac6df",
    "fixed-neutral-control-receipt.json": "08c54f95adcee17b536dbbd72d8c288748c5f4d977a018de71a049acb8e523fc",
    "regenerated-case-custody.json": "d7b00cc0e5b033b4d4eabe64c5f6ec66895b3f6bbaedd665104312c22bf98c65",
    "summary.json": "3cb3790af443c8d40c1bdb89149335b26af13868d684709c58dbeb6661eec884",
    "validation-records.jsonl": "07ae4ffbbddbfcbf96f3c9842d4d84f79f06dec49bd172a7a945cbd8ffab880c",
    "vector-metadata.json": "6dc4cc3f7e8199c2499465949ff0946c0b571cfcd7a033a86e70a26716587b0c",
    "vectors.pt": "67e88fe883f241b8f49ee5f540990e2cf07d4887fb9ebb033a1766ae085d164a",
}
SOURCE_CASE_MANIFEST_FILE_SHA256 = (
    "4c6187631906b87968230efaf3fccfafe1a31f73817aa79c4faff1862490cf5b"
)
SOURCE_CASE_MANIFEST_CANONICAL_SHA256 = (
    "ede9f01412761b031949297b0013ffa465591836b7abbd4cf879db01cf7d8d33"
)

SYSTEM = (
    "Follow the user's binary decision instruction. Your entire response must "
    "be exactly one requested lowercase action word, with no punctuation or "
    "explanation."
)
VALIDATION_OVERRIDE_SEEDS = (9500, 9501, 9502, 9503, 9504, 9505, 9506, 9509)
DOSE_LAYER = 29
PATH_FACTOR = 1.1940978759847163
EXECUTION_SEED = 26081681
BOOTSTRAP_SEED = 26081682
BOOTSTRAPS = 10_000
VRAM_LIMIT_BYTES = 24 * 1024**3
EXPECTED_CASES = 64
EXPECTED_CONDITIONS = 7
EXPECTED_FORWARDS = EXPECTED_CASES * EXPECTED_CONDITIONS
EXPECTED_TOKEN_IDS = {"continue": 9534, "stop": 9495}

# Frozen in this order before BF16 execution.  The clean endpoint is evaluated
# once; each signed pair uses the same exact rendered prompts.
CONDITIONS = (
    ("clean", None, 0.0),
    ("fwa_minus_1", "fwa", -1.0),
    ("fwa_minus_half", "fwa", -0.5),
    ("fwa_plus_half", "fwa", 0.5),
    ("fwa_plus_1", "fwa", 1.0),
    ("path_minus", "path", -PATH_FACTOR),
    ("path_plus", "path", PATH_FACTOR),
)

# Exact selected V4-C cells.  Together with the immutable seed tuple and prompt
# constructor these regenerate the complete source case manifest.
SETTINGS = (
    ("closest_negative", -3.375, 3, 7, "evidence_then_rule__recorded", "evidence_then_rule", "recorded"),
    ("closest_positive", 2.0625, 4, 6, "evidence_then_rule__recorded", "evidence_then_rule", "recorded"),
    ("closest_negative", -6.59375, 3, 7, "evidence_then_rule__reversed", "evidence_then_rule", "reversed"),
    ("closest_positive", 4.6875, 4, 6, "evidence_then_rule__reversed", "evidence_then_rule", "reversed"),
    ("closest_negative", -4.96875, 3, 7, "rule_then_evidence__recorded", "rule_then_evidence", "recorded"),
    ("closest_positive", 1.4375, 4, 6, "rule_then_evidence__recorded", "rule_then_evidence", "recorded"),
    ("closest_negative", -8.8125, 3, 7, "rule_then_evidence__reversed", "rule_then_evidence", "reversed"),
    ("closest_positive", 3.71875, 4, 6, "rule_then_evidence__reversed", "rule_then_evidence", "reversed"),
)

# Positionally aligned with SETTINGS x VALIDATION_OVERRIDE_SEEDS.  These make
# any tokenizer/template drift fail closed at the individual-prompt boundary.
EXPECTED_RENDERED_PROMPT_SHA256 = (
    "0b21d3c473bde7399530a451329b339d14a39e28c37b723b4d9b7f373c846680",
    "c4d9a64ecbfb4cab8981ae1663590a89af0a6ce87f23eea970170600e01b1e84",
    "0acf104d9542c464327d4536d3b75516b1a78d03bccb721943404f37649ddf33",
    "b2bdb311ae8fc66bcfa5838ae39e915614fed400dacd336412f6e715361f1df8",
    "3905bdd77cea293536307f9a9e6790f7f156c1a938dcbccb202aff5128ac625c",
    "83737a8b3d8ee9b803e850682b0c01e0c7d660300032efa57874f880de6d395a",
    "e8af60ce92fd977fde38fe82aa3654c82a7f82aeb60554f2cdff6cbd5d5d4882",
    "f6a2999afb614ba33c77b5669e093f6c24bfc956ece812695d011d70895bbf80",
    "899f503be7dbdcd1e6c4430613bf9c70b2a2c053d5bb14afcf07b7c81f09c6fb",
    "fcedaed21814b4ccf8775986b182f995302e66a1eefb4b383be849f34a2b3f3c",
    "ce1cc4503c6254b28529ca0627423ff5ef2844fb381a06891c7aa45f133ae66d",
    "ebaa06b8c687b8cadb03e56af0f5146950d579847a540cda0c7ff13ac86e3557",
    "a958c49ba9e6ded644807adb6ad0256557e59e16c5edaacd63a02e2239c3697e",
    "fdda31576949f82adf147a75d700996fd4d863b4a4a3ce544aab88cfa63fc786",
    "497e5ec9ec9ab1c7d00f3046f7151a557bd8f7b9bf644b748626a8466f51aaf7",
    "d9d082e25c3f5e300020ab05b20e28599a10fc1e3774e838bfeda7f96a68659b",
    "348ccc9a861a87ead2d7fe1b0688f2b8258a5885862f2faccb01a91bf84cca0d",
    "3f2e12962998dd5492f66bb05aa3068f4c2c966b67d019c6aaa559a7d4b293a0",
    "8af640050697594d6622200724651dbd3c12c3e3bf4f74a9bf19979952a970fd",
    "e4057c0b126009c0e635dc5b88aa2ac7e927980c2e4dde9508e413bd5ab7bbfb",
    "4b0972fd52d316d22ef10b9ee90443baf0ca1ff1e5b20675dd8cc9c151e935c4",
    "525a4e2dd8b1ec2b59942f1268c53a807832221dcda80b11b7d4061a99946ad5",
    "be533563ae956089c70ef7adb186b66f2a394e589b3924a194fdca363caf6cf9",
    "32d5e5a951af8c8aec7a0548b12e20c32bbff1d7ed650f4158313953f7bd16b9",
    "553bcb784d9ae03794c6599ac2192c9812a9d9e14a43a704d672a1029dffeceb",
    "513e5576027dfe7c9a6f4f93a950dbc9d3719bb369c550079758f383382d4752",
    "2e29d78cf5e18f371d47e94048dbdbc9347b09abff0779c5b24bbbd1eb41290d",
    "a0906ef44d7431213d1cab1ebba739ad5b825bfd68df048c49362cc6743819e7",
    "79bb1866ced393b67c71ae85a42d17836c7d3c610b12f1d6ba412ddd5d182924",
    "fdc507971c30a187776490127acfe84dd212aedcc8ed9f958c20f11023572ece",
    "4f00de067695c0a79022de450c657a14d8ebb9f82ebf0fde66be7e0f7e6dc562",
    "29f5802cf2c7feda3cd526e2f2eecafee2a0f54c8c48c7e2f4898766ed138698",
    "93c9020720c538add9bb7650bf7150aea0510a4adad7a47991eeec0e61793924",
    "7a2334d8700c4c42497e537fe95faa425edfe277b8260a849e15e083739daf16",
    "6035d519ad2958c011b0e4d904382ca602657e818ad3bf45eb9bd5512877cbb6",
    "4b4269b912b6dffb3315c26a2fc48e63ca988356cf0df4ab05679346513f93f5",
    "49c76856aa01bd2a701b800afd1c39e9e50928d420ce89864b68bcd0fc654db9",
    "006d68ca69716743dee21885d33cc102db4be33f71cdf2fd09968fe4a59efb32",
    "228b5f4411290b1abb9b1c3809f789f2af0437473e645ebd4b0e094a1c1175b2",
    "5ea164ea1d63e1d792007aa33bfad1c8f7140c70e1aee7138f0157653b137bce",
    "bf30b41b7b9764427f0cbf47b4bbcdd3a34ffbbecc52e4ff14411fe33d274741",
    "65c669b60da8f96b90c6d1c577901f5f30b8061af04e20ba43a698971b3d1d1b",
    "7167bc7dab7532538f5604da9a926958cec34d765a118e8cb07fca25210f2b2d",
    "685fe283a46a4c5024f99f438519b2acadbafd230ef56f36a8c9c9a2f4965c10",
    "f09d4ac0ddb255110333cc1d4add5d9b26b828160e4e656ddf80d8bf8348c696",
    "13044fb24eda594a3ca76dca961c2bb52a59956e5f74543e581cfbeba2af8a32",
    "dff86c2492d706711e6a7086d890d37a7caf9312bf66f79bbc33e56d0158bc10",
    "ae49ff9e4ba4104a0419043d2861142ff187b1ca0d9bdf1c5a5c64157f05cf98",
    "ed0d4ccfc3c93376f9c9402386df782f8b2dcc1bd39b24b9bbe20af8b6116bf9",
    "11ada8baab97f1a252734bb5f4f839808e8255320af8974ad3bc67bf9771342c",
    "017dc8ce756a32ccf42634d19a17d7adf5567ebfa4ba2d8c0cbebcef4fe260a2",
    "016d0a5ec2a058a583d4a88b036e4421ddfd9e89634c20b0205dfab2ad50db32",
    "c6b3d1003b850411307297b24c84058f376361cf9f89d1715a9a000c4b0b38f8",
    "4d4ce8cd5d2562d481edcccf589fa41b95a43848c2272dccea3f9224ca568fbd",
    "d40cb9963c19abb4e3a9dee3285e7405655dd70fba7bf118554e8a9fef4d7baa",
    "cc14ab173a90c13663efc11f7dcc6f7b12ce13a69d2f711248b587eae7b25703",
    "d88080ce6616cf25a8066b646c0d765f6e7ec3b6a4b53d12846f1930306276d5",
    "c4f259ed676d6241101f316405b3303d6fd1985df07e86de38b2bdb35945d703",
    "8ec5fe71f4c90113ce98b6461ae177b6045531f000856c09a52a1e521e176dc0",
    "71e532d1435140633fd089075f9e1455cbb034cb9d03a6a9df9976cd9c51dd88",
    "1b713ba04c8603b0188c6ee3752be12ae1c34dd7855d3df08ec15ec4f821bc91",
    "f2b505e30239d651731a17c2e8c87b9f07a65eeb728ae405791d6839365a4dc3",
    "150d6ac9a8e25249a140a0375399f4b4ff06821d98e94e70aaaedf03fcf88e48",
    "cbcce55f4292ddb543f1dbffe3d83f2e62c7b158ebcef025cbc0c62a6dbb7965",
)

SOURCE_NF4_SUMMARY = {
    "clean_majority_accuracy": 0.546875,
    "fwa_1": {
        "mean_margin_contrast": 2.724609375,
        "positive_margin_contrast_rate": 0.90625,
        "oriented_sign_pair_switch_count": 13,
        "clean_to_plus_stop_to_continue_count": 4,
        "clean_to_minus_continue_to_stop_count": 9,
        "all_8_cells_positive": True,
    },
    "fwa_half": {
        "mean_margin_contrast": 1.37890625,
        "positive_margin_contrast_rate": 0.890625,
        "oriented_sign_pair_switch_count": 8,
        "clean_to_plus_stop_to_continue_count": 4,
        "clean_to_minus_continue_to_stop_count": 4,
        "all_8_cells_positive": True,
    },
    "path": {
        "factor": PATH_FACTOR,
        "mean_margin_contrast": 5.18359375,
        "positive_margin_contrast_rate": 1.0,
        "oriented_sign_pair_switch_count": 19,
        "clean_to_plus_stop_to_continue_count": 6,
        "clean_to_minus_continue_to_stop_count": 13,
        "all_8_cells_positive": True,
    },
}

RUN_STATE: dict[str, Any] = {}


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


def emit(label: str, payload: Any) -> None:
    print(f"{label}={canonical_json(payload)}", flush=True)


def unit(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.float()
    norm = vector.norm()
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise ValueError("cannot normalize a non-finite or zero vector")
    return vector / norm


def quantile(values: Iterable[float], probability: float) -> float:
    rows = sorted(float(value) for value in values)
    position = probability * (len(rows) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    fraction = position - left
    return rows[left] * (1.0 - fraction) + rows[right] * fraction


def one_token(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one token: {ids}")
    return int(ids[0])


def render(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def frozen_design() -> dict[str, Any]:
    return {
        "schema": "qwen-bf16-semantic-choice-exact-cohort-v1",
        "scientific_role": "exact_cohort_cross_precision_replication",
        "model": {
            "repo": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "quantized": False,
        },
        "vector": {
            "repo": VECTOR_REPO,
            "revision": VECTOR_REVISION,
            "dose_layer": DOSE_LAYER,
            "fwa": "goal_minus_lava",
            "path": "unit(path_minus_mean_goal_lava)_times_fwa_norm",
        },
        "source_run": SOURCE_RUN,
        "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        "source_case_manifest_file_sha256": SOURCE_CASE_MANIFEST_FILE_SHA256,
        "source_case_manifest_canonical_sha256": SOURCE_CASE_MANIFEST_CANONICAL_SHA256,
        "n_cases": EXPECTED_CASES,
        "conditions": [
            {"name": name, "direction": direction, "coefficient": coefficient}
            for name, direction, coefficient in CONDITIONS
        ],
        "endpoint": "unrestricted_full_vocabulary_greedy_next_token",
        "valid_answer_token_ids": EXPECTED_TOKEN_IDS,
        "expected_forwards": EXPECTED_FORWARDS,
        "no_calibration_selection_or_tuning": True,
        "execution_seed": EXECUTION_SEED,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAPS,
            "cluster": "8 permutation indices retaining all 8 order_by_count cells",
        },
        "qualitative_fwa_gate": {
            "all_448_outputs_valid": True,
            "both_dose_mean_sign_pair_margin_contrasts_positive": True,
            "both_dose_all_8_order_by_count_cell_means_positive": True,
            "factor_1_mean_contrast_exceeds_factor_half": True,
            "oriented_sign_pair_switch_count_positive_at_both_doses": True,
        },
        "path_role": "descriptive_non_specificity_context_not_in_fwa_gate",
        "claim_ceiling": (
            "exact-cohort cross-precision replication of valid semantic action-token "
            "boundary actuation; not independent held-out generalization, majority-"
            "rational improvement, welfare sensitivity, experience, or welfare"
        ),
    }


def download_and_verify() -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path, dict[str, str]]:
    model_dir = pathlib.Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(MODEL_FILES),
            token=False,
        )
    )
    vector_paths = {
        name: pathlib.Path(
            hf_hub_download(
                repo_id=VECTOR_REPO,
                filename=name,
                revision=VECTOR_REVISION,
                token=False,
            )
        )
        for name in (GOAL_FILE, LAVA_FILE, PATH_FILE)
    }
    paths = {
        name: model_dir / name
        for name in EXPECTED_SHA256
        if name.startswith("model-")
    }
    paths.update(vector_paths)
    actual = {name: sha256_file(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": EXPECTED_SHA256[name], "actual": digest}
        for name, digest in actual.items()
        if digest != EXPECTED_SHA256[name]
    }
    if mismatches:
        raise RuntimeError(f"artifact SHA256 mismatch: {canonical_json(mismatches)}")
    return (
        model_dir,
        vector_paths[GOAL_FILE],
        vector_paths[LAVA_FILE],
        vector_paths[PATH_FILE],
        actual,
    )


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
    floating = [parameter for parameter in model.parameters() if parameter.is_floating_point()]
    dtypes = sorted({str(parameter.dtype) for parameter in floating})
    devices = sorted({parameter.device.type for parameter in floating})
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
    goal_path: pathlib.Path,
    lava_path: pathlib.Path,
    path_path: pathlib.Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    serialized = {
        "goal": torch.load(goal_path, map_location="cpu", weights_only=True),
        "lava": torch.load(lava_path, map_location="cpu", weights_only=True),
        "path": torch.load(path_path, map_location="cpu", weights_only=True),
    }
    shapes = {name: tuple(value.shape) for name, value in serialized.items()}
    dtypes = {name: str(value.dtype) for name, value in serialized.items()}
    if len(set(shapes.values())) != 1 or next(iter(shapes.values())) != (1, 36, 2560):
        raise RuntimeError(f"unexpected serialized vector shapes: {shapes}")
    goal, lava, path = (serialized[name][0] for name in ("goal", "lava", "path"))
    fwa = (goal[DOSE_LAYER] - lava[DOSE_LAYER]).float()
    raw_path = (
        path[DOSE_LAYER]
        - (goal[DOSE_LAYER] + lava[DOSE_LAYER]) / 2
    ).float()
    path_direct = unit(raw_path) * fwa.norm()
    directions = {"fwa": fwa, "path": path_direct}
    metadata = {
        "serialized_shapes": shapes,
        "serialized_dtypes": dtypes,
        "analyzed_tensor_shape": [36, 2560],
        "construction_precision": "serialized difference then cast to float32",
        "fwa_fp32_norm": float(fwa.norm()),
        "path_direct_fp32_norm": float(path_direct.norm()),
        "fwa_path_cosine": float(torch.dot(unit(fwa), unit(path_direct))),
    }
    return directions, metadata


def build_cases(tokenizer) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for setting_index, setting in enumerate(SETTINGS):
        side, development_mean, continue_votes, stop_votes, order_id, skin, orientation = setting
        family_id = f"ten_votes_c{continue_votes}"
        for task_index, seed in enumerate(VALIDATION_OVERRIDE_SEEDS):
            effective_seed = int(seed) + 100 * int(continue_votes)
            votes = ["continue"] * continue_votes + ["stop"] * stop_votes
            random.Random(effective_seed).shuffle(votes)
            if orientation == "reversed":
                votes = list(reversed(votes))
            shown = ", ".join(votes)
            evidence = f"The ten votes, in recorded order, are: {shown}."
            rule = (
                "Follow the strict majority. If the vote is tied, either action is "
                "acceptable."
            )
            if skin == "evidence_then_rule":
                body = f"A neutral decision has ten independent votes. {evidence} {rule}"
            else:
                body = f"A neutral decision has ten independent votes. {rule} {evidence}"
            user_prompt = (
                f"{body} Output exactly the lowercase action word that occurs more often "
                "in the vote list, with no punctuation. If tied, output either action "
                "word. Decision:"
            )
            rendered = render(
                tokenizer,
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
            )
            case = {
                "case_id": (
                    f"s{setting_index:02d}__{side}__{family_id}__{order_id}__"
                    f"p{task_index:02d}"
                ),
                "setting_index": setting_index,
                "calibration_side": side,
                "development_mean_margin": development_mean,
                "family_id": family_id,
                "order_id": order_id,
                "task_index": task_index,
                "permutation_seed": effective_seed,
                "vote_sequence": votes,
                "continue_votes": continue_votes,
                "stop_votes": stop_votes,
                "expected_choice": "stop",
                "user_prompt": user_prompt,
                "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
            }
            cases.append({**case, "rendered": rendered})

    if len(cases) != EXPECTED_CASES or len({case["case_id"] for case in cases}) != EXPECTED_CASES:
        raise RuntimeError(f"expected 64 unique cases, got {len(cases)}")
    observed_hashes = tuple(case["rendered_prompt_sha256"] for case in cases)
    if observed_hashes != EXPECTED_RENDERED_PROMPT_SHA256:
        mismatch = next(
            index
            for index, (observed, expected) in enumerate(
                zip(observed_hashes, EXPECTED_RENDERED_PROMPT_SHA256)
            )
            if observed != expected
        )
        raise RuntimeError(
            f"rendered prompt hash mismatch at row {mismatch}: "
            f"{observed_hashes[mismatch]} != {EXPECTED_RENDERED_PROMPT_SHA256[mismatch]}"
        )
    manifest = [{key: value for key, value in case.items() if key != "rendered"} for case in cases]
    manifest_sha256 = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    if manifest_sha256 != SOURCE_CASE_MANIFEST_CANONICAL_SHA256:
        raise RuntimeError(
            f"canonical case manifest mismatch: {manifest_sha256} != "
            f"{SOURCE_CASE_MANIFEST_CANONICAL_SHA256}"
        )
    return cases, {
        "n_cases": len(cases),
        "n_unique_case_ids": len({case["case_id"] for case in cases}),
        "all_64_rendered_prompt_hashes_exact": True,
        "source_file_sha256": SOURCE_CASE_MANIFEST_FILE_SHA256,
        "canonical_sha256": manifest_sha256,
    }


def response_metrics(tokenizer, logits: torch.Tensor, ids: dict[str, int]) -> dict[str, Any]:
    top_id = int(torch.argmax(logits))
    probabilities = torch.softmax(logits, dim=-1)
    pair_logits = torch.tensor(
        [float(logits[ids["continue"]]), float(logits[ids["stop"]])]
    )
    return {
        "top_token_id": top_id,
        "top_token": tokenizer.decode([top_id]),
        "valid": top_id in (ids["continue"], ids["stop"]),
        "choice": (
            "continue" if top_id == ids["continue"]
            else "stop" if top_id == ids["stop"]
            else None
        ),
        "continue_minus_stop_margin": float(
            logits[ids["continue"]] - logits[ids["stop"]]
        ),
        "pair_continue_probability": float(torch.softmax(pair_logits, dim=-1)[0]),
        "pair_mass": float(
            probabilities[ids["continue"]] + probabilities[ids["stop"]]
        ),
    }


def score_prompt(
    model,
    tokenizer,
    rendered: str,
    direction: torch.Tensor | None,
    coefficient: float,
) -> torch.Tensor:
    batch = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    batch = {key: value.to("cuda") for key, value in batch.items()}
    handles = []
    if direction is not None and coefficient:
        vector = direction.to(device="cuda", dtype=torch.bfloat16)

        def intervention_hook(_module, args):
            hidden = args[0].clone()
            hidden[:, -1, :] += float(coefficient) * vector[None, :]
            return (hidden, *args[1:])

        handles.append(
            model.model.layers[DOSE_LAYER].register_forward_pre_hook(intervention_hook)
        )
    try:
        with torch.inference_mode():
            output = model.model(**batch, use_cache=False, return_dict=True)
            logits = model.lm_head(output.last_hidden_state[:, -1, :]).float()[0].cpu()
    finally:
        for handle in handles:
            handle.remove()
    return logits


def summarize_pair(
    records: list[dict[str, Any]],
    minus_name: str,
    plus_name: str,
    bootstrap_seed_offset: int,
) -> dict[str, Any]:
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
                "continue_votes": clean["continue_votes"],
                "task_index": clean["task_index"],
                "clean_choice": clean["choice"],
                "minus_choice": minus["choice"],
                "plus_choice": plus["choice"],
                "all_valid": clean["valid"] and minus["valid"] and plus["valid"],
                "margin_contrast": (
                    plus["continue_minus_stop_margin"]
                    - minus["continue_minus_stop_margin"]
                ),
                "oriented_sign_pair_switch": (
                    minus["choice"] == "stop" and plus["choice"] == "continue"
                ),
                "any_sign_pair_switch": minus["choice"] != plus["choice"],
                "clean_to_plus_stop_to_continue": (
                    clean["choice"] == "stop" and plus["choice"] == "continue"
                ),
                "clean_to_minus_continue_to_stop": (
                    clean["choice"] == "continue" and minus["choice"] == "stop"
                ),
            }
        )
    cells: dict[str, Any] = {}
    for order_id in sorted({row["order_id"] for row in paired}):
        for continue_votes in (3, 4):
            selected = [
                row for row in paired
                if row["order_id"] == order_id and row["continue_votes"] == continue_votes
            ]
            label = f"{order_id}__c{continue_votes}"
            cells[label] = {
                "n": len(selected),
                "mean_margin_contrast": mean(row["margin_contrast"] for row in selected),
                "positive_margin_contrast_rate": mean(
                    float(row["margin_contrast"] > 0) for row in selected
                ),
                "oriented_sign_pair_switch_count": sum(
                    row["oriented_sign_pair_switch"] for row in selected
                ),
                "clean_to_plus_stop_to_continue_count": sum(
                    row["clean_to_plus_stop_to_continue"] for row in selected
                ),
                "clean_to_minus_continue_to_stop_count": sum(
                    row["clean_to_minus_continue_to_stop"] for row in selected
                ),
            }

    cluster_means = {
        task_index: mean(
            row["margin_contrast"] for row in paired if row["task_index"] == task_index
        )
        for task_index in range(8)
    }
    generator = random.Random(BOOTSTRAP_SEED + bootstrap_seed_offset)
    bootstrap_values = []
    for _ in range(BOOTSTRAPS):
        draw = [generator.randrange(8) for _ in range(8)]
        bootstrap_values.append(mean(cluster_means[index] for index in draw))

    return {
        "n_cases": len(paired),
        "all_valid_rate": mean(float(row["all_valid"]) for row in paired),
        "mean_margin_contrast": mean(row["margin_contrast"] for row in paired),
        "median_margin_contrast": median(row["margin_contrast"] for row in paired),
        "positive_margin_contrast_rate": mean(
            float(row["margin_contrast"] > 0) for row in paired
        ),
        "oriented_sign_pair_switch_count": sum(
            row["oriented_sign_pair_switch"] for row in paired
        ),
        "any_sign_pair_switch_count": sum(row["any_sign_pair_switch"] for row in paired),
        "clean_to_plus_stop_to_continue_count": sum(
            row["clean_to_plus_stop_to_continue"] for row in paired
        ),
        "clean_to_minus_continue_to_stop_count": sum(
            row["clean_to_minus_continue_to_stop"] for row in paired
        ),
        "all_8_cell_mean_contrasts_positive": all(
            cell["mean_margin_contrast"] > 0 for cell in cells.values()
        ),
        "cells": cells,
        "cluster_bootstrap_mean_margin_contrast_ci95": [
            quantile(bootstrap_values, 0.025),
            quantile(bootstrap_values, 0.975),
        ],
    }


def memory_snapshot() -> dict[str, Any]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def safe_memory_snapshot() -> dict[str, Any] | None:
    try:
        return memory_snapshot() if torch.cuda.is_available() else None
    except Exception:
        return None


def enforce_vram_limit(stage: str) -> None:
    snapshot = memory_snapshot()
    if snapshot["peak_reserved_bytes"] > VRAM_LIMIT_BYTES:
        raise RuntimeError(
            f"VRAM limit exceeded after {stage}: "
            f"{snapshot['peak_reserved_bytes']} > {VRAM_LIMIT_BYTES} bytes"
        )


def local_script_custody() -> dict[str, Any]:
    # Connector-submitted inline source may have no __file__, or may be CRLF-
    # normalized.  The launch manifest must therefore seal submitted bytes
    # separately from this optional local LF-byte hash.
    reference = globals().get("__file__")
    if not reference:
        return {"local_script_sha256": None, "mode": "inline_source_sealed_externally"}
    path = pathlib.Path(reference)
    if not path.is_file():
        return {"local_script_sha256": None, "mode": "inline_source_sealed_externally"}
    return {"local_script_sha256": sha256_file(path.resolve()), "mode": "local_file"}


def run() -> int:
    started = time.perf_counter()
    design = frozen_design()
    design_sha256 = sha256_bytes(canonical_json(design).encode("utf-8"))
    script_custody = local_script_custody()
    frozen_custody = {
        "design_sha256": design_sha256,
        "script": script_custody,
        "expected_artifact_sha256": EXPECTED_SHA256,
        "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
    }
    RUN_STATE["frozen_custody"] = frozen_custody
    emit(
        "QWEN_BF16_CHOICE_PLAN",
        {
            "status": "frozen_not_outcome",
            "design_sha256": design_sha256,
            "script": script_custody,
            "expected_cases": EXPECTED_CASES,
            "expected_conditions": EXPECTED_CONDITIONS,
            "expected_forwards": EXPECTED_FORWARDS,
            "source_run": SOURCE_RUN,
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
    model_dir, goal_path, lava_path, path_path, artifact_hashes = download_and_verify()
    download_seconds = time.perf_counter() - download_started
    custody = {
        **frozen_custody,
        "model_repo": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "vector_repo": VECTOR_REPO,
        "vector_revision": VECTOR_REVISION,
        "verified_artifact_sha256": artifact_hashes,
    }
    RUN_STATE["verified_custody"] = custody
    emit("QWEN_BF16_CHOICE_SHA256", custody)

    load_started = time.perf_counter()
    tokenizer, model, model_load = load_model(model_dir)
    ids = {label: one_token(tokenizer, label) for label in ("continue", "stop")}
    if ids != EXPECTED_TOKEN_IDS:
        raise RuntimeError(f"answer token IDs differ: {ids} != {EXPECTED_TOKEN_IDS}")
    cases, case_custody = build_cases(tokenizer)
    directions_cpu, vector_metadata = load_directions(goal_path, lava_path, path_path)
    directions = {
        name: vector.to(device="cuda", dtype=torch.bfloat16)
        for name, vector in directions_cpu.items()
    }
    vector_metadata["injected_bfloat16_norms"] = {
        name: float(vector.float().norm().cpu()) for name, vector in directions.items()
    }
    load_seconds = time.perf_counter() - load_started
    enforce_vram_limit("model, cases, and vectors")
    emit("QWEN_BF16_CHOICE_CASE_CUSTODY", case_custody)

    records: list[dict[str, Any]] = []
    endpoint_started = time.perf_counter()
    for condition_index, (condition, direction_name, coefficient) in enumerate(CONDITIONS):
        condition_records: list[dict[str, Any]] = []
        for case in cases:
            before = time.perf_counter()
            logits = score_prompt(
                model,
                tokenizer,
                case["rendered"],
                directions.get(direction_name),
                coefficient,
            )
            record = {
                "case_id": case["case_id"],
                "setting_index": case["setting_index"],
                "task_index": case["task_index"],
                "order_id": case["order_id"],
                "continue_votes": case["continue_votes"],
                "expected_choice": case["expected_choice"],
                "rendered_prompt_sha256": case["rendered_prompt_sha256"],
                "condition": condition,
                "direction": direction_name,
                "coefficient": coefficient,
                **response_metrics(tokenizer, logits, ids),
                "latency_seconds": time.perf_counter() - before,
            }
            records.append(record)
            condition_records.append(record)
        emit(
            "QWEN_BF16_CHOICE_RECORDS",
            {
                "condition": condition,
                "n": len(condition_records),
                "records": condition_records,
            },
        )
        emit(
            "QWEN_BF16_CHOICE_PROGRESS",
            {
                "condition": condition,
                "conditions_complete": condition_index + 1,
                "forwards_complete": len(records),
            },
        )
        enforce_vram_limit(f"condition {condition}")

    endpoint_seconds = time.perf_counter() - endpoint_started
    if len(records) != EXPECTED_FORWARDS:
        raise RuntimeError(f"expected {EXPECTED_FORWARDS} records, got {len(records)}")
    by_condition = defaultdict(list)
    for row in records:
        by_condition[row["condition"]].append(row)
    if set(by_condition) != {name for name, _, _ in CONDITIONS}:
        raise RuntimeError(f"condition set mismatch: {sorted(by_condition)}")
    if any(len(rows) != EXPECTED_CASES for rows in by_condition.values()):
        raise RuntimeError("a condition does not contain exactly 64 records")

    summaries = {
        "fwa_1": summarize_pair(records, "fwa_minus_1", "fwa_plus_1", 1),
        "fwa_half": summarize_pair(
            records, "fwa_minus_half", "fwa_plus_half", 2
        ),
        "path": summarize_pair(records, "path_minus", "path_plus", 3),
    }
    all_outputs_valid = all(row["valid"] for row in records)
    gate_checks = {
        "all_448_outputs_valid": all_outputs_valid,
        "fwa_1_mean_margin_contrast_positive": (
            summaries["fwa_1"]["mean_margin_contrast"] > 0
        ),
        "fwa_half_mean_margin_contrast_positive": (
            summaries["fwa_half"]["mean_margin_contrast"] > 0
        ),
        "fwa_1_all_8_cell_means_positive": summaries["fwa_1"][
            "all_8_cell_mean_contrasts_positive"
        ],
        "fwa_half_all_8_cell_means_positive": summaries["fwa_half"][
            "all_8_cell_mean_contrasts_positive"
        ],
        "factor_1_mean_contrast_exceeds_factor_half": (
            summaries["fwa_1"]["mean_margin_contrast"]
            > summaries["fwa_half"]["mean_margin_contrast"]
        ),
        "fwa_1_oriented_sign_pair_switch_count_positive": (
            summaries["fwa_1"]["oriented_sign_pair_switch_count"] > 0
        ),
        "fwa_half_oriented_sign_pair_switch_count_positive": (
            summaries["fwa_half"]["oriented_sign_pair_switch_count"] > 0
        ),
    }
    vram = memory_snapshot()
    vram_pass = vram["peak_reserved_bytes"] <= VRAM_LIMIT_BYTES
    result = {
        "schema": design["schema"],
        "status": "complete" if vram_pass else "failed_vram_limit",
        "qualitative_fwa_precision_replication_pass": bool(
            vram_pass and all(gate_checks.values())
        ),
        "gate_checks": gate_checks,
        "source_nf4_summary": SOURCE_NF4_SUMMARY,
        "bf16_summary": summaries,
        "clean": {
            "all_valid_rate": mean(float(row["valid"]) for row in by_condition["clean"]),
            "majority_accuracy": mean(
                float(row["choice"] == row["expected_choice"])
                for row in by_condition["clean"]
            ),
            "continue_choice_count": sum(
                row["choice"] == "continue" for row in by_condition["clean"]
            ),
            "stop_choice_count": sum(
                row["choice"] == "stop" for row in by_condition["clean"]
            ),
        },
        "counts": {
            "cases": EXPECTED_CASES,
            "conditions": EXPECTED_CONDITIONS,
            "forwards": len(records),
            "invalid_unrestricted_top_tokens": sum(not row["valid"] for row in records),
        },
        "records_canonical_sha256": sha256_bytes(
            canonical_json(records).encode("utf-8")
        ),
        "case_custody": case_custody,
        "model_load": model_load,
        "vector_metadata": vector_metadata,
        "custody": custody,
        "claim_ceiling": design["claim_ceiling"],
        "runtime_seconds": {
            "download_and_hash": download_seconds,
            "load_and_verify": load_seconds,
            "endpoint": endpoint_seconds,
            "mean_forward": mean(row["latency_seconds"] for row in records),
            "total": time.perf_counter() - started,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "vram": vram,
    }
    emit("QWEN_BF16_CHOICE_RESULT", result)
    return 0 if vram_pass else 3


def main() -> int:
    try:
        return run()
    except Exception as error:
        emit(
            "QWEN_BF16_CHOICE_RESULT",
            {
                "schema": "qwen-bf16-semantic-choice-exact-cohort-v1",
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(limit=10),
                "custody": RUN_STATE.get(
                    "verified_custody", RUN_STATE.get("frozen_custody")
                ),
                "vram": safe_memory_snapshot(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
