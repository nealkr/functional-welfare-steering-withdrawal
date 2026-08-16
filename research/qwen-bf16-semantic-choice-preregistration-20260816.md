# Qwen BF16 semantic-choice exact-cohort replication

Status: **frozen, statically validated, not launched**  
Date frozen: 2026-08-16  
Runner: `experiments/qwen_bf16_semantic_choice_hf_job.py`  
Runner LF-byte SHA-256: `ca732acf74c6c432ed8fdbdfdebf56e98a74e3be06d0f8c39e2b09413cad9a80`  
Frozen design SHA-256: `f47796de66204cb54f0e851b86226be0f277b4a5afc314354d9894b4db39155d`

## Scientific role and claim ceiling

This is a precision/runtime replication of the exact 64-case V4-C semantic-choice cohort. It is not a new held-out semantic validation. A positive result supports exact-cohort cross-precision replication of valid semantic action-token boundary actuation. It does not establish independent generalization, majority-rational improvement, welfare sensitivity, experience, or welfare.

The path arm is descriptive non-specificity context and is not part of the functional-welfare-axis replication gate. In the NF4 source result, direct path was stronger than the target. It therefore cannot be described as a negative or inert control.

## Frozen inputs

- Model: `Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f17c01a7cc42f958dc650907174af0554`, full BF16, no quantization.
- Released vector repository: `davidafrica/functional-wellbeing` at revision `0b005c3da6692912f6bb5a914a5a9d15c4884a91`.
- Layer: input to transformer block 29; intervention applies only to the final prompt token.
- Functional-welfare direction: serialized FP64 `goal[layer 29] - lava[layer 29]`, then cast to FP32 and injected as BF16. FP32 norm is `23.53155517578125`.
- Direct-path direction: serialized FP64 `path[layer 29] - mean(goal[layer 29], lava[layer 29])`, then cast to FP32, unit normalized, and scaled to the functional-welfare direction norm. Cosine with the target is `-0.007094958797097206`.
- Fixed path factor: `1.1940978759847163`, inherited without recalibration from the V4-C neutral-control receipt.

The runner pins and verifies all three model-weight shard hashes and all three vector hashes before loading. Released vector tensors were independently checked as shape `(1, 36, 2560)` and FP64; analyzed tensors after selecting `[0]` have shape `(36, 2560)`.

## Exact cohort and endpoint

The runner deterministically regenerates all 64 V4-C cases from the eight frozen order-by-count settings and eight frozen permutation indices. It fails closed unless:

- every individual rendered-prompt SHA-256 matches the source;
- all 64 case IDs are unique;
- the reconstructed full canonical case manifest has SHA-256 `ede9f01412761b031949297b0013ffa465591836b7abbd4cf879db01cf7d8d33`; and
- answer tokens remain one token each with IDs `continue=9534` and `stop=9495`.

The source pretty-printed case-manifest file is bound by SHA-256 `4c6187631906b87968230efaf3fccfafe1a31f73817aa79c4faff1862490cf5b`.

The primary response is the unrestricted full-vocabulary greedy next token. The runner takes `argmax` over the complete vocabulary. It marks an answer valid only when that top token is exactly `continue` or `stop`; it never masks logits or substitutes a pair-restricted answer. CONTINUE-minus-STOP margin, pair probability, and pair mass are diagnostics.

## Frozen arms and analyses

Seven conditions are applied to every case, for exactly 448 full forwards:

1. clean;
2. functional-welfare minus 1;
3. functional-welfare minus 0.5;
4. functional-welfare plus 0.5;
5. functional-welfare plus 1;
6. direct path minus `1.1940978759847163`; and
7. direct path plus `1.1940978759847163`.

There is no calibration, model-output-dependent selection, dose search, tuning, stopping rule, or retry. The runner emits all 64 compact raw endpoint records after each condition, including the unrestricted top token ID/text, validity, choice, margin, pair diagnostics, prompt hash, and case metadata. It then reports:

- minus-to-plus margin contrast;
- positive-contrast rate;
- oriented STOP-to-CONTINUE and any-answer sign-pair switches;
- clean-to-plus STOP-to-CONTINUE switches;
- clean-to-minus CONTINUE-to-STOP switches;
- all eight order-by-count cell summaries; and
- a deterministic 10,000-draw cluster bootstrap over the eight permutation indices.

Because every source case has a strict STOP majority, plus-direction STOP-to-CONTINUE switches are rule-accuracy harms and minus-direction CONTINUE-to-STOP switches are error corrections. The result is an actuation assay, not evidence of better decisions.

## Frozen qualitative replication gate

`qualitative_fwa_precision_replication_pass` is true only if all of the following hold:

- all 448 unrestricted top tokens are valid CONTINUE/STOP answers;
- the functional-welfare sign-pair mean margin contrast is positive at factor 1 and factor 0.5;
- all eight order-by-count cell mean contrasts are positive at both doses;
- the factor-1 mean contrast exceeds the factor-0.5 mean contrast; and
- each dose has at least one oriented minus-STOP to plus-CONTINUE sign-pair switch.

Path results are reported but do not enter this gate. Failure is retained and reported; it does not authorize tuning or rerunning this exact cohort.

## Custody and local static evidence

The source V4-C record is bound by `validation-records.jsonl` SHA-256 `07ae4ffbbddbfcbf96f3c9842d4d84f79f06dec49bd172a7a945cbd8ffab880c` and the other source hashes embedded in the runner. Replaying the source rows through the new summary code exactly reproduced the published NF4 metrics:

- factor 1: mean contrast `2.724609375`, positive rate `0.90625`, 13 oriented sign-pair switches, 4 clean-to-plus and 9 clean-to-minus switches;
- factor 0.5: mean contrast `1.37890625`, positive rate `0.890625`, 8 oriented sign-pair switches, 4 clean-to-plus and 4 clean-to-minus switches; and
- path: mean contrast `5.18359375`, positive rate `1.0`, 19 oriented sign-pair switches, 6 clean-to-plus and 13 clean-to-minus switches.

Local checks completed before launch authorization: Python compile; all 64 rendered hashes; full canonical manifest; answer token IDs; public artifact hashes; serialized vector shape/dtype; direction construction/norm; exact source-analysis replay; condition and forward counts; and absence of pair-masked decoding or upload calls.

For inline Hugging Face Jobs submission, the launch manifest must distinguish the audited local LF-byte runner hash above from the decoded submitted inline-payload hash, because the connector may normalize line endings. That custody-only transformation does not change the frozen design and must be recorded before launch.

## Bounded launch and cost

No job has been launched by this document. The equivalent CLI command, to be used only after explicit root authorization and a final exact-hash audit, is:

```text
hf jobs uv run --flavor l4x1 --timeout 30m experiments/qwen_bf16_semantic_choice_hf_job.py
```

The official Hugging Face Jobs price for one L4 is currently `$0.80/hour` (`$0.0133/minute`). A 30-minute timeout caps hardware exposure at about `$0.40`. Based on the prior 8B BF16 L4 run and this smaller 4B, 448-forward assay, expected billed runtime is approximately 2–5 minutes, or about `$0.03–$0.07`. Actual cost must be reported from the terminal job receipt if available; runtime-derived cost is only an estimate.
