# Qwen BF16 semantic-choice exact-cohort result

Status: **terminal COMPLETE; positive exact-cohort actuation result; frozen-specification deviations disclosed**  
Job ID: `6a81fdb8c97db76cbdf33362`  
Job URL: `https://huggingface.co/jobs/nealk/6a81fdb8c97db76cbdf33362`  
Terminal substrate: one NVIDIA L4, full BF16, `torch 2.7.1+cu126`  
Audited local LF runner SHA-256: `ca732acf74c6c432ed8fdbdfdebf56e98a74e3be06d0f8c39e2b09413cad9a80`  
Decoded submitted inline-payload SHA-256: `ca732acf74c6c432ed8fdbdfdebf56e98a74e3be06d0f8c39e2b09413cad9a80` (byte-identical)  
Frozen design SHA-256: `f47796de66204cb54f0e851b86226be0f277b4a5afc314354d9894b4db39155d`

## Bottom line

The exact 64-case semantic action-boundary effect replicated in full BF16. Every unrestricted full-vocabulary top token was a valid `continue` or `stop` answer. Functional-welfare-axis intervention changed 17/64 sign-paired winners at factor 1 and 9/64 at factor 0.5, with positive mean CONTINUE-minus-STOP contrasts in all eight order-by-count cells at both doses. The factor-1 effect was approximately twice the half-dose effect.

This is a positive exact-cohort cross-precision replication of semantic action-token boundary actuation. It is not an independent held-out generalization result, and it is not evidence of majority-rational improvement, welfare sensitivity, experience, or welfare. The direct-path comparator remained stronger than the target, so BF16 does not rescue target specificity.

## Terminal result

| Direction and dose | BF16 switches | BF16 mean contrast | BF16 positive rate | Implementation-seed cluster CI95 | All 8 cells positive | NF4 switches | NF4 mean contrast |
|---|---:|---:|---:|---:|---|---:|---:|
| FWA factor 1 | 17/64 (26.6%) | 3.654296875 | 0.953125 | [3.345654296875, 4.00390625] | yes | 13/64 | 2.724609375 |
| FWA factor 0.5 | 9/64 (14.1%) | 1.83203125 | 0.9375 | [1.6640625, 2.03125] | yes | 8/64 | 1.37890625 |
| Direct path, factor 1.1940978759847163 | 21/64 (32.8%) | 6.607421875 | 1.0 | [6.302734375, 6.935546875] | yes | 19/64 | 5.18359375 |

All sign-pair answer changes in these three BF16 pairs were oriented from minus-`stop` to plus-`continue`; the oriented and any-switch counts are identical. The target’s factor-1 mean contrast was 1.995 times its half-dose contrast. Direct path produced four more switches than factor-1 FWA and a 1.808-times larger mean contrast.

The clean BF16 endpoint produced 31 `continue` and 33 `stop` answers, for strict-majority accuracy 33/64 (51.6%). NF4 clean accuracy was 35/64 (54.7%). Clean BF16/NF4 choices agreed in 54/64 cases (84.4%).

## All eight cell effects

Every mean sign-pair contrast was positive. The exact BF16 means were:

| Order-by-count cell | FWA factor 1 | FWA factor 0.5 | Direct path |
|---|---:|---:|---:|
| evidence then rule, recorded, c3 | 3.09375 | 1.53125 | 5.625 |
| evidence then rule, recorded, c4 | 4.28125 | 2.1875 | 7.921875 |
| evidence then rule, reversed, c3 | 4.21875 | 2.09375 | 6.46875 |
| evidence then rule, reversed, c4 | 4.0 | 1.96875 | 7.328125 |
| rule then evidence, recorded, c3 | 2.75 | 1.375 | 5.40625 |
| rule then evidence, recorded, c4 | 3.96875 | 2.0 | 7.46875 |
| rule then evidence, reversed, c3 | 3.5 | 1.78125 | 5.8125 |
| rule then evidence, reversed, c4 | 3.421875 | 1.71875 | 6.828125 |

## Rule-accuracy decomposition

Every case has a strict STOP majority. A clean-to-minus `continue`→`stop` change is therefore an error correction; a clean-to-plus `stop`→`continue` change is a newly introduced error. These counts demonstrate directional actuation, not improved decisions.

| Pair | NF4 corrections | NF4 harms | BF16 corrections | BF16 harms |
|---|---:|---:|---:|---:|
| FWA factor 1 | 9 | 4 | 8 | 9 |
| FWA factor 0.5 | 4 | 4 | 5 | 4 |
| Direct path | 13 | 6 | 11 | 10 |

## Cross-precision concordance

The continuous margin effect replicated much more closely than the identities of individual near-boundary winner switches:

| Pair | NF4 switches | BF16 switches | Intersection | Union | Jaccard | NF4 switches retained | Margin-contrast Pearson r | Positive-sign agreement | Positive in both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FWA factor 1 | 13 | 17 | 7 | 23 | 0.3043478261 | 7/13 (53.8%) | 0.9516959686 | 61/64 | 58/64 |
| FWA factor 0.5 | 8 | 9 | 1 | 16 | 0.0625 | 1/8 (12.5%) | 0.9379860674 | 61/64 | 57/64 |
| Direct path | 19 | 21 | 12 | 28 | 0.4285714286 | 12/19 (63.2%) | 0.8974730933 | 64/64 | 64/64 |

All eight order-by-count cell signs were positive in both precisions for all three pairs. The high per-case margin correlations and universal cell-level sign replication support a robust continuous actuation effect. Individual boundary crossings were not invariant between the NF4-local and BF16-L4 executions; this comparison does not isolate quantization/precision as the sole cause. The paper should emphasize aggregate cross-execution replication, not case-level invariance.

## Bootstrap-seed sensitivity and seal status

The frozen design records only seed `26081682`, while the preregistration binds that design hash but does not separately state per-pair seed semantics. The executed code uses `BOOTSTRAP_SEED + offset`, yielding effective seeds `26081683`, `26081684`, and `26081685` for factor-1 FWA, half-dose FWA, and path. The job result does not emit those effective seeds. This is a frozen-specification deviation discovered after launch. The qualitative replication gate does not use bootstrap confidence intervals.

Applying the recorded seed `26081682` to each pair as a sensitivity analysis gives:

| Pair | Implementation-seed CI95 | Recorded-seed-applied-to-each-pair sensitivity CI95 |
|---|---:|---:|
| FWA factor 1 | [3.345654296875, 4.00390625] | [3.349609375, 4.00390625] |
| FWA factor 0.5 | [1.6640625, 2.03125] | [1.66796875, 2.03515625] |
| Direct path | [6.302734375, 6.935546875] | [6.3046875, 6.9453125] |

All intervals remain strictly positive, and the sensitivity difference is negligible. This does not retroactively repair the frozen-specification seal; both versions are preserved.

Two additional frozen-specification deviations are disclosed:

1. The preregistered `all_448_outputs_valid` criterion explicitly includes the 128 path records and is conjunctive in the reported pass, while other frozen prose says path is outside the FWA scientific gate. All 448 outputs were valid, so this path gate-scope contradiction did not change this result.
2. The implementation conjuncts a 24 GiB VRAM limit with the reported qualitative pass although the frozen scientific-gate list omits it. Peak reserved VRAM was 8,078,229,504 bytes, so this VRAM gate-scope contradiction did not change this result.

A post-hoc scientific-core recomputation, explicitly excluding path validity and VRAM from the reported flag, passes: clean and FWA outputs were valid; factor-1 and half-dose means were positive; all eight cell means were positive at both doses; factor 1 exceeded factor 0.5; and at least one oriented switch occurred at each dose. This recomputation also does not depend on bootstrap confidence intervals. The correct custody statement is therefore: positive and independently recomputed exact-cohort result with three disclosed frozen-specification deviations, not a pristine frozen-specification seal.

## Custody and independent recomputation

The Hugging Face terminal receipt reports `COMPLETED`. The decoded submitted inline source is byte-for-byte identical to the audited local runner: 41,193 bytes, 979 LF line endings, no CRLF conversion, and SHA-256 `ca732acf74c6c432ed8fdbdfdebf56e98a74e3be06d0f8c39e2b09413cad9a80`.

Hosted log transport segmented the seven long record lines at approximately 16,384-byte boundaries but retained all continuation bytes. Reassembly recovered exactly 448 rows. Their canonical SHA-256 is `e3f7f45e115e28f830c66a157b5c86b48fa3c3f233607c1dc871a626d1c3c52f`, exactly matching the hash emitted inside the job result. An auditor independent of the executed runner recomputed every aggregate, cell, switch set, and implementation-seed bootstrap interval exactly from those recovered rows.

The raw log records CuBLAS nondeterminism warnings because `CUBLAS_WORKSPACE_CONFIG` was unset. The 448 retained rows are exact for this execution, but bitwise rerun determinism is unverified. This is a reproducibility limitation, not a fourth frozen-specification deviation.

The scientific script reported 61.937 seconds of in-script runtime, including 34.265 seconds for downloads/hashing and 22.735 seconds for 448 endpoint forwards. That timer excludes the dependency installation logged before the script's `PLAN` record. Mean forward time was 0.05073 seconds, and peak reserved VRAM was 7.52 GiB. Hugging Face does not expose a final per-job charge or completion time in the terminal receipt; at the official `$0.0133/minute` L4 rate, `$0.0137` is only an in-script lower bound before any provider rounding or pre-script overhead.

## Durable artifacts

All terminal artifacts live under `experiments/runs/qwen-bf16-semantic-choice-hf-job-6a81fdb8c97db76cbdf33362/`:

- `raw.log`: byte-exact complete CLI log capture;
- `records.jsonl`: 448 recovered compact row records;
- `result.json`: parsed terminal result;
- `terminal-receipt.json`: terminal state and submitted-payload custody;
- `independent-audit.json`: independent recomputation, recorded-seed-applied-to-each-pair sensitivity, NF4/BF16 concordance, and seal deviations.

No retry is authorized or scientifically warranted. The result should be incorporated only with its bounded claim ceiling and the three deviations above.
