# Qwen BF16 semantic-choice static audit

Status: **GO; immutable runner; not launched at audit completion**  
Audit date: 2026-08-16  
Runner: `experiments/qwen_bf16_semantic_choice_hf_job.py`  
Immutable local LF-byte SHA-256: `ca732acf74c6c432ed8fdbdfdebf56e98a74e3be06d0f8c39e2b09413cad9a80`  
Frozen design SHA-256: `f47796de66204cb54f0e851b86226be0f277b4a5afc314354d9894b4db39155d`  
Preregistration: `research/qwen-bf16-semantic-choice-preregistration-20260816.md`  
Preregistration SHA-256: `f6cd38ef9b733d644f3012d5288abf3fe6c076eaff680c973cfeda4116c49348`

## Independent verdict

The initial independent exact-hash review returned **GO with no blockers** before launch. This is the historical prelaunch verdict; a postlaunch correction identifying three frozen-specification deviations is preserved below. The initial review separately reverified the immutable runner and design hashes and passed:

- exact 64-case and rendered-prompt custody;
- seven conditions by 64 cases, exactly 448 forwards;
- unrestricted full-vocabulary greedy top-token evaluation;
- the transformer-block-29, final-prompt-token pre-hook;
- functional-welfare and direct-path constructions and the frozen path factor;
- model/vector revision and artifact-hash pins;
- switch, cell, bootstrap, and qualitative-gate logic;
- bounded claim language and path-control role;
- the prelaunch expectation that raw-record payloads and errors would persist in hosted logs;
- inline `__file__` fallback; and
- one-L4 fit, launch command, and cost ceiling.

No hosted job was launched by the audit.

## Local static receipt

| Check | Receipt |
|---|---|
| Python syntax | PASS |
| Runner bytes | `41,193` |
| Runner LF SHA-256 | `ca732acf74c6c432ed8fdbdfdebf56e98a74e3be06d0f8c39e2b09413cad9a80` |
| Frozen design SHA-256 | `f47796de66204cb54f0e851b86226be0f277b4a5afc314354d9894b4db39155d` |
| Reconstructed cases | 64 unique case IDs, PASS |
| Individual rendered hashes | 64/64 exact, PASS |
| Canonical source manifest | `ede9f01412761b031949297b0013ffa465591836b7abbd4cf879db01cf7d8d33`, exact |
| Source case-manifest file | `4c6187631906b87968230efaf3fccfafe1a31f73817aa79c4faff1862490cf5b`, bound |
| Answer tokens | `continue=9534`, `stop=9495`, both one token, PASS |
| Conditions | clean; FWA ±0.5/±1; path ±1.1940978759847163, exact |
| Forward count | 7 × 64 = 448, exact |
| Vector files | all three public revision-pinned hashes exact |
| Serialized tensors | goal/lava/path each `(1, 36, 2560)` FP64 |
| FWA construction | serialized goal-minus-lava at layer 29, then FP32; norm `23.53155517578125` |
| Path construction | serialized path-minus-goal/lava midpoint, then FP32 and norm match; norm `23.531553268432617` |
| Target/path cosine | `-0.007094958797097206` |
| Endpoint | one full-vocabulary `argmax`; no logit masking, top-k, sampling, or substituted pair answer |
| External effects | public downloads only; no upload/publish/message |
| Remote job | not launched |

The full script was not executed locally because CUDA was available but the laptop GPU has only 8,188 MiB, below the frozen 24 GiB L4 target. A direct-script smoke began caching the public pinned artifacts and was interrupted after download, before model load or any scientific forward. This does not consume or modify the cohort and creates no hosted charge.

## Exact analysis replay

The new BF16 summary implementation was applied to the frozen NF4 source rows as a static oracle test. It reproduced all target quantities exactly:

| Pair | Mean contrast | Positive rate | Oriented sign-pair switches | Any switches | Clean→plus | Clean→minus | Eight cells positive |
|---|---:|---:|---:|---:|---:|---:|---|
| FWA factor 1 | 2.724609375 | 0.90625 | 13 | 13 | 4 | 9 | yes |
| FWA factor 0.5 | 1.37890625 | 0.890625 | 8 | 8 | 4 | 4 | yes |
| Direct path | 5.18359375 | 1.0 | 19 | 19 | 6 | 13 | yes |

This establishes that the new code preserves the intended clean-to-sign and symmetric sign-pair definitions. It is not a BF16 result.

## Launch boundary

Only the following frozen workload is approved by this audit:

```text
hf jobs uv run --flavor l4x1 --timeout 30m experiments/qwen_bf16_semantic_choice_hf_job.py
```

The official current L4 rate is `$0.80/hour` (`$0.0133/minute`), giving a hard 30-minute hardware ceiling of about `$0.40` and an expected 2–5-minute charge of about `$0.03–$0.07`.

At launch, seal the exact submitted inline bytes. If the connector converts the local LF runner to CRLF, record both the audited local LF hash and the decoded submitted-payload hash, with the exact line-ending transformation. Do not treat those hashes as interchangeable. Capture complete logs, the terminal job receipt, and any inspectable cost. Do not retry automatically after a valid terminal result.

## Postlaunch custody and audit correction

The historical prelaunch version of this audit note had SHA-256 `ba8639d72ec52bdee8459c227a7172f2390cd0582d58972483e4a960cfe7138b`. The authorized job subsequently completed under ID `6a81fdb8c97db76cbdf33362`. Its decoded submitted inline payload was byte-identical to the audited local LF runner: 41,193 bytes, 979 LF line endings, no CRLF conversion, and SHA-256 `ca732acf74c6c432ed8fdbdfdebf56e98a74e3be06d0f8c39e2b09413cad9a80`.

Hosted log transport segmented each long record-payload line at approximately 16,384-byte boundaries. It did not lose the continuation bytes. Deterministic reassembly recovered all seven payloads, all `7 × 64 = 448` row records, and a canonical row SHA-256 of `e3f7f45e115e28f830c66a157b5c86b48fa3c3f233607c1dc871a626d1c3c52f`, exactly matching the hash computed inside the terminal job result. Thus the prelaunch persistence expectation succeeded, but consumers must reassemble unprefixed continuation segments rather than parse physical log lines independently.

The raw log also records CuBLAS nondeterminism warnings because `CUBLAS_WORKSPACE_CONFIG` was unset. The recovered 448 rows are exact for this execution, but bitwise rerun determinism is unverified. This is a reproducibility limitation, not a fourth frozen-specification deviation.

After launch, the independent reviewer superseded the initial no-blocker wording with three exact frozen-specification deviations:

1. The frozen design records only seed `26081682`, while the preregistration binds that design hash but does not separately state per-pair seed semantics. The implementation uses effective arm seeds `26081683`, `26081684`, and `26081685`, which were not emitted. The qualitative gate does not use bootstrap intervals; both implementation-seed and recorded-seed-applied-to-each-pair sensitivity intervals are preserved in the terminal audit.
2. The preregistered `all_448_outputs_valid` criterion explicitly includes the 128 path records and is conjunctive in the reported pass, while other frozen prose says path is outside the FWA gate. All path records were valid, so this path gate-scope contradiction did not change the observed pass.
3. The implementation additionally conjuncts the VRAM limit with the reported qualitative pass although the frozen scientific-gate list omits it. Peak reserved VRAM passed, so this VRAM gate-scope contradiction did not change the observed pass.

These deviations are not retroactively repaired. They lower the custody description from a pristine frozen-specification seal to a positive, independently recomputed exact-cohort result with disclosed frozen-specification deviations. No retry is authorized.
