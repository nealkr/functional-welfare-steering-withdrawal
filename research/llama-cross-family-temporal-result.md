# Llama Cross-Family Withdrawal Result

Date: 2026-08-16

## Verdict

The frozen Llama-3.1-8B run passed every preregistered numerical promotion
gate. On the plain-language semantic endpoints, the signed readout-intervention
contrast was 6.00586 while the earlier-only contrast was -0.03125. The opaque
aggregate contrasts were 1.59375 and 0.06250, respectively, but that aggregate
contains material K/M token bias and is secondary evidence.

This is a cross-model replication of the qualitative locus dissociation on the
same frozen endpoint families. It changes model family, tokenizer,
architecture, scale, training run, and extracted direction; it is not an
independent semantic generalization. The strongest precise claim is:

> Across separately extracted Qwen and Llama functional-welfare directions,
> contemporaneous intervention at the answer position had a large logit
> effect, whereas intervention confined to earlier positions had a much
> smaller effect at the answer under a fixed visible transcript.

The earlier condition intervened at 15--16 positions and the readout condition
at one position, both with per-position coefficients of +/-2. Their ratio is
therefore a schedule-specific locus comparison, not a fraction retained, an
equal-dose temporal decay rate, or a half-life. The result also does not
establish universal behavior across models, absence of rotated or nonlinear
traces, welfare, experience, or sentience.

## Frozen gates and outcomes

| Gate | Frozen threshold | Result | Pass |
|---|---:|---:|:---:|
| Mapping | at least 28/32 | 32/32 | yes |
| Semantic readout | 95% CI lower bound above 0 | 6.00586 [5.55078, 6.45117] | yes |
| Opaque readout | 95% CI lower bound above 0 | 1.59375 [1.34570, 1.80859] | yes |
| Semantic earlier/readout ratio | full 95% CI inside [-0.25, 0.25] | -0.00520 [-0.00768, -0.00262] | yes |
| Opaque earlier/readout ratio | full 95% CI inside [-0.25, 0.25] | 0.03922 [0.02909, 0.05058] | yes |

The semantic earlier effect was small and opposite-signed: -0.03125
[-0.04688, -0.01563]. The opaque aggregate earlier effect was small but
positive: 0.06250 [0.04492, 0.08008]. Thus “much smaller earlier-only effect”
is accurate for this assay. “Exactly zero,” “complete erasure,” “99.5%
decay,” and “3.9% retention” are not.

## Same-axis projection contrasts

| Endpoint group | Earlier (+2)-(-2) contrast at block-21 input, dose-position mean | Earlier (+2)-(-2) contrast at block-21 input, final answer position | Final/dose contrast magnitude |
|---|---:|---:|---:|
| Semantic | 18.45976 | 0.00763 | 0.041% |
| Opaque | 18.45935 | 0.00033 | 0.0018% |

These are not absolute activation levels. Each entry is the family-averaged
`+2 minus -2` contrast in projection onto the unit goal-minus-lava direction
at the input to block 21. The first column averages the 15--16 intervened
historical positions; the second measures the final answer position in the
same earlier-only conditions. For reference, the active-readout `+2 minus -2`
contrast at the final answer position was 18.50877 for semantic and 18.70691
for opaque prompts.

The earlier intervention therefore created a large signed projection contrast
where it was applied but little projection on that same released axis at the
later answer position. This is a same-axis, one-layer/one-position result, not
direct evidence of dynamic erasure. It cannot exclude a rotated, recoded,
distributed, or nonlinear carrier. `use_cache=False` means fresh full-prompt
recomputation with no persisted incremental cache; it does not eliminate the
altered keys and values produced from intervened earlier-token states within
that forward pass.

## Reversed-code audit

Llama showed substantial physical-token preference under active steering:

| Opaque surface | Readout semantic binding | Readout fixed-token bias |
|---|---:|---:|
| Persistence | 2.02734 [1.66016, 2.39063] | 1.92578 [1.72656, 2.10938] |
| Status | 1.16016 [0.98438, 1.30078] | 1.58984 [1.46094, 1.73438] |

The preregistered aggregate opaque gate mechanically passed. Both
mapping-decomposed semantic-binding contrasts had family-bootstrap intervals
above zero, but fixed-token bias was comparable for persistence and larger for
status. At the individual-surface level, the active readout contrast was 2.75
for K-coded status but -0.42969 for its M-coded reversal, and 3.95313 versus
0.10156 for the persistence pair. The 32/32 clean mapping check establishes
that the model could follow the codebook; it does not remove this
intervention-induced K/M bias.

Accordingly, the opaque aggregate is not clean mapping-invariant semantic
evidence and cannot carry the central generality claim by itself. Any paper use
should foreground the plain-language semantic replication, show this
decomposition, and label the opaque evidence as materially token-confounded.

## Execution and custody

- Model: `NousResearch/Meta-Llama-3.1-8B-Instruct` at revision
  `d10aef7999a2b5ba950ab3974312feeedbfe0b77`
- Vector artifact: `davidafrica/functional-wellbeing` at revision
  `0b005c3da6692912f6bb5a914a5a9d15c4884a91`
- Frozen design SHA256:
  `b87e10431905b49e690435d941a1b5ff3b7cfcebd59d41a86c97b8576ee1dac0`
- Preregistration SHA256:
  `9ecb6e389ca5964fee0c6306daaa17b3583bfa51652bcc5ecd3d2b3feb65d199`
- Prelaunch-audited local retry runner SHA256 (37,596 bytes, LF file):
  `ad8757f509e9722add0c58b4c6480fad5137f536b608d97aa5a8d726993e8b0a`
- Decoded connector-submitted source SHA256 (37,598 bytes):
  `f748ec1eb2fccd9f0f008d760e65eb87d3361e6f5253e6fcec6a8fb817191843`
- Terminal job: `6a81429b1f5885ae605bc9ee`, status `COMPLETED`
- Scientific forwards: 512
- Scientific runtime: 135.4244936 seconds
- Peak reserved VRAM: 16,114,515,968 bytes on NVIDIA L4

The first job `6a8140ddc97db76cbdf32af1` failed before any scientific forward
because `__file__` was undefined under inline execution. Its incident record
and the independently audited custody-only repair are retained separately.

The connector did not submit a byte-for-byte copy of the local LF file: decoding
the retry job command yields the complete 37,596 local bytes followed by one
terminal CRLF (`0d0a`). The overlapping bytes are identical, so this is a
line-ending custody distinction rather than a code or design change. The
prelaunch GO audit applies to local hash `ad8757...`; the actual submitted
payload is hash `f748ec...`. The inline job consequently logged
`script_sha256=null`; its logged `audited_parent_script_sha256=fb8ea782...`
identifies the pre-repair parent, not either retry-source hash. The preserved
terminal receipt's `retry_runner_sha256_external_seal` likewise denotes the
local audited artifact; this section and the retry manifest record the decoded
submitted payload separately.

The retry emitted a CUDA warning because `CUBLAS_WORKSPACE_CONFIG` was not set;
bitwise replay is not guaranteed even though deterministic algorithms were
requested with warning-only enforcement. Report bootstrap uncertainty over the
16 task families as designed, but do not imply that it includes hardware,
checkpoint, or between-model uncertainty.

## Preserved evidence

- Normalized complete log SHA256:
  `fa1a4371a62cc9a9d7c053a38839b5152bdc7e3cd6ca6b31f453e7efcd39cf8b`
- Parsed result SHA256:
  `767ab2768e2a2502b820cc1f211813ab373775affb3f16c966c1de026b1e66d5`
- Terminal receipt SHA256:
  `b75072a68bc8cf3efcf022bd7de3f8a794e9bc32ca077f52887f37e9e7153f9e`

Exact hardware charge is not returned by `hf jobs inspect`. The retry's hard
ceiling was `$0.60`; at the published `$0.80/hour` minute-billed rate and the
observed runtime, the actual charge should be only a few cents.
