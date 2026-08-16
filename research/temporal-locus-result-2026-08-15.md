# Temporal Welfare Locus: Confirmatory Result

Status: complete local confirmatory assay; incorporated into the final verified
manuscript candidate.

Evidence directory: `experiments/runs/temporal-welfare-locus-full-20260815T193114Z`

## Question

When a published functional-welfare direction exerts causal leverage over a model's next-token decision margin, does that leverage survive withdrawal as a retained state, or is it concentrated while the readout is being computed?

## Frozen design

- Exact local model: Qwen3-4B-Instruct-2507, pinned revision, NF4 with BF16 compute.
- Layer-29 goal-minus-lava direction from the public independent-replication artifact `davidafrica/functional-wellbeing`, revision `0b005c3d...`.
- Sixteen independent task families.
- Short and long earlier assistant spans.
- Intervention locus crossed between earlier-only and readout-time exposure.
- Semantic choices, reversed opaque one-token codes, and a numeric 0--9 response surface.
- Dose series and path, sentiment-residualized, and orthogonal-vector controls.
- 5,376 fixed-transcript next-token records; 21 minutes 23 seconds; peak allocated VRAM about 3.05 GB.

## Main result

At factor 0.5 on long histories:

| Response surface | Earlier-only shift | Readout-time shift | Earlier/readout ratio |
| --- | ---: | ---: | ---: |
| Semantic success/failure choice | 0.047 [0.004, 0.090] | 4.000 [3.875, 4.145] | 0.012 |
| Reversed opaque code | 0.129 [0.100, 0.158] | 2.311 [2.217, 2.402] | 0.056 |
| Numeric 0--9 rating | -0.009 [-0.039, 0.021] | -0.113 [-0.205, -0.038] | not interpretable as intended |

Intervals are 10,000-draw task-family bootstrap intervals. The dose-site projection was about +22 for the earlier intervention, but the final unsteered projection was slightly negative rather than retained. Readout effects were approximately dose-ordered: semantic shifts 1.969, 4.000, and 7.797 at factors 0.25, 0.5, and 1.0.

The reversed-code decomposition shows that the logit effect is substantially mapping-aware rather than a fixed-token preference. On the long status surface, semantic binding was 2.883 logits and fixed-token bias 0.086; on persistence, binding was 1.738 and fixed-token bias 0.395.

The clean binary margins were saturated: pooled semantic clean margin was 23.08 logits (16.99 status and 29.17 persistence). Across the 64 semantic and 128 opaque task-by-interval cells, the factor-0.5 readout contrast changed the sign of the pairwise margin in 0 and 1 cells respectively. Therefore this assay establishes causal leverage over selected next-token log-odds, not robust answer switching.

## Interpretation

The result is positive evidence that this direction has strong, mapping-aware causal leverage over selected next-token log-odds at the decision readout. It is negative evidence for a simple tonic-state account in which the same direction remains present until a later unsteered report. The safe wording is:

> In this checkpoint and assay, a released welfare direction exerts much greater mapping-aware leverage over next-token log-odds while active at readout than after withdrawal, with little retained component along the same axis at an unsteered readout.

Do not say that the model has no persistent internal state. A transformed or redistributed trace could exist outside the measured direction, and the earlier opaque effect is small but nonzero.

## Limitations that must remain visible

- Numeric ratings failed to move in the intended direction, so the effect does not generalize across all report surfaces.
- Clean binary margins were strongly saturated and almost never crossed zero, so the present result should not be described as changing answers.
- Orthogonal directions sometimes moved logits, so direct output susceptibility is not welfare-specific by itself.
- These are independent-replication tensors, not original-author Han et al. tensors. They were extracted from an associated LoRA maze-trained Qwen checkpoint using 300 trajectories per tile and transplanted into a separately downloaded maze-naive Qwen checkpoint. The source base revision is not hash-bound in the vector metadata.
- Trained-to-naive transplantation follows Han et al.'s evaluation paradigm, but the combined goal-minus-lava direction, layer 29, NF4 target, single-token fixed-prompt intervention, and endpoints are derived rather than an exact paper reproduction.
- The primary magnitudes came from NF4. A frozen exact-checkpoint BF16 run later reproduced the qualitative temporal contrast on six long-history binary surfaces; it did not replicate the short interval, numeric surface, controls, dose series, or another model. See `research/temporal-locus-bf16-result.md`.
- The study concerns functional computation, not experience, suffering, consciousness, or sentience.

## Constructive contribution

For a steering result to support state language, require both:

1. a retained effect after the intervention is withdrawn before the outcome is computed; and
2. a corresponding signal at an unsteered readout, with semantically equivalent output encodings and matched vector controls.

All later challengers are terminal. The susceptibility assay stopped at its
response-code precondition; endogenous goal-versus-reference Stage 1 failed its
frozen specificity gates; and the redesigned value-versus-prediction-error assay
stopped after exactly 115 capability forwards with zero target projections. None
supports an additional construct claim or authorizes a rescue run. The temporal
result, BF16 precision check, and associated-adapter comparison form the final
scientific spine.
