# Source-adapter temporal extension: result

Status: complete, preregistered, integrity-valid. This is a robustness extension to the main temporal-locus assay, not a replacement headline or a magnitude replication.

## Question

Could the main assay's low earlier-to-readout transport be an artifact of applying a released goal-minus-lava direction to maze-naive weights rather than to the model associated with the direction's extraction?

The extension compared the associated step-400 LoRA reconstruction (`source`) with the same NF4-wrapped Qwen3-4B model under adapter disablement (`naive`). It used matched one-token pulses at the earlier and readout sites, 16 fresh task families, semantic labels and both orientations of an opaque K/M codebook, and a paired task-family bootstrap.

## Frozen execution

- Run: `experiments/runs/source-adapter-temporal-20260815T210504Z`
- Planned and observed forwards: 1,491 = 3 checksum + 432 calibration + 960 primary + 96 lower-dose linearity
- Target records: 1,056, all unique and completely paired
- Wall time: 279.0 seconds; peak allocated VRAM: 3,012,160,512 bytes
- Adapter toggle checksum: naive repeat maximum error 0; source-minus-naive maximum logit difference 2.015625
- Runner SHA256: `9d3151df61136b7a35c6b506b0b925221aa625c808ed0577b00774904e7930b7`
- Summary SHA256: `01ba25165072c75729b564f37add6953c479aab2b0732f6b7d58f5dff1d11ada`
- Records SHA256: `80b9e084ce47078124f1ccfe8259a6a0d6e63fccb053332241529c8925bd0990`

## Result

| Arm | Earlier slope | Readout slope | Q | Earlier/readout ratio rho |
|---|---:|---:|---:|---:|
| Associated LoRA | 0.024740 | 2.303385 | -0.551107 | 0.010741 |
| Adapter disabled | 0.024740 | 2.518229 | -0.604818 | 0.009824 |

- Difference in rho, source minus naive: 0.000916, 95% CI [-0.012857, 0.016224]. The complete interval lies within the frozen +/-0.10 equivalence bound.
- Source rho 95% CI: [-0.002180, 0.024363]; naive rho 95% CI: [-0.001581, 0.021138]. Both upper bounds are below 0.025 and far below the frozen 0.25 ceiling.
- Difference in Q: 0.053711, 95% CI [0.017904, 0.089844]. This is entirely due to lower contemporaneous readout susceptibility under the adapter, not increased earlier transport; the earlier slopes are identical.
- Final/immediate same-axis ratios were -0.002886 (source) and -0.000606 (naive), with interaction -0.002280. Both are substantively near zero, while the opposite interaction sign blocks a source-specific transfer claim.

The preregistered conclusion is **shared ephemerality**. This is a positive bounded-equivalence result for this assay: the released axis has very little same-axis transport in both arms. It is not evidence of exact equality, absence of all persistent hidden changes, absence of a transformed trace, welfare, sentience, or subjective state.

## Mandatory caveats

- No candidate opaque code pair passed the frozen balance gate. K/M was the prespecified saturated fallback. It passed explicit mapping comprehension at 100%, but all clean surfaces were saturated; claims remain about selected logits and hidden-state transport, not answer switching.
- Opaque mapping 1 and mapping 2 had opposite tiny earlier slopes. `Opaque abstraction passed` means mapping comprehension and the preregistered source-versus-naive interaction orientation, not stable mapping-invariant earlier susceptibility.
- The extension used one-token earlier pulses and longer pulse-to-readout distances: semantic 47-48 tokens short and 85-86 long; opaque 55-56 and 93-94. The main assay used 15-16-token earlier exposure and shorter comparable distances. This is not a distance-invariance test or magnitude replication.
- The formal pooled lower-dose linearity gate passed, but the readout site dominates it. The near-zero earlier slope changed sign across factors and should not be called demonstrably linear.
- The associated LoRA reconstruction's original base revision is not hash-bound. The comparison is exact within the local wrapped model and adapter toggle, not proof that the original extraction checkpoint was reconstructed byte-for-byte.

## Paper use

Use only as a compact robustness paragraph addressing the transplant objection. Do not change the title, make it a new figure, or elevate it above the main fixed-text withdrawal/readout-locus result.
