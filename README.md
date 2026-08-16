# Functional-Welfare Steering Acts at Readout

This repository contains the paper, analysis code, compact retained results,
and figure source for:

> **Functional-Welfare Steering Acts at Readout: Cross-Family Tests Separate
> Output Actuation from Persistent State**  
> Neal Krishna, University of Connecticut School of Medicine, with Apart Research

## Main result

Across separately released Qwen and Llama functional-welfare directions,
steering at the answer boundary produced large semantic effects, while the
same-axis effect was small after earlier steering was withdrawn. A separate
Qwen assay showed genuine unrestricted answer switching in both NF4 and BF16,
but path and orthogonal controls showed that this actuation was not unique to
the released welfare direction.

The paper's central distinction is between three claims:

1. a direction can causally control an output while applied;
2. the induced signal persists after the intervention ends; and
3. the direction specifically identifies the construct used to name it.

The experiments strongly support the first claim and provide little support
for the latter two in the tested assays.

## Contents

- `manuscript/main.pdf`: final seven-page paper.
- `manuscript/main.tex`: editable LaTeX source.
- `manuscript/figures/`: publication figures and exact TSV inputs.
- `experiments/`: principal local and hosted runners plus the figure generator.
- `experiments/runs/`: retained row-level records and summaries for the primary
  temporal assay, matched adapter comparison, NF4 answer-switching assay, and
  BF16 answer-switching replication.
- `research/`: concise result notes for the reported experiments.

## Reproduce the figures

From the repository root:

```bash
python experiments/make_temporal_figures.py
```

Model weights and direction tensors are not included. The paper identifies the
public model and direction repositories and their pinned revisions.

## Scope

This is a compact disclosure-reviewed research release. It excludes model
weights, direction or activation tensors, credentials, infrastructure,
browser state, and private operational records.
