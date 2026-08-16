# Repaired semantic choice V4-C: bounded result

Status: **complete; eligible for a bounded native-action actuation claim;
specificity promotion failed**

Run:
`experiments/runs/repaired-semantic-choice-validation-v4c-20260816T174042Z`

## Exact result

V4-C completed one frozen target/control run on 64 **adaptively clean-
calibrated, intervention-unseen cases** with semantic `CONTINUE`/`STOP`
actions, after the V4-A and V4-B pretarget calibration stops. The record
contains 1,088 rows:

- 64 clean rows;
- 128 rows each for full-dose and half-dose functional-welfare steering; and
- 128 rows each for path, sentiment, and four orthogonal controls.

Every intervention output had a valid unrestricted semantic action token. At
full target dose, the minus-to-plus arms switched `13/64 = 0.203125` actions in
the prespecified STOP-to-CONTINUE orientation. The mean plus-minus selected-
logit contrast was `2.724609375`, and its mean was positive in all eight
order-by-count cells. At least one clean-to-single-sign action switch occurred
in each of the four order cells. Half dose produced `8/64 = 0.125` oriented
switches and mean contrast `1.37890625`.

This clears the frozen **bounded actuation** rule. It repairs the earlier
statement that the project had only selected-logit leverage: on this adaptively
clean-calibrated, intervention-unseen boundary, applying the released direction
at readout changed the model's unrestricted top-token semantic action.

## Specificity failure

The same run does not identify the effect as functional-welfare-specific.

- The path control produced `19/64 = 0.296875` oriented switches and mean
  contrast `5.18359375`, both larger than the target.
- Random orthogonal control 01 was endpoint-KL comparable to the target
  (`0.820734` ratio), produced exactly the same `13/64` oriented switch count,
  and chose the same unrestricted minus/plus answers as the target in all 64
  cases. Its mean contrast was `2.53515625`, versus target `2.724609375`.
- The strongest target-versus-control margin contrast was
  `-2.458984375`, cluster-bootstrap 95% interval
  `[-2.78515625, -2.158203125]`.
- The target oriented-switch rate minus the maximum control either-direction
  switch rate was `-0.09375`, interval `[-0.171875, -0.03125]`.
- Mandatory-control realized intervention-endpoint KL ratios did not all fall in
  the required `[0.8, 1.25]` interval.

Accordingly, the descriptive and bootstrap-supported specificity gates both
failed. The result must not be called welfare sensitivity, a welfare-specific
decision effect, or evidence of experience.

## Normative boundary

All c3/c4 cases have `STOP` as the majority-rule answer. Positive steering can
therefore turn a correct STOP into an incorrect CONTINUE, while negative
steering can repair a clean CONTINUE error. The result establishes causal
action-token actuation, not rational improvement or beneficial choice.

## Promotion decision

The following sentence is supported:

> On 64 adaptively clean-calibrated, intervention-unseen cases, unrestricted
> `CONTINUE`/`STOP` answers differed between the negative and positive
> intervention signs in 13 cases, with positive mean signed leverage across
> every wording/order cell. Endpoint-KL-comparable random orthogonal control 01
> reproduced every full-dose signed target answer, and therefore the same 13/64 switching
> pattern, with mean contrast 2.535 versus the target's 2.725.

It must be followed by the control qualification: path steering was stronger,
so the switching result establishes bounded actuation rather than construct
specificity. No post-target tuning or retry was performed on this cohort.

Independent arithmetic checks pass:

- `13 / 64 = 0.203125`;
- `19 / 64 = 0.296875`;
- `0.203125 - 0.296875 = -0.09375`; and
- `2.724609375 - 5.18359375 = -2.458984375`.

## Custody

Custody limitation: the final `custody.json` seals the V4-C and V4-B runners,
the V4-C amendment, vectors, receipts, and output artifacts, but it does **not**
seal the imported `temporal_welfare_locus.py` dependency or the model/tokenizer
files. It also omits the copied `regenerated-case-custody.json` hash; that copied
file currently matches the V4-B artifact at SHA-256
`d7b00cc0e5b033b4d4eabe64c5f6ec66895b3f6bbaedd665104312c22bf98c65`.
The final custody file is therefore not a complete transitive runtime-dependency
or copied-input seal. This does not change the recorded endpoint rows, but it
limits the strength of the reproducibility claim and must remain disclosed.

- V4-C runner SHA-256:
  `32dd79f6aee6ccbd8dcf9f815c9531972dd50588881cbf09d029dd9ca2196dd9`
- V4-C amendment SHA-256:
  `12e516d449c450b09e65d22bbcf82a43ba122fe0fbc287acb5ab69e7a30442c1`
- `summary.json` SHA-256:
  `3cb3790af443c8d40c1bdb89149335b26af13868d684709c58dbeb6661eec884`
- `validation-records.jsonl` SHA-256:
  `07ae4ffbbddbfcbf96f3c9842d4d84f79f06dec49bd172a7a945cbd8ffab880c`
- `vector-metadata.json` SHA-256:
  `6dc4cc3f7e8199c2499465949ff0946c0b571cfcd7a033a86e70a26716587b0c`
- `custody.json` SHA-256:
  `7999c97235d84b73a4c2fe69b1ee6f7e4c73559d33c78a1163a613a050bac6df`
- fixed-neutral-control receipt SHA-256:
  `08c54f95adcee17b536dbbd72d8c288748c5f4d977a018de71a049acb8e523fc`
