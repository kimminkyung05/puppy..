# Holdout evaluation lock

The holdout split has not been scored.

## Frozen evaluation policy

- Purebred result: Tsinghua validation Top-1 accuracy.
- Mixed-dog result: UF Top-1 accuracy, reported separately from purebred accuracy.
- DNA labels are partial labels containing up to four reported breeds; percentages are not forced to sum to 100.
- If multiple breeds share the largest reported percentage, predicting any mapped co-dominant breed counts as correct.
- Dogs whose dominant breed has no Tsinghua mapping are excluded from the accuracy denominator and included in the coverage report.
- The three seed probabilities are averaged for the seed-ensemble result; individual-seed accuracy and range are also retained.
- No temperature calibration is required for Top-1 because a positive temperature does not change the argmax.

## Frozen model conditions

- CLIP: `cls`, 512 dimensions, row-wise L2 normalization.
- DINOv2: `cls`, 768 dimensions, row-wise L2 normalization.
- DINOv3: `cls`, 768 dimensions, row-wise L2 normalization.
- SIFT: BoVW-512, row-wise L2 normalization.
- Classifier: single linear layer with softmax, balanced training weights.
- Seeds: 1, 2, and 3.

## One-time holdout rule

The 80-dog holdout split may be scored only after the team explicitly confirms that the policy and model conditions above are final. No model, mapping, threshold, or exclusion rule may be changed in response to the holdout result.
