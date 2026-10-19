# Method

All tests use the **block** as the experimental unit. Trial-level rows are averaged
within their block over measured trials only; missing trials are counted, never zeroed.

## Real vs each control

For the real condition *R* and control *C*, with block means `r_1..r_n` and `c_1..c_m`:

- **Statistic:** `T = mean(r) - mean(c)`.
- **Null:** condition labels are exchangeable between blocks. Without strata, all
  `C(n+m, n)` relabellings are enumerated when there are at most 20,000; otherwise
  10,000 are sampled and `p = (b + 1) / (B + 1)`, where `b` counts sampled statistics at
  least as extreme as observed. With strata (suite, culture, day), labels are only
  exchanged within a stratum, which keeps stratum-level differences out of the null
  (always sampled).
- **Multiplicity:** Holm-Bonferroni across controls. A control is *beaten* when its
  adjusted p is below alpha *and* the difference points in the claimed direction.
- **Resolution:** the smallest achievable p is `1 / #relabellings` (one-sided). If a
  control is not beaten and the design cannot reach `alpha / #controls`, the verdict is
  *inconclusive*.
- **Interval:** percentile bootstrap of `T`, resampling whole blocks within each
  condition (5,000 resamples).
- **Effect size:** Hedges' g with the `1 - 3 / (4(n+m) - 9)` correction.

## Data integrity

1. Per condition, count missing and measured units (trials, or blocks for block-level
   tables). Fisher's exact test, real vs each control, Holm-adjusted.
2. **Worst-case sensitivity:** replace every missing value with the least favourable
   measured value (the minimum for the real condition and the maximum for controls when
   the claim is "greater"), recompute block means and repeat the controls comparison.
3. Status: **fail** if missingness differs *and* a beaten control no longer is under the
   worst case; **warn** if only one of those holds or data is missing; **pass** if
   nothing is missing.

## Order effects

Values are centred within condition. The Spearman correlation between the centred
values and schedule position is tested by permutation (two-sided). Schedule balance is
the largest absolute offset of a condition's mean position from the overall mean, as a
fraction of the run length. **fail** when drift is significant and the offset exceeds
10%; **warn** when only one holds.

## Improvement over time

- Trend in the real condition: Spearman permutation test, in the claimed direction.
- Specificity: the difference in OLS slopes, real minus pooled controls, after centring
  each sample, tested by permuting condition labels between blocks.
- **pass** when real improves and its slope exceeds the controls'; **fail** when real
  improves but the slope difference is not significant; **info** when real shows no
  detectable improvement.

## Interim looks

Simulate 200,000 standardised random walks under the null with `K` equally spaced
looks. The naive false positive rate is `P(max_k Z_k > z_alpha)`. The constant
per-look threshold is the `1 - alpha` quantile of `max_k Z_k`, converted back to a
nominal p. With `K = 5`, two-sided, alpha 0.05, this gives about 0.0155, close to
Pocock's published 0.0158.

## Pre-registration

The plan is serialised as canonical JSON (sorted keys, no whitespace) and hashed with
SHA-256. On load, the hash is recomputed; a mismatch makes the verdict *unreliable*.
