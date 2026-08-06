# PF initial-state estimator audit

## Decision

**Reject the initial-state replacement. Do not combine it with explicit Euler,
package it, upload it, or submit it.**

The preregistered Dev40 grid found an apparently strong arm: estimating the
initial structural rate from the endpoints of the last 120 visible rows reduced
PF8 RMSE from **7.507195 to 6.946554** (gain **+0.560640 ft**). It improved all
five folds and its well-bootstrap 95% gain interval was
**[+0.044633, +1.205470]**. The arm therefore passed the discovery gate and was
frozen before transfer.

On Exact80 it failed decisively: RMSE changed from **8.873345 to 10.665499**
(gain **-1.792153 ft**), only three of five folds improved, and the well-level
bootstrap interval was **[-4.861749, +0.185981]**. No submission file was
created.

## Hypothesis and fixed experiment

The PF propagates the structural coordinate

```text
U = TVT + Z
```

with a persistent latent rate. The incumbent initializes that rate as the
median of the last 30 prefix first differences. Since persistence is 0.998, an
initial error decays slowly and can influence hundreds of suffix rows. The
fixed one-factor grid varied only this scalar initial state:

- visible-prefix windows: 15, 30, 60, and 120 rows;
- estimators: median first difference, endpoint slope, and OLS slope;
- multiplicative shrink: 0.0, 0.5, and 1.0.

This gives 36 registered arms. All twelve shrink-zero arms are mathematically
identical, so they were computed once; 25 distinct PF trajectories were run.
Every trajectory retained seeds 0--7, 500 particles, likelihood temperature 8,
transition noise, resampling, rejuvenation, posterior estimator, and random-draw
order. The incumbent replay was **float32 bit-identical** to the archived PF,
with maximum absolute difference exactly zero.

The selection rule was specified before scoring: pooled gain at least 0.05 ft,
positive gain in 5/5 folds, and a strictly positive lower endpoint of a
whole-well bootstrap 95% interval. Dev40 was the only selection panel. Exact80
was opened only for the single serialized winner.

## Dev40 discovery

| Arm | RMSE | Gain | Positive folds | Bootstrap 95% gain CI |
|---|---:|---:|---:|---:|
| Endpoint, window 120, shrink 1.0 | **6.946554** | **+0.560640** | **5/5** | **[+0.044633, +1.205470]** |
| OLS, window 60, shrink 1.0 | 7.151272 | +0.355923 | 4/5 | [-0.106756, +0.816176] |
| OLS, window 15, shrink 0.5 | 7.327621 | +0.179574 | 4/5 | [-0.284553, +0.791456] |
| Median difference, window 120, shrink 1.0 | 7.450889 | +0.056306 | 4/5 | [-0.012727, +0.142653] |
| Incumbent median difference, window 30 | 7.507195 | 0 | -- | [0, 0] |
| Zero initial rate | 8.831852 | -1.324657 | 1/5 | [-2.542150, -0.197032] |

The selected arm's fold gains were:

```text
+0.292822, +0.500207, +0.865649, +0.885233, +0.351754 ft
```

Only one of 36 registered arms passed all gates. At the individual-well level,
however, it improved just **24/40 wells**; the positive pooled and fold results
were driven by several large wins. This was an early warning that the aggregate
gate did not fully control catastrophic-well risk.

## Frozen Exact80 transfer

| Model | RMSE | Gain vs PF8 | Positive folds | Bootstrap 95% gain CI |
|---|---:|---:|---:|---:|
| Incumbent PF8 | **8.873345** | 0 | -- | -- |
| Frozen endpoint-120 PF8 | **10.665499** | **-1.792153** | **3/5** | **[-4.861749, +0.185981]** |

Frozen fold gains were:

```text
-8.892049, -1.369231, +0.465338, +0.258060, +0.021299 ft
```

The candidate improved 43/80 wells and had a median per-well gain of only
+0.019 ft. Two new wells were catastrophic:

| Well | Reference RMSE | Candidate RMSE | Per-well gain |
|---|---:|---:|---:|
| `70e1788b` | 8.214873 | 36.807555 | -28.592682 |
| `00bbac68` | 5.382842 | 20.136066 | -14.753225 |
| `97cd5bf9` | 6.824649 | 12.768150 | -5.943501 |

The endpoint rate changes for the first two wells were small (about -0.0040 and
-0.0013 relative to the incumbent median). Their huge downstream errors are
therefore not explained by a simple open-loop slope bias. The more plausible
mechanism is nonlinear PF mode selection: a small initial displacement changes
GR phase alignment, likelihood weights, and resampling, after which the filter
locks onto a different stratigraphic mode. This is precisely the tail behavior
that pooled RMSE and five broad fold signs can miss on a 40-well discovery set.

## Red-team conclusion

This experiment is useful despite rejection:

1. It confirms that initial-state estimation materially affects the current PF.
2. It falsifies the stronger claim that a longer endpoint estimate is a robust
   replacement for the median-30 state.
3. It shows why selecting among many nonlinear PF arms needs an untouched
   transfer panel and an explicit catastrophic-well guard, not just pooled RMSE.
4. Because the frozen arm reverses strongly, combining it with explicit Euler
   would be an unfrozen rescue experiment and is not justified near the
   deadline. The separately audited Euler arm already failed its own stability
   criterion.

A future research version should represent the initial rate as a distribution
or a mixture of short- and long-window states inside the particle population,
rather than hard-switching the whole filter to one estimated scalar. Such a
model would need a new preregistered panel and tail-risk objective.

## Leakage and reproducibility contract

- Prediction receives horizontal `MD, Z, GR, TVT_input` and typewell `TVT, GR`.
- Suffix `TVT` is read only after prediction for truth/order canaries and RMSE.
- No neighboring wells, coordinates, formation/contact targets, spatial cache,
  or suffix-derived router enter inference.
- Bootstrap resampling is by whole well.
- No Kaggle kernel, `submission.csv`, dataset version, or competition
  submission was created.

Artifacts:

- `rogii_pf_initial_state_audit.py`
- `rogii_pf_initial_state_transfer.py`
- `tests/test_pf_initial_state_audit.py`
- `pf_initial_state_dev40_evidence/`
- `pf_initial_state_exact80_evidence/`

Dev40 grid runtime was **1,130.16 s** on four shared CPU workers. The single-arm
Exact80 transfer took **110.58 s**. The focused test suite passes 5/5 tests.
