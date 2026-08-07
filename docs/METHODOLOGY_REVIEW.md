# Methodology review and research decisions

This document is where I audit my own pipeline. It works through the proposals
a careful reviewer would raise against it, reports what the saved evidence
actually supports, and records a decision for each one. Several of those
decisions are refusals to promote something that looked good.

## Technical summary

The component-disjoint sequential path remains the reference. Learned GLS
fusion, exact projection, switching state, and the group-robust stack stay as
research controls. Their saved results are reported below.

One caveat governs everything that follows. Both 160-well panels have by now
been inspected and reused in retrospective experiments, so every result in this
document is labelled **post-hoc**. The words `holdout` and `confirmation`
describe the panels' original registry roles; they no longer certify any new
experiment as an untouched test. Reporting a number without that label would be
the easiest way to overstate this work, so the label comes first.

| Proposal | Evidence available here | Decision |
|---|---|---|
| Repeated geological-component evaluation | Four registered contracts and 2,000 whole-component bootstrap draws per contract | Keep; do not call this refit-seed stability |
| Leave-region/typewell-family-out | No frozen region contract, verified CRS, or explicit typewell-family identifier | Future opt-in evaluation |
| CVaR/group-robust selection | Tail diagnostics are useful, but frozen transfer is asymmetric | **HOLD** as a model-selection rule |
| Constrained GLS with shrinkage | Exact simplex solver exists, but the current covariance is fixed rather than learned fold-locally | Opt-in only |
| Exact $L_2/L_\infty$ projection | Correct implementation and counterexample test exist; saved cap activation is very low | Correctness option, not a score claim |
| Four-mode switching state | Target-free implementation, synthetic controls, and a 160-well component pilot | **HOLD**; worse on OOF and holdout |
| Capped 5% Ridge correction | Positive row-RMSE direction in four reused contracts; one tail/bootstrap caveat | Opt-in research candidate only |

## Component-based evaluation

I registered the Primary/Complement OOF and run-level holdout artifacts in
[`robustness_registry.json`](../configs/robustness_registry.json). The resulting
audit covers 1,556,878 scored rows, 320 wells, and 243 global components; all 85
integrity checks passed. Sequential final improved on the incumbent in all ten
saved outer folds.

The repeated calculation is a whole-component bootstrap, not repeated model
training. All saved predictions use training seed `20260806`, so the 2,000
draws per contract measure sensitivity to the represented components only. I
would need at least three independently refit seeds under one frozen split
contract before making a training-seed stability claim.

I define the reported worst-10% component CVaR as the mean component RMSE among
the worst `ceil(0.10 * G)` components. This tail view changes the interpretation
of the Complement holdout: sequential final improved pooled RMSE from 15.4761
to 14.8174, while its tail statistic worsened from 29.1741 to 29.5035. I
therefore keep pooled row RMSE as the primary metric and use macro-component
RMSE, harmed-well rate, and component CVaR as promotion guardrails. This use of
an upper-tail mean follows the CVaR construction of
[Rockafellar and Uryasev (2000)](https://uryasev.ams.stonybrook.edu/wp-content/uploads/2011/11/CVaR1_JOR.pdf).

The full definitions and saved results are in
[`ROBUSTNESS_EVALUATION.md`](ROBUSTNESS_EVALUATION.md) and
[`summary.json`](../artifacts/robustness_evaluation/summary.json).

## Leave-region and type-well-family evaluation

The current graph keeps nearby or highly similar wells in the same component,
which is appropriate for leakage control. A contiguous leave-region-out split
would instead test spatial extrapolation. I keep that as a separate future
contract rather than averaging it with component CV. Before running it, I would
freeze the coordinate reference system, block construction, boundary buffer,
and component assignment without looking at outcomes.

I also defer leave-typewell-family-out. The repository has target-free
typewell fingerprints but no authoritative family identifier; choosing a
similarity threshold after observing scores would create another selection
degree of freedom. These decisions follow the structured and spatial CV
principles in [Roberts et al. (2017)](https://doi.org/10.1111/ecog.02881) and
[Valavi et al. (2019)](https://doi.org/10.1111/2041-210X.13107).

## Constrained GLS

[`meta_state.py`](../src/rogii_portfolio/meta_state.py) now contains an exact
active-set solution of

$$
\min_{w\geq0,\;\mathbf 1^\top w=1} w^\top\Sigma w.
$$

I keep `constrained_weights=False` and `covariance_shrinkage=0` by default. The
current $\Sigma$ is constructed from nominal expert weights, fixed
correlations, and prefix variability; it is not a sample covariance. Shrinking
that matrix is a sensitivity control, not evidence that expert reliability was
estimated from data.

For a promotable version, I would estimate a component-equal error second
moment from inner-OOF residual vectors $e_i$ inside each outer-training fold,

$$
\widehat M=\frac1G\sum_g\frac1{n_g}\sum_{i\in g}e_i e_i^\top,
\qquad
M_\lambda=(1-\lambda)\widehat M+
\lambda\operatorname{diag}(\widehat M)+\epsilon I,
$$

then solve the simplex problem with $M_\lambda$. The second moment includes
expert bias; a centered covariance-only objective would need a separate bias
term. Shrinkage is useful for conditioning noisy covariance estimates, but its
intensity and target must be selected inside the nested contract, as in
[Ledoit and Wolf (2004)](https://ledoit.net/ole1a.pdf) and
[Schäfer and Strimmer (2005)](https://www.cs.princeton.edu/~bee/courses/read/schafer-SAGMB-2005.pdf).

## Projection variants

[`hgrg.py`](../src/rogii_portfolio/hgrg.py) now distinguishes the retained
direction-preserving `radial_project` from the Euclidean
`exact_l2_linf_project`. The test case $v=(10,1)$, row cap 5, makes the
difference explicit: radial scaling returns $(5,0.5)$, whereas the nearest
feasible point is $(5,1)$. Meta-State exposes this through
`exact_projection=True`, while leaving the saved default unchanged.

Across the four saved prediction tables, I found no 10-ft Meta-State row-cap
saturation and only one well at the 5-ft Meta-State RMS cap. That audit infers
activation from the saved HGRG-to-Meta-State moves because the original scalar
projection diagnostics were not persisted. Exact projection is therefore a
correctness improvement and useful unit-tested option, but I do not expect a
material score change on these saved panels.

## Switching-state control

[`switching_state.py`](../src/rogii_portfolio/switching_state.py) implements an
IMM-style approximation with `smooth`, `fault_up`, `fault_down`, and
`uncertain` modes. It consumes only target-free PF, GeoHMM, regional, HGRG, and
prefix-variability inputs. The flag `extra.enable_switching_state` is false by
default, the arm is excluded from the frozen stack, and its synthetic tests
cover alignment, finite probabilities, movement caps, a positive step, and
expert disagreement.

IMM provides a low-cost approximation for Markov-switching linear systems
([Blom and Bar-Shalom, 1988](https://doi.org/10.1109/9.1299)). Rao-Blackwellised
particle filtering is a later option if continuous jump amplitudes or more
nonlinear observations become necessary
([Doucet et al., 2000](https://research.google/pubs/rao-blackwellised-particle-filtering-for-dynamic-bayesian-networks/)).
For now I call the outputs switching-regime probabilities, not confirmed
geological faults. The actual Primary160 pilot was negative: outer OOF was
15.2866 versus 14.8993 for sequential final, and the run-level holdout was
14.6019 versus 13.9504. I therefore keep the arm default-off and assign
**HOLD**. A future version would need inner-fold calibration of transition and
jump scales, a false-switch control, and a new component panel before retesting.

## Group-robust transfer

I cross-fitted the component-risk-aware blend on one OOF panel, froze it, and
applied it to the other. The direction changed the conclusion:

| Frozen direction | Group-robust row RMSE | Sequential row RMSE | Result |
|---|---:|---:|---|
| Primary → Complement | 14.1512 | **12.6000** | Worse |
| Complement → Primary | **13.9487** | 14.8993 | Better |

The Primary-to-Complement bootstrap versus sequential crossed zero and had
37.0% positive probability; the reverse direction's interval was positive.
The grid also selected zero macro and zero CVaR penalty in both directions,
despite tail-aware selection scoring. I interpret this as heterogeneity and
selector instability, not as evidence for a transferable robust objective.
Group-DRO work likewise shows that worst-group optimization can fail without
appropriate regularization and validation
([Sagawa et al., 2020](https://openreview.net/forum?id=ryxGuJrFvS)).

The saved artifact is explicitly marked `retrospective_reused_panels` in
[`group_robust_transfer/summary.json`](../artifacts/group_robust_transfer/summary.json).
I therefore assign **HOLD** and make no fresh confirmation claim in either
direction.

## Five-percent Ridge trust region

The unstable group optimizer suggested a simpler question: can Ridge contribute
without being allowed to replace the stable sequential path? I fitted

$$
\widehat y_w=(1-w)\widehat y_{seq}+w\widehat y_{ridge},
\qquad 0\leq w\leq w_{max},
$$

with row, macro-component, worst-component, and $L_2$ terms. Repeated
component folds on Complement OOF selected a 5% cap, with fold weights between
3.5% and 5%. Freezing 5% gives the following post-hoc sensitivity result:

| Reused contract | Sequential RMSE | 5% trust-region RMSE | Gain | CVaR gain |
|---|---:|---:|---:|---:|
| Primary OOF | 14.8993 | 14.6403 | +0.2589 | +0.5957 |
| Primary holdout | 13.9504 | 13.5701 | +0.3803 | +1.0081 |
| Complement OOF | 12.6000 | 12.4656 | +0.1344 | -0.0731 |
| Complement holdout | 14.8174 | 14.5428 | +0.2746 | +0.4534 |

Whole-component bootstrap intervals for row-RMSE gain were positive in three
contracts. Complement OOF was the exception: $[-0.1270,0.3150]$ ft, despite
an 87.3% positive probability, and its tail statistic regressed by 0.0731 ft.
This is materially more stable than the free group optimizer, but the 5% value
was found after both panels had already been inspected. I expose it only as an
opt-in diagnostic arm and do not change the default or submission path. The
full weight sweep is saved in
[`trust_region_ridge/summary.json`](../artifacts/trust_region_ridge/summary.json).

## Next experiment

I would next:

1. register genuinely new predictions from at least three refit seeds;
2. freeze a separate region or externally defined family contract before
   inspecting its outcomes;
3. evaluate empirical-second-moment GLS and recalibrated switching state only
   as opt-in candidates;
4. require positive pooled transfer without a material CVaR regression before
   changing the sequential reference.

Until these checks are complete, the new modules remain opt-in research
controls and the default path stays unchanged.
