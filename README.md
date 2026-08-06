# physics-guided-wellbore-alignment
### Aligning well logs and correcting geological depth using physics-aware modeling and robust validation

Physics-guided reconstruction of hidden geological depth for a horizontal
wellbore.

I built this project around a simple but stubborn problem: given a horizontal
well's drilled path, its noisy gamma-ray log, a short visible stretch of its
true vertical thickness (TVT), and a paired vertical reference well, can you
recover the TVT for the part of the well you haven't drilled yet? Instead of
treating every row as its own regression target, I treated the whole thing as
a leakage-sensitive, sequence-level inverse problem - you're not fitting
points, you're recovering a path.

The core idea stayed the same throughout: let machine learning provide a
strong local reference, and let particle filtering, sequence alignment,
regional geological structure, and bounded residual corrections add the
physically meaningful parts on top. Nothing gets to make an unlimited
correction. Every proposed change has to survive a frozen, component-safe
transfer test before I trust it.

## The problem I was actually solving

For each horizontal well, I had:

- `MD`: measured depth along the wellbore;
- `X`, `Y`, `Z`: the known 3-D trajectory of the well;
- `GR`: a gamma-ray log that's noisy and sometimes missing;
- `TVT_input`: TVT that's only observed up to the Prediction Start point;
- a paired **typewell**: a vertical reference well with its own `GR`–`TVT`
  curve and geological context.

The target is the missing tail of `TVT` - a geological/stratigraphic
coordinate, not just the borehole's geometric vertical position. Scoring is
row-level RMSE in feet, lower is better.

The GR signal you're given is an indirect, ambiguous read on where the well
sits inside the geological column that the typewell's `GR(TVT)` curve
represents. The same GR pattern can show up at more than one depth,
measurements drop out, and faults or changing dip can quietly break any simple
one-to-one correlation between the two curves. So instead of treating this as
a pile of unrelated rows, I modeled a single latent geological path. A useful
coordinate for that is

$$
U(s)=TVT(s)+Z(s), \qquad TVT(s)=U(s)-Z(s),
$$

where $s$ is measured depth. This splits borehole motion (known, via `Z`)
from the motion of the geological surface itself, which is what I actually
need to infer. The visible prefix anchors datum and short-range trend; the
horizontal and typewell GR curves give uncertain alignment evidence for
everything past that.

## What I was trying to do

1. **Recover a coherent path**, not a scatter of isolated predictions.
2. **Combine complementary evidence** , ML features, particle trajectories, GR
   alignment, regional structure , without letting any single source make an
   unbounded correction.
3. **Kill geological leakage** by keeping nearby, duplicated, or highly
   similar wells inside the same validation component, so validation actually
   looks like deploying on unseen geology.
4. **Track uncertainty and harm**, not just pooled RMSE , fold consistency,
   component-level confidence intervals, harmed-well rate, tail error, and
   worst-well regret.
5. **Keep negative results around** so I don't keep re-discovering and
   re-overfitting the same tempting-but-wrong mechanism.

This is a reproducible research framework for physics-guided log alignment and
depth correction (not a claim that the pipeline is production-ready
geosteering software.)

## Pipeline

```text
component-safe data
        ↓
observable ML reference
        ↓
particle-filter trajectory + GeoHMM alignment path
        ↓
HGRG bounded disagreement-aware projection
        ↓
regional Meta-State direction and scaling
        ↓
visible-prefix boundary condition
        ↓
conditional GeoHMM shape correction
```

I kept the nested residual experts and the CHRRC roughness control around as
audited alternatives - they were part of the research, but they weren't
stages in the final submitted Boundary + Meta-State + shape candidate.

### 1. Observable ML reference

The baseline uses prefix-derived trajectory geometry, GR summaries, typewell
descriptors, and a nonlinear regressor. For honest scoring, every prediction
for a held-out component had to come from a model and preprocessing state
fitted without that component. I treated this as an anchor, not as proof that
local nonlinear features alone can recover long-range geology.

### 2. A hidden-generic nonlinear control

I also built a deterministic 5-fold LightGBM control that predicts the TVT
increment from the last visible anchor using 121 target-free features , no
well identity, no suffix targets, no formation surfaces, no global spatial
interpolation. On the full 773-well component-safe OOF, it went
**13.6550 → 13.6065**, improved every fold, with a whole-component gain
interval of **[+0.0421, +0.0559] ft**. Its official visible/hidden scores were
**11.223/11.670**. It's a solid robustness check, not the competitive model.

### 3. Particle filter

Propagates uncertainty over structural datum and rate from the visible heel.
It encodes trajectory continuity and gives you multiple plausible paths when
the log alone doesn't pin down a unique alignment.

### 4. GeoHMM alignment

Aligns the horizontal suffix GR sequence against the vertical typewell GR
sequence and smooths the resulting path. I'm not claiming GeoHMM is a great
standalone predictor , its value is that its errors don't look like the
particle filter's or the learned reference's.

For a fixed coefficient $\alpha$, the combined proposal is

$$
Q_\alpha=(1-\alpha)Q_{PF}+\alpha Q_{HMM}.
$$

### 5. HGRG and bounded movement

Uses disagreement between the reference, PF, and GeoHMM as an uncertainty
signal, shrinks changes it doesn't trust, and projects the whole well
correction onto a per-well RMS budget. This favors a coherent trajectory shift
over independently clipping individual points.

### 6. Regional Meta-State

A training-side regional surface gives estimates of datum, slope, and
curvature. I use it as a direction, not an unrestricted replacement ,
prediction-time disagreement decides how far the model is allowed to move
toward it.

### 7. Prefix-Boundary hand-off

The first hidden point should stay consistent with the structural datum and
tangent right before Prediction Start. A robust prefix-only tangent applies
strongly near the boundary and decays with distance. This frozen policy moved
Exact80 from **7.4324 to 7.4141** and the strict component-absent confirmation
from **11.4162 to 11.4014**.

### 8. Conditional GeoHMM shape

A cheaper stride-6 GeoHMM works as a separate shape expert. Its disagreement
with PF is centered within each well so this branch can't shift a well's
overall datum. Target-free amplitude and roughness gates scale a correction
capped at **0.75 ft well RMS** and **2.5 ft per row**. Shape alone improved
Exact80 by **0.0160 ft**; composed with the separately frozen boundary
correction, it reached **7.3986 RMSE**.

### 9. Nested residual stack and roughness control

Direct and structured residual experts get combined with non-negative convex
weights, learned strictly inside each outer fold. A small final correction
smooths out unstable high-frequency wiggle while leaving the long structural
trend alone. Any gate that failed frozen transfer got disabled rather than
retuned on the same labels.

## Leakage-conscious validation

A random row split doesn't work here , thousands of adjacent rows from the
same well share basically the same geological error. A well-only split is
better, but it still leaks through nearby wells, shared typewells, and
near-duplicate GR curves.

I built connected geological components using spatial proximity, Prediction
Start proximity, exact typewell identity, and high GR-curve similarity, then
assigned whole components to folds and used those same components as
bootstrap units.

My promotion rule, every time:

1. define a parent prediction and one hypothesis;
2. tune only on the declared development population;
3. freeze the policy and predictions before opening transfer labels;
4. evaluate once on a component-disjoint or otherwise frozen panel;
5. check pooled gain, fold signs, confidence interval, and well-level harm;
6. promote it, shrink it, or zero it out.

Throughout,

$$
\text{gain}=RMSE_{base}-RMSE_{candidate},
$$

so positive is better, and I only compare absolute RMSE values when they share
the same parent prediction, population, folds, and scoring contract.

## What actually worked

These numbers come from different experiment populations , **don't add them
together or read them as one leaderboard estimate.**

| Experiment | Validation contract | Base → candidate RMSE | Gain |
|---|---|---:|---:|
| PF + 20% GeoHMM | 40-well discovery panel | 7.5072 → 6.5775 | +0.9297 ft |
| HGRG bounded projection | 80-well transfer surrogate | 7.8299 → 7.5006 | +0.3293 ft |
| Regional Meta-State | frozen 160-well confirmation | 9.8871 → 9.1804 | +0.7067 ft |
| Nested residual stack | same 160-well confirmation | 9.1804 → 9.1087 | +0.0717 ft |
| Prefix-Boundary | Exact80 trajectory | 7.4324 → 7.4141 | +0.0183 ft |
| Boundary + conditional shape | Exact80 composition | 7.4324 → 7.3986 | +0.0338 ft |

The residual stack improved all five confirmation folds, but only added a
small amount beyond its best single expert, and the fold-wise weights were
unstable , so I'm keeping it as a bounded overlay, not calling it a
breakthrough.

A few other things that kept showing up:

- long-range trend matters more than how good a correction looks right next
  to the anchor;
- a small tail of hard wells drives most of the pooled squared error;
- PF/HMM disagreement is useful for flagging risky wells, but not reliable for
  telling you which direction to correct in;
- moving the whole trajectory within a budget transfers more safely than
  aggressive pointwise edits;
- an honest, geography-grouped score can look worse than a leaked one while
  being far more useful.

## Things that didn't work

Failed hypotheses are part of the result, not something to bury:

- a longer heel-slope initialization improved development and then failed on
  transfer;
- a directional structural field improved discovery folds and reversed on a
  frozen panel;
- a learned regret router showed there was oracle headroom, but couldn't
  actually predict which way to correct;
- a PF policy I thought was horizon-specific collapsed to just one global
  coefficient;
- 3-D chord increments helped in discovery and lost on transfer;
- transporting the actual suffix target from the same well was a useful
  diagnostic, but I excluded it from the hidden-well methodology since it's
  not a valid deployment assumption.

These controls back up the main lesson here: a physically sensible formula can
still overfit, and a failed frozen transfer should usually mean you abstain or
zero the weight , not go back and search over the same labels again.

## Final result

The candidate I ended up deploying combined **HGRG, Meta-State,
Prefix-Boundary, and conditional stride-6 GeoHMM shape**. It scored:

| Partition | RMSE |
|---|---:|
| Visible scoring partition | **6.536** |
| Hidden final partition | **9.091** |

<img width="2012" height="154" alt="image" src="https://github.com/user-attachments/assets/43d1b40c-57c8-4f31-b0f1-94328eaecb75" />


That hidden score landed comfortably inside the bronze medal range. The catch: my
submission finished scoring after the official competition deadline, which
made it ineligible for final judging. I don't think one hidden-partition
result proves every individual overlay works, but it's a real, meaningful
end-to-end result for the combined method , I just didn't get it in under the
wire.

That part stings a bit, and it was a time-management problem, not just bad
luck. My three-well canary run finished its model audit and full log
comfortably fast, but all three of those wells were protected overlaps , the
run verified ID order, finite predictions, checksums, and code connectivity,
but it never told me anything about hidden-scale inference time or queue
latency on genuinely new wells. In a code competition, runtime, checkpoints,
caching, restart behavior, launch margin, and queue time aren't details you
sort out after the modeling is done , they're part of the methodology. Next
time I'd profile on representative non-overlap wells, freeze earlier, and
leave real margin before the deadline.

## Limitations

- The strongest 40-, 80-, and 160-well results come from different parents and
  play different roles in the research loop , their gains don't add up.
- Some confirmation wells were used in earlier studies of mine, so those
  results are replicated evidence, not a pristine external test.
- I never pinned down the coordinate reference system for `X`/`Y`, so spatial
  thresholds are reported in raw coordinate units.
- Prediction-time disagreement scores are risk signals, not calibrated
  probabilities.
- Large saved predictions and the competition data itself may be left out of
  public bundles for size, licensing, or access reasons.

## What I'd do next

1. Rebuild every imputer, reference model, and stacker inside each geological
   component fold.
2. Push the categorical alignment state to a finer resolution around the exact
   OOF parent.
3. Learn an uncertainty model with component-balanced regret and explicit
   worst-well constraints.
4. Evaluate the fully frozen path once, on a panel that's genuinely never been
   touched for selection.
5. Profile and cache the exact inference graph before running any large model
   search again.

## About this project

This was built for the
[ROGII – Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
competition on Kaggle. The
[community discussion thread](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367)
around it was genuinely useful for framing the problem as an inversion/alignment
task rather than plain regression, and I'd credit it for a lot of the early
direction here. Everything else , the experimental design, validation choices,
implementation, and any mistakes , is mine.

Competition data and any attached community materials are subject to Kaggle's
rules and their original authors. This is an independent project, not an
official ROGII product.
