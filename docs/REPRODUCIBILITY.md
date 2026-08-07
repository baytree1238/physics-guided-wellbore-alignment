# Reproducibility and experiment guide

This guide shows how to run and audit the wellbore trajectory study. The
repository includes the clean-room pipeline, the audited historical
121-feature builder, and an adapter for external scored artifacts.

Read the root [`README.md`](../README.md) for the research overview and
[`CODEBASE_GUIDE.md`](CODEBASE_GUIDE.md) for the source layout.

## Experiment summary

The task is to predict the hidden TVT suffix of a directional well from its
visible prefix, geometry, gamma ray log and a reference type well. A well-level
split was too optimistic because nearby or similar wells could appear in
different folds. The final evaluation splits connected components of an XY/GR
similarity graph.

The scored research path combined a particle filter, GeoHMM, HGRG, Meta-State
and bounded boundary/shape corrections. The reimplemented branch refits every
learned model, spatial RBF and stack weight inside outer component folds. I ran
that contract on two component-disjoint halves of the full 320-well universe.

The historical pipeline scored 6.536 on the visible partition and 9.091 on the
hidden partition, but completed after the deadline. I treat that as one result
for the whole pipeline, not as an ablation of each block. Recreating the exact CSV
also requires the external model artifacts used by that run.

## Publishing on Kaggle

Import `portfolio_notebook_executed.ipynb` as the notebook. Saved outputs make
it readable on its own, but rerunning it also requires the companion files in
`src/`, `evidence/` and `artifacts/`. Build the small companion archive with:

```bash
make kaggle-bundle
```

Upload the files under `dist/` as a Kaggle Dataset and attach it to the
notebook. See [`KAGGLE_UPLOAD.md`](../KAGGLE_UPLOAD.md) for the exact layout.

## One-command reproduction

```bash
make reproduce
```

On Windows or on a minimal machine without GNU Make, the equivalent command is:

```bash
python reproduce.py
```

This command creates the local environment, runs the model stages on
deterministic synthetic wells, performs nested component CV and a holdout
evaluation, rebuilds the notebook, runs the tests, and verifies tracked outputs
by SHA-256.

For the official data (the data are not redistributed):

```bash
make reproduce-full DATA_ROOT=/path/to/rogii-wellbore-geology-prediction.zip
```

For a small official-schema integration check, run:

```bash
make realdata-smoke DATA_ROOT=/path/to/raw-data-directory-or-zip
```

It uses 18 wells and reduced PF/HMM budgets. It verifies raw-data wiring and
the complete nested pipeline, but is not a competition-score estimate. Components
are built only within that 18-well subset, so this target is an integration
test, not a substitute for the full-universe leakage graph.

For the larger CPU validation used in this repository, run:

```bash
make realdata-nested-160 DATA_ROOT=/path/to/raw-data-directory-or-zip
```

This first constructs the similarity graph over all 320 available training wells, then
samples whole components up to 160 wells and runs 5 outer × 3 inner folds.
PF/HMM compute is reduced so this is a validation of the complete pipeline and
its generalization direction, not a byte-equivalent run of the expensive
historical submission policy.

Run the complete two-panel research protocol with:

```bash
make realdata-research-320 DATA_ROOT=/path/to/raw-data-directory-or-zip
```

The full profile uses 12 PF seeds × 500 particles, stride-6 GeoHMM, five outer
folds, four inner folds and 5,000 component bootstrap draws. It is intentionally
much slower than the smoke profile.

## Current real-data result

The reduced-compute clean-room pipeline has now been run over two disjoint halves
of the full 320-well component universe:

| Component panel | Incumbent outer OOF | Sequential outer OOF | Gain |
|---|---:|---:|---:|
| Primary160 (96 development components) | 15.5551 | 14.8993 | +0.6558 ft |
| Complement160 (100 development components) | 13.5177 | 12.6000 | +0.9177 ft |

The corresponding run-level holdout gains were +0.5928 ft across 23 components
and +0.6587 ft across 24 components. These are validation RMSE values for the
retrained clean-room parent, not estimates of the historical 6.372 notebook or
the competition leaderboard.

A Ridge-heavy policy looked excellent on Primary160, then failed on the
component-disjoint Complement160 OOF (13.5177 incumbent → 14.3089 expanded;
−0.7912 ft). This negative transfer is included in the results. The
HGRG-centered sequential path remains the selected method.
Likewise, removing Meta-State gained 0.0481 ft on primary OOF but lost 0.0543
ft on complementary OOF; it remains a post-hoc HOLD rather than a promoted
weight change.

### Post-hoc robustness audit

I reused the four saved Primary160 and Complement160 prediction contracts to
check whether the headline ranking survives different aggregation choices. This
is a post-hoc diagnostic of panels I had already inspected, not a new
confirmation experiment. The registry is
[`configs/robustness_registry.json`](../configs/robustness_registry.json), and I
rerun it with:

```bash
make robustness-evaluation
```

Across 1,556,878 scored rows, the sequential path beat the incumbent in all ten
outer folds; its mean fold rank was 2.35 with a rank standard deviation of 0.32,
and its worst fold still gained 0.2666 ft. Ridge had a similar mean rank of 2.50
but a 3.07 standard deviation and a worst-fold loss of 14.3103 ft. On
Complement160 outer OOF, Ridge's worst-10% component RMSE was 37.4849 versus
29.6505 for the incumbent. This is why I report pooled, macro-well,
macro-component, harmed-well and tail metrics together.

All four saved real-data contracts use one training seed (`20260806`). The
2,000 whole-component bootstrap repeats per contract measure evaluation-sample
variation; they do not establish refit-seed stability. I would need new
registered predictions from several independent fits before making that claim.
The complete audit is in
[`ROBUSTNESS_EVALUATION.md`](ROBUSTNESS_EVALUATION.md).

### New methodology controls

I implemented the main actionable ideas from the methodology review without
changing the frozen default path. Exact simplex GLS and exact joint
$L_2/L_\infty$ projection are opt-in correctness variants. A four-mode
switching-state model was also connected as a diagnostic arm, but its actual
Primary160 pilot was negative: 15.2866 versus 14.8993 on outer OOF and 14.6019
versus 13.9504 on the run-level holdout, so it remains **HOLD**.

A simpler candidate was a 5% trust-region move from the sequential path toward
Ridge. It improved row RMSE in all four reused contracts by 0.1344 to 0.3803
ft. Whole-component bootstrap intervals were positive in three contracts;
Complement OOF crossed zero and its worst-component statistic regressed by
0.0731 ft. Because both panels had already been inspected, the weight is
exposed only through `extra.enable_trust_region_ridge` and is not a promoted
default. See
[`METHODOLOGY_REVIEW.md`](METHODOLOGY_REVIEW.md) for the equations,
negative results and decision gate.

```bash
make posthoc-methods
make realdata-switching-160 DATA_ROOT=/path/to/raw-data-directory-or-zip
```

## What is actually implemented

```text
official/synthetic raw wells
    ├── geological similarity graph ──> component-disjoint folds + holdout
    ├── visible-prefix inference view (suffix TVT removed)
    ├── Monte-Carlo PF ───────────────┐
    ├── forward/backward GeoHMM ──────┼──> HGRG
    ├── 121 target-local features ────┘       │
    │      └── Ridge + nonlinear control      v
    └── outer-training-only RBF ───────> correlated GLS + RTS Meta-State
                                               │
                                               v
                                   Prefix-Boundary → conditional shape
                                               │
                             inner-OOF convex stack (research control)
```

The implementations are not plotting stubs:

- `particle.py` propagates and resamples particles, returns every seed path and
  predictive log evidence.
- `geohmm.py` constructs the TVT × slope state grid and executes checkpointed
  forward/backward inference.
- `features.py` constructs a readable clean-room version of the 121-name
  FAST-SAFE schema for nested retraining; `historical_features.py` separately
  vendors the actual audited notebook formulas and their source hashes.
- `hgrg.py` implements the frozen regret equation and movement budgets.
- `switching_state.py` implements the default-off IMM-style changepoint
  control; `trust_region.py` implements the capped Ridge correction.
- `physics.py` and `meta_state.py` implement a clean-room structural
  observation + correlated GLS + constant-acceleration RTS state-space fusion.
- `overlays.py` implements a clean-room fixed-window prefix correction and the
  formula-faithful conditional shape correction.
- `stack.py` fits nonnegative, sum≤1 convex weights from inner-OOF predictions.
- `pipeline.py` connects all blocks, refitting learned state inside every outer
  component fold and applying the frozen pipeline once to the untouched holdout.
- `evaluation.py` validates the saved experiment registry and reports row,
  well, component, horizon, worst-tail, fold and bootstrap-seed metrics.
- `deployment.py` refits the selected clean-room pipeline, rejects target-bearing
  test objects, preserves sample-submission ID order and records CSV hashes.
- `parents.py` verifies a pinned historical expert artifact before it can enter
  the scored-lineage overlay path.

See the [implementation map](IMPLEMENTATION_MAP.md) and
[method cards](METHOD_CARDS.md) for formulas, pseudocode, complexity,
closest baselines, ablations and failure modes. The
[historical-parity ledger](HISTORICAL_PARITY_GAPS.md) lists every exact,
analogue and externally blocked block. The
[real-data nested result](REALDATA_NESTED_RESULTS.md) gives the complete
two-panel score and negative-transfer audit.

## Reproduction modes

| Mode | Purpose | Required inputs |
|---|---|---|
| `make reproduce` | Clean-room code path, target isolation, nested split, figures and notebook execution | Repository only |
| `make reproduce-full` | Clean-room algorithmic retraining and a new component-disjoint holdout result | Official competition ZIP/directory |
| exact historical bytes | Same historical prediction/CSV bytes | Official data **plus** pinned external pretrained artifacts and their hashes |

The last mode is fail-closed in `configs/exact_artifacts.json`.
The original scored notebook loaded third-party feature/model packages; until
their exact versions and hashes are supplied, claiming byte identity would be
false.

## Validation contract

The final experiment performs the following sequence:

1. Build graph-connected geological components before model fitting.
2. Reserve complete components as a run-level holdout.
3. Within each outer development fold, fit the 121-feature estimators and RBF
   using only outer-training components.
4. Produce inner-OOF expert paths on the outer-training components and fit the
   residual stack there.
5. Apply that fold-frozen pipeline to outer validation components.
6. Fit one final stack from development outer-OOF predictions, refit the base
   models on all development components, and open the holdout once.
7. Bootstrap complete components, never individual rows.

Target-free PF/HMM/features may be cached because each depends on only one
well's visible prefix, suffix covariates and type well. Model-facing objects
have no truth field; scorer labels live in a separate wrapper. Official-data
mode also refuses to synthesize a broken prefix from truth.

The source data do not establish a coordinate reference system. Spatial graph
thresholds are therefore distances in raw coordinate units; the `_ft` suffix
on legacy config fields is retained for compatibility and is not evidence of a
verified unit conversion.

## Scored lineage versus research controls

The historical 9.091 pipeline was:

```text
incumbent → HGRG → Meta-State → Prefix-Boundary → conditional GeoHMM shape
```

This repository does **not** relabel its retrained parent as that historical
incumbent. The scored parent also contained SP45/beam selection, robust
polynomial projection, visible-prefix calibration, contact logic, DTRT and
external learned packages. HGRG and conditional shape are formula-faithful;
the new RBF/Meta-State and fixed-window boundary modules are clean-room
analogues. Exact historical expert paths must enter through the hash-verified
`HistoricalArtifactParent` adapter.

The 121-feature nonlinear retrain, nested residual stack and optimizer sweep
were separate controls. They are implemented and evaluated here, but their
gains are not attributed to the historical score. Likewise, results from the
40-, 80- and 160-well panels are not added across incompatible parents.

The latest optimizer control is a useful negative result: fold-specific choices
included higher L2, adaptive learning rate, modified Huber loss, label
smoothing, 300 epochs and a softened decoder. The nested candidate nevertheless
worsened 9.4531 → 9.4680 on its own Dev160 contract, improved only 2/5 folds,
and had a component-bootstrap interval crossing zero. It was held before the
confirmation labels were opened.

## Repository layout

```text
README.md
portfolio_notebook.ipynb
portfolio_notebook_executed.ipynb       # produced by make reproduce
evidence/                               # source-backed and generated evidence
src/rogii_portfolio/                    # complete model implementation
scripts/                                # bootstrap, reproduce, notebook, verify
configs/                                # smoke, full and exact-artifact contracts
tests/                                  # target isolation and end-to-end tests
docs/                                   # implementation map and method cards
requirements.txt / environment.yml
LICENSE
Makefile
```

## Research lessons

I ran into four distinct overfitting mechanisms in one project: geological
split leakage, adaptive reuse of transfer panels, repeated search inside a
single PF neighbourhood, and operational overfitting to an unrepresentative
runtime canary. The useful part was discovering that each one needs its own
control and that the controls do not substitute for each other. A component
graph does nothing about research-selection bias. A bootstrap does not price in
the uncertainty created by trying many ideas. And a numerically correct CSV is
no evidence that hidden execution will finish in time.

That last one was expensive. In a code competition the model, the queue and the
hidden execution time are all part of the method, whether or not you chose to
think of them that way. My final run landed in the displayed bronze-score band
and completed after the cutoff, which made it ineligible. It was a
disappointing way to learn the lesson, but it did change how I define an
experiment: the runtime protocol and the artifact manifest now get frozen
alongside the estimator, not after it.

On a lighter note, the competition quietly handed me a mathematics curriculum.
Bayesian filtering, HMM dynamic programming, generalized least squares, RTS
smoothing, connected components, signal processing, constrained optimization.
Each one arrived because a specific part of the problem refused to work without
it, which turns out to be a much better way to learn a topic than picking it
off a syllabus. Filtering and constrained optimization are where I want to go
deeper next.
