# Physics-Guided Wellbore Alignment

Recovering hidden geological depth along a horizontal well, using only the
part of the well that was actually measured.

The setting is the
[ROGII Wellbore Geology Prediction competition](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction).
For each horizontal well the trajectory and gamma-ray log are known, but true
vertical thickness (TVT) is observed only near the heel. Everything past that
point has to be inferred.

I treat this as trajectory reconstruction rather than row-wise regression. A
learned reference path sets the starting point. A particle filter and a
sequence aligner then argue about where the geology actually went, and a gate
built for this problem, the Hierarchical Geology Regret Gate, decides how far
the reference is allowed to follow them. Regional structure and two small
boundary corrections come last, each with its own movement budget.

Two decisions shaped the project more than the model did. Validation splits on
connected geological components instead of wells, because neighbouring wells
leak into each other. And every failed transfer experiment stayed in the
repository next to the contract that rejected it.

## Project at a glance

| Item | Summary |
|:--|:--|
| Research question | Can noisy log alignment and geological structure improve a strong tabular reference without inventing implausible well paths? |
| Core methods | Particle filter, GeoHMM, HGRG, regional state-space fusion, bounded boundary and shape corrections |
| Validation unit | Connected geological components, not rows or individual wells |
| Primary metric | Row-level RMSE in feet |
| Historical submitted run | 6.536 visible RMSE, 9.091 hidden RMSE |
| Reproducible package | [`src/rogii_portfolio/`](src/rogii_portfolio/) |
| Best place to start | [Executed research notebook](portfolio_notebook_executed.ipynb) |

That hidden score belongs to the whole submission pipeline, so it is one
observation of the full path and not an ablation result for any single overlay.
The run also finished after the competition deadline and was never eligible for
the official ranking. Both facts are here because leaving either out would
misrepresent the experiment.

## The problem

Each horizontal well contains:

- `MD`: measured depth along the borehole;
- `X`, `Y`, `Z`: the three-dimensional well trajectory;
- `GR`: a noisy gamma-ray log with occasional missing values;
- `TVT_input`: TVT observed only before the prediction boundary;
- a paired type well with a reference `GR(TVT)` curve.

TVT is a geological coordinate, so the target cannot be recovered from the
borehole geometry alone. Gamma-ray patterns are indirect and often ambiguous:
similar motifs can occur at several depths, measurements can be missing, and
faults or changing dip can break a simple one-to-one match.

I model the latent structural coordinate

$$
U(s) = TVT(s) + Z(s), \qquad TVT(s) = U(s) - Z(s),
$$

where $s$ is measured depth. This separates the known vertical motion of the
borehole from the geological surface that must be inferred.

## Method

```text
component-safe training data
            |
            v
learned local reference
            |
            +-----------------------+
            |                       |
            v                       v
     particle trajectories     GeoHMM alignment
            |                       |
            +-----------+-----------+
                        v
               bounded HGRG update
                        |
                        v
              regional Meta-State
                        |
                        v
             prefix boundary condition
                        |
                        v
           conditional GeoHMM shape update
```

### Learned reference

The reference model uses trajectory geometry, prefix summaries, gamma-ray
descriptors, and type-well features. The model and preprocessing state are
refitted in each outer fold without the held-out geological component.

### Particle filter

The particle filter propagates several structural paths forward from the
visible heel. When the log admits no unique alignment, this keeps competing
datum and local-rate hypotheses alive instead of collapsing to one of them
early.

### GeoHMM

The GeoHMM aligns the horizontal suffix against the type-well log on a grid of
TVT and local slope. It sees the same data as the particle filter but searches
it differently, so it fails differently too. That is the point: it serves as a
second opinion, never as a standalone predictor.

### HGRG

The Hierarchical Geology Regret Gate is the piece I designed specifically for
this problem. Both physics experts can be confidently wrong at the same time,
so the question is not which one to trust but how much movement to authorise at
all. HGRG reads the disagreement between the reference, the particle filter and
the GeoHMM as a risk signal available at prediction time, and converts it into
a shrinkage coefficient. Every resulting correction is then capped twice, by a
well-level RMS budget and a row-level limit.

### Regional Meta-State

A surface fitted on outer-training wells provides the regional observation.
Correlated generalized least squares and a state-space smoother combine it
with the local experts. Prediction-time reliability sets the movement size.

### Boundary and shape corrections

The boundary correction extends the visible-prefix tangent into the first part
of the hidden interval and decays with distance. The shape correction uses a
centered GeoHMM and particle-filter disagreement signal, preserving local
shape information without shifting the overall well datum.

## Design choices

- **Protected-parent inference:** physics experts propose bounded moves from a
  stable learned path.
- **HGRG:** relative PF/GeoHMM disagreement measures risk, while an independent
  Ridge direction provides weak support for or against the proposed move.
- **Geometric role separation:** regional trend, prefix continuity, and local
  shape are handled by separate corrections with separate movement budgets.
- **Target-isolated regional fusion:** spatial structure is fitted only on
  outer-training wells, calibrated on the target's visible prefix, and fused
  with explicitly correlated local experts.
- **Validation as part of the estimator:** the same observable geological
  component graph governs fitting, outer validation, holdout assignment, and
  uncertainty resampling.
- **Research governance:** frozen policies, negative-transfer results, and
  historical lineage are recorded with the reported scores.

See the [methodology deep dive](docs/METHODOLOGY_DEEP_DIVE.md) for the equations,
four system diagrams, and a guide to the saved trajectory figure. The
[technical methodology report](evidence/methodology_report.html) provides the
same material with validation charts and source metadata.

## Validation design

A random row split is far too optimistic here, since adjacent rows in the same
well carry almost the same geological error. Splitting by well is the obvious
fix and still leaks: nearby wells, a shared type well, or simply similar
gamma-ray curves can land on opposite sides of a fold and hand the model the
answer.

So the split happens before any model is fitted. Wells are connected by
observable geological similarity, and complete connected components, never
fragments of one, are assigned to development folds or the run-level holdout.
The same components serve as bootstrap units, which keeps the uncertainty
estimate honest about what is actually being resampled.

The promotion rule was:

1. Define one parent prediction and one testable change.
2. Tune on the declared development population only.
3. Freeze the policy before inspecting transfer labels.
4. Evaluate on a component-disjoint or previously frozen panel.
5. Check pooled error, fold consistency, component uncertainty, and harmed-well rate.
6. Retain, shrink, or reject the change based on that contract.

Throughout the project,

$$
\operatorname{gain} = RMSE_{parent} - RMSE_{candidate},
$$

so a positive value indicates improvement. Results are compared only when the
parent, population, folds, and scoring contract match.

## Evidence

These rows come from different experiment populations. They are deliberately
not stacked into one cumulative ladder, because their parents and scored rows
differ.

| Experiment | Validation contract | Parent RMSE | Candidate RMSE | Gain |
|:--|:--|--:|--:|--:|
| PF with 20% GeoHMM | 40-well discovery panel | 7.5072 | 6.5775 | +0.9297 ft |
| HGRG projection | 80-well transfer surrogate | 7.8299 | 7.5006 | +0.3293 ft |
| Regional Meta-State | Frozen 160-well confirmation | 9.8871 | 9.1804 | +0.7067 ft |
| Nested residual stack | Same 160-well confirmation | 9.1804 | 9.1087 | +0.0717 ft |
| Prefix boundary | Exact80 trajectory | 7.4324 | 7.4141 | +0.0183 ft |
| Boundary with conditional shape | Exact80 composition | 7.4324 | 7.3986 | +0.0338 ft |

The clean-room nested pipeline was also run on two component-disjoint halves of
the 320-well universe, which is the closest thing here to an honest
generalization test:

| Panel | Parent outer OOF | Sequential outer OOF | Gain |
|:--|--:|--:|--:|
| Primary160 | 15.5551 | 14.8993 | +0.6558 ft |
| Complement160 | 13.5177 | 12.6000 | +0.9177 ft |

The detailed evidence trail is available in the
[real-data report](docs/REALDATA_NESTED_RESULTS.md)
and [robustness audit](docs/ROBUSTNESS_EVALUATION.md).

## Negative results

Six plausible ideas did not survive contact with a frozen transfer panel. They
stay in the repository because the reasons they failed were more informative
than the wins:

- **Directional structural field.** Reversed sign on frozen transfer.
- **Learned well-level regret router.** Abstained on 64.7% of holdout wells and
  still trailed the plain sequential expert by 0.0171 ft. The uncertainty
  signal was real; it simply was not sharp enough to pick an expert.
- **Ridge-heavy expanded policy.** The strongest candidate on Primary160 and
  the clearest failure on the component-disjoint panel (−0.7912 ft). One
  component alone reached 149.5 ft RMSE against the incumbent's 33.6 ft.
- **Four-regime switching-state model.** Worse on Primary160 OOF (15.2866 vs
  14.8993) and on its run-level holdout (14.6019 vs 13.9504).
- **Longer heel-slope initialization** and **three-dimensional chord
  increments.** Rejected on transfer; no per-experiment numbers were kept.

All six are disabled in the default path. The Ridge-heavy failure is the one I
would show first: it looked excellent on the panel that selected it, which is
exactly what a leaking validation design is supposed to prevent me from
believing.

## Repository guide

```text
.
├── README.md                         # portfolio overview
├── portfolio_notebook_executed.ipynb # rendered research narrative
├── src/rogii_portfolio/              # model and validation implementation
├── tests/                            # leakage, contract, and pipeline tests
├── configs/                          # frozen experiment configurations
├── evidence/                         # curated figures and result tables
├── artifacts/                        # local experiment outputs, ignored by Git
├── docs/                             # guides, method cards, and result notes
├── scripts/                          # reproducibility and audit entry points
└── Makefile                          # documented research workflows
```

For a guided source-code tour, see the
[codebase guide](docs/CODEBASE_GUIDE.md). The
[implementation map](docs/IMPLEMENTATION_MAP.md)
links each research claim to code and evidence. The
[method cards](docs/METHOD_CARDS.md) collect the
equations, complexity, assumptions, and known failure modes. The
[methodology deep dive](docs/METHODOLOGY_DEEP_DIVE.md) provides the most
detailed explanation of the original research design.

## Reproduce the clean-room study

From the repository root:

```bash
make reproduce
```

On Windows or a machine without GNU Make:

```bash
python reproduce.py
```

The command creates the environment, runs the synthetic nested pipeline,
executes the notebook and tests, and verifies the generated files by SHA-256.

Official competition data are not redistributed. To run the full clean-room
pipeline with a local competition archive:

```bash
make reproduce-full DATA_ROOT=/path/to/rogii-wellbore-geology-prediction.zip
```

See the [reproducibility guide](docs/REPRODUCIBILITY.md) for the
smaller integration profile, the two-panel research protocol, and Kaggle
publishing instructions.

## Scope and limitations

- The clean-room package reproduces the algorithmic research path, not the
  exact bytes of the historical competition submission.
- Exact historical reconstruction needs external model artifacts that are not
  redistributed here. The adapter requires their hashes and fails closed
  without them.
- Some panels had already been inspected in earlier studies. Results measured
  on them are replication evidence, not a fresh final test, and are labelled
  post-hoc wherever they appear.
- The source data never establish a coordinate reference system, so every
  spatial threshold is reported in raw coordinate units.
- Prediction-time disagreement is a useful risk feature. It is not a calibrated
  probability of geological error, and nothing here should be read as one.

## Research record

What this repository is really recording is a decision process: how the problem
was framed, why each algorithm was given a movement budget instead of free
rein, how the target was isolated, which hypotheses were rejected and on what
contract, and which historical results remain out of reach without external
artifacts.

The implementation and research decisions are my own. Competition data and
community materials remain subject to Kaggle's rules and their original
licenses. ROGII did not sponsor or maintain this project.
