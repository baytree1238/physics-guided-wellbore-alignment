# Codebase guide

This guide gives a short tour of the repository. It starts from the prediction
flow and then maps each research block to its implementation.

## Prediction flow

The pipeline predicts one continuous TVT trajectory for each horizontal well.
It does not treat rows as unrelated samples. Three types of information are
combined:

1. Observable local features provide a strong reference prediction.
2. The particle filter and GeoHMM propose physically coherent alignments.
3. Bounded overlays decide how much the reference is allowed to move.

Learned objects are fitted inside their geological component folds. Target
values are stored separately from model inputs to prevent suffix leakage.

## Data flow

```text
WellRecord
   |
   +--> prepare_inference_well() --> PreparedWell
   |                                  contains no suffix target
   |
   +--> prepare_scored_well() -----> ScoredPreparedWell
                                      keeps targets in a scoring wrapper

PreparedWell
   |
   +--> build_fast_safe_features() --> 121 observable features
   +--> run_physics() ----------------> particle and GeoHMM paths
   +--> StructuralSurface.predict() --> regional structural path
                                          |
                                          v
        incumbent -> HGRG -> Meta-State -> boundary -> shape
                                          |
                                          v
                                   PredictionBundle
                                          |
                                          v
                              component-aware evaluation
```

`run_nested_experiment()` in `pipeline.py` owns the complete training and
evaluation sequence. It builds target-free per-well caches, creates outer and
inner component folds, refits learned models, produces out-of-fold paths, fits
the residual stack, and evaluates one untouched component holdout.

## Core data contracts

Start with the contracts in `contracts.py`.

| Contract | Purpose |
|:--|:--|
| `WellRecord` | Raw horizontal well and paired type-well log |
| `PreparedWell` | Validated inference object with no `TVT` target column |
| `ScoredPreparedWell` | Inference object plus a separate suffix target array |
| `PipelineConfig` | Frozen split, model, inference, and evaluation settings |
| `PredictionBundle` | Row-aligned predictions, labels, metadata, and diagnostics |

`PreparedWell` contains inference inputs. `ScoredPreparedWell` adds the target
through a separate scoring wrapper. Tests can therefore poison or remove the
target without changing model input.

## Source map

### Validation and orchestration

| Module | Responsibility |
|:--|:--|
| `contracts.py` | Schemas and target-isolation rules |
| `components.py` | Geological similarity graph and component-disjoint splits |
| `pipeline.py` | Nested fitting, out-of-fold prediction, stacking, and holdout evaluation |
| `evaluation.py` | Row, well, component, horizon, tail, fold, and bootstrap summaries |
| `artifacts.py` | Deterministic evidence files, figures, hashes, and manifests |
| `reproduce.py` | Synthetic and official-data reproduction workflow |

### Predictive models

| Module | Responsibility |
|:--|:--|
| `features.py` | Readable 121-feature clean-room implementation |
| `historical_features.py` | Hash-audited historical feature body |
| `models.py` | Ridge and nonlinear reference models |
| `particle.py` | Seed-level particle trajectories and evidence aggregation |
| `geohmm.py` | Grid-based sequence alignment with forward and backward inference |
| `physics.py` | Input preparation, physics expert execution, and regional surface fitting |

### Bounded fusion and research controls

| Module | Responsibility | Default role |
|:--|:--|:--|
| `hgrg.py` | Disagreement-aware move from the parent toward the physics bridge | Main path |
| `meta_state.py` | Correlated expert fusion and state-space smoothing | Main path |
| `overlays.py` | Prefix-boundary and local-shape corrections | Main path |
| `stack.py` | Inner out-of-fold convex residual stack | Research control |
| `switching_state.py` | Four-regime state-space model | Disabled negative control |
| `regret_router.py` | Well-level expert selection with abstention | Disabled negative control |
| `group_robust.py` | Component-risk-aware blending | Transfer study |
| `trust_region.py` | Small capped move toward Ridge | Optional diagnostic |
| `refit_stability.py` | Hierarchical bootstrap across components and refit seeds | Stability audit |

### Deployment boundary

`deployment.py` refits the selected clean-room pipeline, rejects target-bearing
test inputs, restores sample-submission row order, and records output hashes.
`parents.py` loads historical expert tables only after verifying their expected
hash and row identifiers.

## How to read one prediction

Start in `_predict_one()` in `pipeline.py`. The function receives a cached
target-free well and fold-fitted models, then constructs the candidate path in
the following order:

1. Predict the learned reference and Ridge control.
2. Retrieve the particle-filter and GeoHMM trajectories.
3. Apply HGRG under well RMS and row movement limits.
4. Fuse local and regional information with Meta-State.
5. Apply the one-sided prefix boundary correction.
6. Add the centered conditional shape correction.
7. Emit optional research arms only when their config flags are enabled.

The order matters because each stage treats the previous stage as its protected
parent. Reported gains should therefore name both the candidate and its parent.

## How leakage is prevented

No single check is trusted on its own. Seven independent safeguards overlap, so
that any one of them failing silently still leaves the boundary intact:

- `model_frame()` returns only permitted inference columns.
- `PreparedWell.validate()` rejects a frame containing `TVT`.
- Suffix values in `TVT_input` must be missing at inference.
- Connected geological components are formed before fold assignment.
- Learned reference models, spatial surfaces, and stack weights are refitted
  inside their declared folds.
- Bootstrap sampling draws complete components rather than rows.
- Historical artifacts require both a content hash and exact row identifiers.

Together these define the leakage boundary the experiments are measured
against. `tests/test_contracts.py` attacks that boundary directly: it shifts
the suffix target by 100,000 ft and asserts the model-facing frame comes back
identical, while the scoring wrapper correctly registers the change.

## Clean-room code and historical code

There are two feature-building paths.

`features.py` is written for readability and nested retraining. It reproduces
the 121-column schema with formulas derived for this repository.

`historical_features.py` contains an audited source body recovered from the
historical notebook. Its formatting and internal comments are intentionally
preserved because a source hash verifies its identity. Treat that section as a
vendored research artifact rather than the style reference for new code.

The clean-room reference in `models.py` is separate from the historical scored
parent. Historical expert paths are loaded through the hash-verified adapter
in `parents.py`.

## Experiments, evidence, and artifacts

The repository separates three kinds of files:

- `configs/` records the settings that define an experiment.
- `evidence/` contains the curated files used by the public notebook.
- `artifacts/` contains run-specific predictions, diagnostics, and summaries.

The JSON manifests record file hashes so a reproduced result can be checked
against its inputs. Large prediction tables are compressed because they are
evidence for audits, not the preferred entry point for a reader.

The scripts in `scripts/` are command-line entry points. Most only parse paths
and configuration before calling package functions. Research logic belongs in
`src/rogii_portfolio/`, where it can be imported and tested.

## Suggested reading order

For a 15-minute review:

1. Root `README.md`
2. `portfolio_notebook_executed.ipynb`
3. `docs/IMPLEMENTATION_MAP.md`
4. `contracts.py`
5. `_predict_one()` and `run_nested_experiment()` in `pipeline.py`

For a method review:

1. `docs/METHODOLOGY_DEEP_DIVE.md`
2. `evidence/methodology_report.html`
3. `docs/METHOD_CARDS.md`
4. `particle.py` and `geohmm.py`
5. `hgrg.py`, `meta_state.py`, and `overlays.py`
6. `docs/REALDATA_NESTED_RESULTS.md`
7. `docs/ROBUSTNESS_EVALUATION.md`

For a reproducibility review:

1. `Makefile`
2. `scripts/bootstrap.py`
3. `reproduce.py` and `pipeline.py`
4. `artifacts.py`
5. `tests/test_contracts.py` and `tests/test_nested_smoke.py`

## Running the project

The repository-only smoke study is the default:

```bash
make reproduce
```

To run tests without rebuilding evidence:

```bash
make test
```

To exercise the official input schema with reduced inference budgets:

```bash
make realdata-smoke DATA_ROOT=/path/to/data
```

The smoke profile validates wiring and contracts. It is not a score estimate.
The larger profiles are documented in `REPRODUCIBILITY.md` and are much more
computationally expensive.

## Safe extension points

When adding a new expert or correction:

1. Define its observable inputs explicitly.
2. Add it as a candidate arm without changing the protected parent.
3. Fit any learned state inside inner or outer component folds as appropriate.
4. Add a target-poisoning or target-removal test.
5. Record movement limits and a failure condition.
6. Compare it on one declared contract before promoting it to the default path.

Record the implementation change and the selection decision separately.
