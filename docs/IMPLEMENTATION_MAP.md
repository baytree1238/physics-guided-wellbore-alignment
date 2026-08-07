# Implementation map

For the reasoning behind these modules, start with
[`METHODOLOGY_DEEP_DIVE.md`](METHODOLOGY_DEEP_DIVE.md). It separates the
established mathematical foundations from the project-specific formulation,
gating rules, role constraints, and validation design.

| Research block | Executable implementation | Test/evidence | Historical fidelity |
|---|---|---|---|
| Raw data and fail-closed inference view | `src/rogii_portfolio/io.py`, `contracts.py` | truth-free model dataclass, prefix topology assertions | New stricter contract |
| Geological component graph | `components.py` | component-disjoint assertions and `geological_components.csv` | Validation only |
| Particle filter | `particle.py` | deterministic seed paths and predictive evidence | Formula-faithful research implementation |
| GeoHMM | `geohmm.py` | no-prior forward/backward smoother | Formula-faithful no-prior branch |
| 121 local-observable features | `features.py` (readable clean-room formulas); `historical_features.py` (frozen audited formulas) | schema parity, source hashes, deterministic rerun and suffix-truth poison test | No; separate control |
| FAST-SAFE residual models | `models.py` | refit inside every outer fold | No; separate control |
| Historical scored parent | `parents.py` | pinned expert-table SHA and exact-ID adapter | External artifact required; fail closed |
| Retrained clean parent | `models.py` | fitted inside every outer fold | Clean-room analogue; not the scored incumbent |
| HGRG | `hgrg.py` | movement budgets and diagnostics | Formula-faithful overlay |
| Structural RBF + Meta-State | `physics.py`, `meta_state.py` | outer-training-only RBF, GLS and RTS | Clean-room analogue; historical q25/ramp/transport differ |
| Switching-state pilot | `switching_state.py` | synthetic regime controls + Primary160 OOF/holdout | Default-off negative research control |
| Prefix-Boundary | `overlays.py` | one-sided prefix-only slope | Clean-room fixed 256-ft window; historical branch selected among five windows |
| Conditional shape | `overlays.py` | mean removal, target-free gate and projection | Formula-faithful overlay |
| Nested residual stack | `stack.py` | inner OOF weights, outer OOF evaluation | No; separate control |
| Pairwise-regret abstention | `regret_router.py` | component-cross-fitted selector + frozen transfer | New research control |
| Component-risk evaluation | `evaluation.py`, `configs/robustness_registry.json` | 85 integrity checks; row/well/component/CVaR metrics | Post-hoc evaluation layer |
| Group-robust fusion | `group_robust.py` | repeated component folds + bidirectional frozen transfer | Asymmetric HOLD |
| Capped Ridge trust region | `trust_region.py` | weight sweep + four-contract component bootstrap | Default-off post-hoc candidate |
| Test-to-CSV path | `deployment.py` | target rejection, exact ID-order and SHA checks | Clean-room deployment only |
| Full experiment | `pipeline.py`, `reproduce.py` | outer OOF + run-level component holdout | Strong clean-room contract; not exact 9.091 reconstruction |

The exact scored notebooks also load third-party Kaggle model artifacts. Their
presence and version hashes are required for exact-byte reproduction.
`configs/exact_artifacts.json` makes that dependency fail closed. The source
package supports algorithmic retraining from official raw data, but a retrained
parent is not byte-identical to the historical scored parent.

### Why there are two 121-feature builders

`features.py` is the readable implementation used by the new fully nested
experiment. It preserves the frozen 121-column names, but its formulas were
re-derived for this repository. `historical_features.py` vendors the actual
target-isolated notebook formulas, including the 600-particle filters, seven
beam searches, multi-scale NCC and original float32 transforms. Its source
records both the original cell SHA-256 and the transformed-body SHA-256.
Neither implementation is presented as the other: schema parity is asserted,
while numerical parity is claimed only for deterministic reruns of the frozen
historical body.
