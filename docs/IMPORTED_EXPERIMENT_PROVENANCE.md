# Imported iterative experiment provenance

This repository includes four unsquashed experiment commits imported from the
earlier local research repository
[`physics-aware-wellbore-geology`](https://github.com/baytree1238/physics-aware-wellbore-geology).
They preserve two compact discovery-to-transfer reversals that directly support
the negative-result claims in the portfolio.

## What the commit sequence does—and does not—show

These are **archival experiment snapshots committed after the runs**, not a
contemporaneous record of experiment execution. Their original author
timestamps are only 11–13 seconds apart because the completed artifacts were
published in a batch. They do not imply that the models were designed, run,
reviewed, and transferred in those few seconds.

The development audit documents also summarize the later transfer result.
Accordingly, the four-commit sequence preserves the source repository's
publication decomposition—development bundle followed by transfer bundle—but
must not be presented as proof of when labels were opened or how long the
scientific work took. The reports' recorded runtimes and frozen-policy files
are the relevant evidence for execution and decision boundaries.

Each imported commit retains its original author and author date and includes
Git's `(cherry picked from commit ...)` trailer. No imported commit is part of
the package's default execution path.

## Imported commits

| Experiment stage | Source commit | Imported commit | Result |
|---|---|---|---|
| PF initial-state Dev40 audit | [`9255291`](https://github.com/baytree1238/physics-aware-wellbore-geology/commit/9255291feec776165139c58c0fa6492f52018583) | `734b112` | +0.560640 ft; 5/5 folds |
| Frozen initial-state Exact80 transfer | [`ca30354`](https://github.com/baytree1238/physics-aware-wellbore-geology/commit/ca30354b181acc414c0261d8bdacc0472b2f2412) | `8e3ab29` | −1.792153 ft; 3/5 folds; rejected |
| 3-D chord/step Dev40 audit | [`c0bb158`](https://github.com/baytree1238/physics-aware-wellbore-geology/commit/c0bb158a30296219dd01d22f6c9a99c6ed30095d) | `04be8a9` | +0.169516 ft; 4/5 folds |
| Frozen chord Exact80 transfer | [`fcebfa9`](https://github.com/baytree1238/physics-aware-wellbore-geology/commit/fcebfa9a6214e8da9e7bd91022893a740aa8235a) | `28490c6` | −0.077674 ft; 3/5 folds; rejected |

The imported commit IDs above identify this branch's current history. They may
change if the branch is rebased before merge; the full source IDs and
cherry-pick trailers remain the durable provenance anchors.

## Why these two experiments were selected

The source repository contains many more exploratory branches. These two were
selected because they are small enough for a public portfolio, have clear
one-factor hypotheses, include code and persisted evidence, and show an
unambiguous discovery-to-transfer reversal. Larger experiment families and
redundant prediction archives were deliberately not imported.

### Longer PF initial-state estimate

The frozen `endpoint-120` initial-rate estimator improved Dev40 from 7.507195
to 6.946554, then worsened Exact80 from 8.873345 to 10.665499. Small initial
rate changes sent a nonlinear particle filter into different alignment modes,
including catastrophic well-level errors. See
[`PF_INITIAL_STATE_AUDIT.md`](../PF_INITIAL_STATE_AUDIT.md).

### Three-dimensional chord increments

Using adjacent XYZ chord length instead of measured-depth increments improved
Dev40 PF RMSE from 7.507195 to 7.337679, then worsened Exact80 from 8.873345 to
8.951019. The apparent blend improvement was positive in only three folds, and
MD remains the physically correct along-hole coordinate. See
[`PF_STEP_DISCRETIZATION_AUDIT.md`](../PF_STEP_DISCRETIZATION_AUDIT.md).

## Scope and reproducibility limits

- Dev40 is a discovery panel; Exact80 had already been opened elsewhere in the
  research program and is a frozen sensitivity/transfer panel, not a pristine
  external test.
- The scripts use only their declared observable prediction columns, but full
  reruns require the competition data and upstream cached PF/GeoHMM artifacts.
- Persisted summaries, fold scores, hashes, and predictions are included so the
  reported decisions can be audited without claiming that this public package
  independently regenerates every upstream cache.
- Neither experiment created a submission file or changed the deployed model.
- These files are historical standalone runners at repository root. The
  maintained clean-room package remains under `src/rogii_portfolio/`.

## Integrity checks used for this import

The migration compares stable patch IDs between each source and cherry-picked
commit, scans the imported paths for credentials and submission artifacts,
checks file-size limits, runs the focused imported test plus the public test
suite, and verifies that no pre-existing code, results, notebooks, or curated
evidence changed apart from the explicit README disclosure.
