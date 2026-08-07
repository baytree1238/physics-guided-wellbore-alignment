# Real-data nested validation result

## Contract

- Official raw training wells available: **320**
- Graph at the frozen 1,000-ft / 0.95 similarity rule: **243 components**
- Primary panel: **160 wells, 119 complete components**
- Complement panel: **160 wells, 124 complete components**
- Overlap between panels: **0 components**
- Each panel: **5 outer × 3 inner component folds**, plus a run-level
  component holdout
- Reduced CPU inference: PF 2 seeds × 32 particles, GeoHMM stride 24

The clean-room parent and every stage below are scored on identical rows inside
each contract. These values are not leaderboard estimates and are not directly
comparable with the historical 6.372 notebook.

## Stage RMSE

| Stage | Primary OOF | Primary holdout | Complement OOF | Complement holdout |
|---|---:|---:|---:|---:|
| Clean incumbent | 15.5551 | 14.5432 | 13.5177 | 15.4761 |
| HGRG | 14.9121 | 13.9264 | 12.7094 | 14.8275 |
| Meta-State | 14.9593 | 14.0037 | 12.6550 | 14.8825 |
| Prefix-Boundary | 14.9320 | 13.9832 | 12.6386 | 14.8591 |
| Sequential final | **14.8993** | **13.9504** | **12.6000** | **14.8174** |

Sequential gain versus the same clean incumbent was +0.6558 and +0.9177 ft on
the two outer-OOF panels. Run-level holdout gains were +0.5928 and +0.6587 ft.
The primary holdout component-bootstrap 95% interval was [+0.1290,+1.0245].
The complementary holdout interval was [-0.0159,+1.2297], so its lower bound
barely crossed zero despite a 97.2% positive bootstrap probability.

## Frozen expanded-policy failure

Primary160 made Ridge look unusually strong (13.1629 OOF). I froze an expanded
policy before the complementary run completed:

- 0.7393 Ridge
- 0.0486 nonlinear
- 0.2121 sequential final
- effectively zero weight on the other arms

It failed on complementary outer OOF: incumbent 13.5177, sequential 12.6000,
expanded frozen 14.3089, Ridge-only 16.5376. The expanded policy's gain was
-0.7912 ft and its component-bootstrap interval was [-6.7286,+3.7809]. It is
rejected. Its strong 11.7473 score on the much smaller complementary run-level
holdout was recorded as well. The frozen complementary OOF result controls the
selection, so the policy stays rejected.

## Uncertainty and post-hoc controls

The predeclared well-level regret router abstained on 64.7% of primary holdout
wells and improved the incumbent by 0.5757 ft, with interval
[+0.1148,+1.0051]. It nevertheless trailed the simple sequential expert by
0.0171 ft, so it was not promoted.

A post-hoc Meta-State removal selected $\alpha=0$ on primary OOF and improved
sequential by 0.0481 ft there. It then worsened complementary OOF by 0.0543 ft.
Because the hypothesis was conceived after complementary stage metrics were
visible, it remains `POSTHOC_HOLD` regardless of the individual holdout gains.

## Selected path

The Ridge-heavy mixture did not transfer to the complementary panel. The
bounded HGRG-centered path improved both non-overlapping component panels and
remains the selected method.
