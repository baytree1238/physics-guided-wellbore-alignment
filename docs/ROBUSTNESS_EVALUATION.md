# Robustness evaluation of the saved real-data predictions

## Scope

I evaluated the four saved primary/complement contracts at row, well, and
geological-component grain. The artifacts are internally consistent and useful
for comparing failure modes. They do not establish training-seed stability,
because all four prediction files came from seed `20260806`.

The evaluation covers 1,556,878 scored rows, 320 wells, and 243 disjoint global
components. Source hashes, required columns, row keys, finite values, well-to-
component mappings, reconstructed splits, stored summary RMSEs, and the ten
stored OOF fold scores all reconciled. The run recorded 85 passed checks and no
failures.

## Metrics

- Pooled row RMSE gives each row equal weight, so long wells and large
  components matter more.
- Macro-well RMSE averages the RMSE of each well with equal weight.
- Macro-component RMSE gives each geological component equal weight.
- Harmed-well rate is the share of wells with worse RMSE than the incumbent.
- Worst-10% component CVaR is the mean RMSE of the worst
  `ceil(0.10 * components)` components.
- Horizon scores use fixed 0-250, >250-500, >500-1000, >1000-2000, and >2000 ft
  bins.

## Main result

The sequential correction is more consistent than Ridge across folds and
contracts.

| Contract | Incumbent pooled | Sequential pooled | Sequential macro-component | Harmed wells | Sequential worst-10% CVaR |
|---|---:|---:|---:|---:|---:|
| Primary OOF | 15.5551 | 14.8993 | 12.1109 | 27.0% | 31.9479 |
| Primary holdout | 14.5432 | 13.9504 | 11.7138 | 26.5% | 28.1947 |
| Complement OOF | 13.5177 | 12.6000 | 9.9958 | 19.5% | 27.8290 |
| Complement holdout | 15.4761 | 14.8174 | 12.2153 | 28.1% | 29.5035 |

Sequential final beat the incumbent in every one of the ten outer folds. Its
mean fold rank was 2.35 with rank standard deviation 0.32; its worst fold still
improved pooled RMSE by 0.2666 ft. It also improved pooled RMSE in all 20
contract-by-horizon cells.

Sequential final still harmed about 20-28% of wells depending on the contract.
On the complement holdout its
worst-10% component CVaR was 29.5035, slightly worse than the incumbent's
29.1741, even though pooled RMSE improved by 0.6587 ft. In the >2000 ft bin of
that contract, 37.5% of wells were harmed.

## Pooled and component metrics

Ridge had the best macro-component RMSE in three of the four contracts, but its
rank standard deviation across the ten OOF folds was 3.07. It won eight folds
and suffered a worst-fold loss of 14.3103 ft versus the incumbent.

On complement OOF, Ridge improved macro-component RMSE from 10.8324 to 10.5323
while worsening pooled row RMSE from 13.5177 to 16.5376. Its worst-10% component
CVaR reached 37.4849 versus 29.6505 for the incumbent. Component 185 alone had
RMSE 149.4963 versus 33.5648 for the incumbent. Equal-component averaging makes
the typical component look acceptable, while pooled and tail metrics expose
the large failure. A promotion rule needs all three views.

## Repeated evaluation

For each contract I ran 2,000 whole-component bootstrap repeats and recomputed
macro-component ranks. Sequential final beat the incumbent in 100.0%, 99.95%,
100.0%, and 99.75% of repeats for primary OOF, primary holdout, complement OOF,
and complement holdout respectively.

These are evaluation-sample seeds. They measure sensitivity to which components
are represented, not sensitivity to model fitting randomness. A true seed
claim needs new prediction artifacts from at least three independently refit
training seeds, registered under the same split contract.

## Recommended gate

For the next candidate, I would predeclare a gate that requires:

1. positive pooled and macro-component gain on both OOF panels;
2. positive fold gain in at least 8 of 10 outer folds;
3. no material worst-10% CVaR regression on the complementary OOF panel;
4. harmed-well rate reported beside the headline score; and
5. the same direction over at least three model-training seeds.

This keeps sequential final as the stable reference and prevents a high-mean,
high-tail-risk fit such as Ridge from being promoted on one aggregation alone.

## Reproduction

```bash
.venv/bin/python scripts/evaluate_robustness.py
```

The registry is `configs/robustness_registry.json`. Results and their hash
manifest are in `artifacts/robustness_evaluation/`.
