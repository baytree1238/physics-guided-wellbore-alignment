# Historical-parity ledger

This repository separates two claims that are easy to confuse:

1. the clean-room pipeline is an executable end-to-end research and deployment
   implementation;
2. the 9.091 historical run is a scored lineage whose exact bytes require
   pinned external artifacts.

| Historical block | Status here | What prevents an exact claim |
|---|---|---|
| 121 SAFE feature formulas | Vendored and source-hash verified | Nothing for feature calculation itself |
| PF seed aggregation | Formula-faithful module | Exact upstream parent package still matters |
| no-prior GeoHMM | Formula-faithful module | Exact runtime/library numerics are not promised |
| HGRG | Formula-faithful frozen equation | Requires exact historical incumbent/Ridge/PF/HMM paths |
| conditional shape | Formula-faithful frozen equation | Requires exact stride-6 HMM and parent paths |
| scored incumbent | Adapter only | SP45/beam selection, IRLS projection, prefix calibration, contact/Q0522, DTRT and external pretrained packages are not rebuilt here |
| scored structural Meta-State | Clean-room analogue | q25 anchor, horizon ramp, 3-ft pre-consensus projection and historical transport/source rules differ |
| scored Prefix-Boundary | Clean-room analogue | historical visible-prefix rolling selection among 64/128/256/512/1024-ft windows is not ported |
| exact `submission.csv` | Fail closed | external Kaggle artifacts have no verified local paths and SHA-256 values in `configs/exact_artifacts.json` |

`src/rogii_portfolio/parents.py` is the boundary: a historical expert table can
enter only when its bytes and exact ID coverage match a pinned manifest. Until
then, every result produced by `pipeline.py` is labelled `retrained_cleanroom`.
