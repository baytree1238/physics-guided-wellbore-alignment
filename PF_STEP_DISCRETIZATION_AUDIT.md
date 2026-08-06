# PF motion-step discretization audit

## Decision

**Do not change the production PF.**  The proposed continuous-time
`delta_MD` correction is mathematically sound for irregular sampling, but the
ROGII suffix rows in both audited panels are sampled at exactly 1.0-ft MD
increments.  It is therefore exactly the incumbent update and has exactly zero
RMSE effect.  Using XYZ chord length creates a small blend gain after transfer,
but the effect is fold-unstable and the physical coordinate is less correct
than measured depth.

No submission file was created.

## Proposed correction

For incumbent rate persistence `a=0.998`, rate noise `sigma_v=0.002`, position
noise `sigma_p=0.005`, and step `d`, the audited update was:

```text
a_d       = a ** d
sigma_v_d = sigma_v * sqrt((1 - a ** (2*d)) / (1 - a ** 2))
sigma_p_d = sigma_p * sqrt(d)
rate      = a_d * rate + sigma_v_d * epsilon_v
position += rate * d + sigma_p_d * epsilon_p
```

Only MD/X/Y/Z/GR/TVT_input and typewell TVT/GR enter prediction.  Suffix TVT is
loaded after prediction for alignment and RMSE scoring.

## Exact MD finding

| Panel | Suffix rows | min ΔMD | max ΔMD | fraction ΔMD != 1 | RMSE change |
|---|---:|---:|---:|---:|---:|
| Dev40 | 191,205 | 1.0 | 1.0 | 0.0% | **0.000000** |
| Exact80 | 394,934 | 1.0 | 1.0 | 0.0% | **0.000000** |

At `d=1`, all three transformed quantities reduce exactly to `a`, `sigma_v`,
and `sigma_p`.  No Monte Carlo rerun is needed to establish identity.

## XYZ chord-length experiment

The 3-D straight-line chord between adjacent X/Y/Z samples ranged from about
0.986 to 1.014 ft, with median essentially 1.0 ft.  The fixed Dev40 experiment
used 8 seeds, 500 particles, temperature 8, and four CPU workers.

| Panel / arm | Reference RMSE | Candidate RMSE | Gain | Positive folds |
|---|---:|---:|---:|---:|
| Dev40: incumbent PF → chord PF | 7.507195 | 7.337679 | +0.169516 | 4/5 |
| Exact80 transfer: incumbent PF → chord PF | 8.873345 | 8.951019 | **-0.077674** | 3/5 |
| Dev40: PF25+stride6 → chord-PF25+stride6 | 6.445572 | 6.351102 | +0.094470 | 3/5 |
| Exact80: PF25+stride6 → chord-PF25+stride6 | 8.213131 | 8.162747 | +0.050384 | 3/5 |
| Exact80: PF20+stride6 → chord-PF20+stride6 | 8.196841 | 8.167353 | +0.029488 | 3/5 |

Dev40 chord-PF fold gains versus PF were:

```text
+0.480928, +0.223338, -1.430161, +0.537970, +0.777212 ft
```

Exact80 chord-PF fold gains were:

```text
-1.483462, +0.673397, -0.918459, +0.310758, +0.120468 ft
```

Thus the standalone PF signal reverses on transfer, and the blend increment is
small and positive in only three folds.  The apparent blend gain is not strong
enough to replace the audited production trajectory.

## Physical interpretation

MD is intended to approximate along-hole arc length.  Adjacent XYZ distance is
a chord, so it is necessarily an approximation to that arc and may also include
coordinate rounding.  The roughly ±1% perturbation changes the particle RNG
trajectory and likelihood resampling enough to create noisy panel effects, but
does not add geological information.  If a future dataset has irregular MD
sampling, the continuous-time formula should be retained as a principled
generalization; it is simply inactive for this competition's 1-ft rows.

## Artifacts

- `rogii_pf_step_discretization_audit.py`
- `pf_step_discretization_dev40_evidence/summary.json`
- `pf_step_discretization_dev40_evidence/scores.csv`
- `pf_step_discretization_exact80_evidence/summary.json`
- `pf_step_discretization_exact80_evidence/scores.csv`

Runtime was **34.44 s** on Dev40 and **94.19 s** on Exact80 with four CPU
workers.
