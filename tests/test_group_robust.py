import numpy as np
import pandas as pd

from rogii_portfolio.group_robust import apply_group_robust_blend, fit_group_robust_blend


def test_group_robust_policy_obeys_simplex_and_reduces_worst_group():
    rows = []
    for component, offset in (("a", 0.0), ("b", 4.0), ("c", -3.0)):
        for index in range(20):
            truth = offset + 0.1 * index
            rows.append(
                {
                    "component": component,
                    "truth": truth,
                    "incumbent": truth + (2.0 if component == "b" else 1.0),
                    "stable": truth + 0.4,
                    "fragile": truth if component != "b" else truth + 5.0,
                }
            )
    frame = pd.DataFrame(rows)
    policy = fit_group_robust_blend(
        frame,
        ("stable", "fragile"),
        macro_weight=0.5,
        cvar_weight=2.0,
        ridge=0.01,
    )
    prediction = apply_group_robust_blend(frame, policy)
    assert np.all(policy.weights >= 0)
    assert policy.weights.sum() <= 1.0 + 1e-8
    base_worst = frame.assign(error=np.square(frame.truth - frame.incumbent)).groupby("component").error.mean().max()
    candidate_worst = frame.assign(error=np.square(frame.truth - prediction)).groupby("component").error.mean().max()
    assert candidate_worst < base_worst
