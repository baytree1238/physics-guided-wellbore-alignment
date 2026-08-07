#!/usr/bin/env python3
"""Build the public research notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def md(text: str):
    # Use the same math delimiters in Jupyter, nbconvert and Kaggle.
    normalized = (
        text.strip()
        .replace(r"\[", "$$")
        .replace(r"\]", "$$")
        .replace(r"\(", "$")
        .replace(r"\)", "$")
    )
    return nbf.v4.new_markdown_cell(normalized + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build():
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.11"}
    nb.cells = [
        md(r"""
# Reconstructing hidden well trajectories, and the validation design that kept me honest

## Short version

The task is to continue TVT past the observed part of a directional well, using its geometry, its gamma ray log and a reference type well. My first validation split made the problem look much easier than it is: nearby and nearly identical wells could land on opposite sides of a fold and quietly hand the model its answer. I replaced it with a graph split that keeps each connected geological group intact.

The model grew out of two trajectory estimates that disagree in useful ways, a particle filter and a GeoHMM. HGRG decides how far their agreement is allowed to pull a protected baseline; the state-space and boundary terms make smaller adjustments on top. I ran the complete clean-room version on all 320 training wells. The similarity graph produced 243 components, which I split into two non-overlapping 160-well panels.

One distinction runs through the whole notebook. The code here reruns the clean-room pipeline and the audited 121-feature calculation. It does not recreate the exact 9.091 submission, which depended on external model files that are not redistributed. Those two lineages are kept apart everywhere below, because conflating them would be the easiest way to claim more than I can show.

> To rerun this notebook on Kaggle, attach the companion dataset built from this folder. The saved outputs make it readable on its own, but the code cells load `src/`, `evidence/` and selected files under `artifacts/`.
"""),
        code(r"""
from pathlib import Path
import inspect
import json
import sys
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display

def find_project_root():
    candidates = [Path.cwd()]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(sorted(path for path in kaggle_input.glob("*") if path.is_dir()))
        candidates.extend(sorted(path for path in kaggle_input.glob("*/*") if path.is_dir()))

        archives = sorted(kaggle_input.glob("**/rogii_portfolio_companion.zip"))
        if archives:
            extracted = Path("/kaggle/working/rogii_portfolio_companion")
            marker = extracted / "evidence/reproduction_summary.json"
            if not marker.exists():
                extracted.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archives[0]) as bundle:
                    for member in bundle.infolist():
                        member_path = Path(member.filename)
                        if member_path.is_absolute() or ".." in member_path.parts:
                            raise ValueError(f"Unsafe path in companion archive: {member.filename}")
                    bundle.extractall(extracted)
            candidates.insert(0, extracted)

    for candidate in candidates:
        if (candidate / "evidence/reproduction_summary.json").exists() and (candidate / "src/rogii_portfolio").is_dir():
            return candidate
    raise FileNotFoundError("Attach the ROGII portfolio dataset or run this notebook from the project folder.")

ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "src"))
from rogii_portfolio.artifacts import sha256_file, verify_manifest
from rogii_portfolio.features import SAFE_FEATURES
from rogii_portfolio.historical_features import HISTORICAL_ORIGINAL_BUILDER_SHA256
from rogii_portfolio.hgrg import apply_hgrg
from rogii_portfolio.meta_state import apply_meta_state

summary = json.loads((ROOT / "evidence/reproduction_summary.json").read_text())
manifest = verify_manifest(ROOT)
lineage = json.loads((ROOT / "evidence/method_lineage.json").read_text())

def load_optional(path):
    path = ROOT / path
    return json.loads(path.read_text()) if path.exists() else None

def load_optional_csv(path):
    path = ROOT / path
    return pd.read_csv(path) if path.exists() else None

primary160 = load_optional("artifacts/realdata_nested_160/evidence/reproduction_summary.json")
confirmation160 = load_optional("artifacts/realdata_nested_160_confirmation/evidence/reproduction_summary.json")
frozen_confirmation = load_optional("artifacts/realdata_nested_160_confirmation/frozen_policy_evaluation.json")
regret_router = load_optional("artifacts/realdata_nested_160/regret_router/summary.json")
meta_shrinkage = load_optional("artifacts/realdata_nested_160_confirmation/posthoc_meta_shrinkage.json")
graph_audit = load_optional("artifacts/realdata_nested_160/graph_audit/cross_partition_diagnostics.json")
robustness_summary = load_optional("artifacts/robustness_evaluation/summary.json")
robustness_metrics = load_optional_csv("artifacts/robustness_evaluation/model_metrics.csv")
robustness_ranks = load_optional_csv("artifacts/robustness_evaluation/rank_stability.csv")
robustness_manifest = load_optional("artifacts/robustness_evaluation/manifest.json")
if robustness_manifest is not None:
    robustness_root = ROOT / "artifacts/robustness_evaluation"
    for item in robustness_manifest["artifacts"]:
        path = robustness_root / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Robustness evidence hash mismatch: {item['path']}")
switching_pilot = load_optional("artifacts/realdata_switching_160/evidence/reproduction_summary.json")
trust_region = load_optional("artifacts/trust_region_ridge/summary.json")
deployment_audit = load_optional("evidence/deployment_smoke_audit.json")
print("Project data loaded.")
print(f"Checked {len(manifest['artifacts'])} saved artifacts")
print(f"Validation contract: {summary['contract']}")
"""),
        code(r"""
executive = pd.DataFrame([
    ["Data", summary["wells"], summary["components"]],
    ["Development", summary["development_wells"], summary["development_components"]],
    ["Untouched holdout", summary["holdout_wells"], summary["holdout_components"]],
], columns=["Partition", "Wells", "Geological components"])
display(Markdown("**Deterministic CI smoke (not a score estimate)**"))
display(executive)
display(pd.DataFrame({
    "Outer OOF RMSE": summary["outer_oof_rmse"],
    "Untouched holdout RMSE": summary["untouched_holdout_rmse"],
}).loc[["pf", "incumbent", "hgrg", "meta_state", "prefix_boundary", "sequential_final", "nested_stack"]])
"""),
        md(r"""
## 1. Why I stopped using well-level splits

A well ID is not an independent geological unit, and treating it as one is the mistake this whole design is built around avoiding. I build a graph whose vertices are wells and whose edges mean strong XY proximity or high GR/type-well similarity. Validation then splits on connected components, never on rows or IDs. That tests the exact failure mode that worried me: a model that appears to understand geology when it is really recognising a neighbouring training well.

The graph is built before any fitting happens, and each run-level holdout is chosen before that run's models are trained. Its labels touch nothing: not feature construction, not policy selection, not stack weights.

What this does **not** do is erase my own research history. A panel I have already looked at several times is a transfer panel, not a pristine confirmation, and I label it that way throughout. Two smaller caveats belong here as well. No coordinate reference system was supplied, so spatial thresholds are reported in **raw coordinate units**, and the legacy config fields ending in `_ft` are a naming leftover rather than a verified unit conversion.
"""),
        code(r"""
components = pd.read_csv(ROOT / "evidence/geological_components.csv")
fig, ax = plt.subplots(figsize=(8, 5))
for component, group in components.groupby("component"):
    ax.scatter(group.x, group.y, s=24, alpha=.75)
ax.set_title("Geological similarity components (colour = indivisible split unit)")
ax.set_xlabel("X (raw coordinate units)"); ax.set_ylabel("Y (raw coordinate units)"); ax.grid(alpha=.2)
plt.show()
"""),
        md(r"""
## 2. Keeping the scored pipeline and this reconstruction separate

The historical scored lineage is:

```text
incumbent baseline → HGRG → Meta-State → Prefix-Boundary → conditional GeoHMM shape
```

The 121-feature nonlinear retrain and the nested residual stack were controls, not part of that scored path. Results from the 40-, 80- and 160-well experiments are also not additive, because their parents and populations differ; stacking them into one cumulative ladder would produce a number that means nothing.

The executable nested pipeline runs on a **retrained clean parent**. That is not the historical incumbent, which additionally carried SP45/beam selection, robust polynomial projection, visible-prefix calibration, contact logic, DTRT and external learned packages. HGRG and conditional shape are formula-faithful; the structural Meta-State and fixed-window boundary blocks here are clean-room analogues. Exact expert paths can enter only through a SHA-verified historical-artifact adapter.

The implementation lives in `src/rogii_portfolio`. `make reproduce` runs the tests and the clean-room pipeline before rebuilding this notebook. The external Kaggle artifacts are listed separately, since they are not bundled here.
"""),
        code(r"""
implementation = pd.read_csv(ROOT / "evidence/historical_stage_scores.csv")
display(implementation)
print("Scored lineage:", " → ".join(lineage["scored_lineage"]))
print("Separate controls:", ", ".join(lineage["separate_research_controls"]))
fidelity = pd.DataFrame([
    ["Historical 121 features", "source-hash verified"],
    ["PF / no-prior GeoHMM", "formula-faithful research implementation"],
    ["HGRG / conditional shape", "formula-faithful overlay"],
    ["Scored incumbent", "external artifact required"],
    ["Structural Meta-State", "clean-room analogue"],
    ["Prefix-Boundary", "clean-room fixed-window analogue"],
], columns=["Block", "Status in this repository"])
display(fidelity)
"""),
        md(r"""
### Where the constants came from

This pipeline carries more hard-coded constants than I would choose if I were starting clean. Some are inherited from the historical baseline, some were fitted inside outer training folds, and the rest are movement caps I set on purpose. Keeping that provenance visible matters: a decimal that reproduces exactly is still just a decimal, and I did not want any of these read as measured physical quantities.

| Block | Important constants | Provenance | How I use it |
|---|---|---|---|
| Historical incumbent | legacy mixtures, SP45/beam, projection, contact/DTRT | external scored lineage | exact only with pinned artifact |
| Retrained clean parent | 0.30 Ridge + 0.70 PF; 0.00425 HGB cap | fixed clean-room heuristic | defines this validation parent |
| HGRG | beta 0.5; 250-ft ramp; 2.5/10-ft caps | frozen project policy | same-contract block result |
| Clean Meta-State | 0.65/0.25/0.10; 0.32 coefficient; 5/10-ft caps | clean-room analogue | no historical hidden attribution |
| Prefix-Boundary | tau/lookback 256; 2.5/10-ft caps | clean-room analogue | same-contract result |
| Conditional shape | 6.41825; 0.0090056; 0.25; 0.75/2.5-ft caps | frozen prediction-only policy | transfer evidence only |
| Expanded stack | learned convex weights | primary outer OOF | complementary components decide promotion |
"""),
        md(r"""
## 3. PF and GeoHMM: two views of the trajectory

The particle filter tracks structural position \(U=TVT+Z\) and incidence rate. For particle \(i\), a simplified update is

\[
U_t^{(i)} = U_{t-1}^{(i)} + r_t^{(i)}\Delta MD_t + \epsilon_t^{(i)},
\qquad
w_t^{(i)} \propto w_{t-1}^{(i)}
\exp\!\left[-\frac{(GR_t-g(TVT_t^{(i)}))^2}{2\sigma_{GR}^2}\right].
\]

GeoHMM approaches the same data differently: a 0.5-ft TVT grid, 25 slope states, robust GR emissions, and checkpointed forward and backward inference. Neither expert ever sees coordinates, neighbouring-well labels or suffix TVT.

The reason I keep both is that they fail in different places, so their disagreement carries information about where the alignment is genuinely ambiguous. That signal drives HGRG in the next section. It is worth being precise about what it is not, though: these two are **distinct but correlated**, not independent, so their spread is a disagreement measure and never a confidence interval.
"""),
        md(r"""
## 4. HGRG: a bounded PF/GeoHMM update

This is the part I designed specifically for this problem, so it is worth stating the reasoning before the equations. A wrong geological alignment is not a slightly worse prediction; it puts the well path in the wrong place entirely. The dangerous case is not an expert that is uncertain, it is an expert that is confidently wrong. So the useful question is not *which* expert to believe but *how much movement to authorise at all*, and that is a quantity I can estimate at prediction time from how much the experts disagree.

HGRG moves the parent coordinatewise toward the PF/GeoHMM bridge under nonnegative shrinkage and hard movement caps.

\[
Q=P+\beta(H-P),\qquad d=Q-B,
\]

\[
u=\frac{\operatorname{RMS}(H-P)}{\max(\operatorname{RMS}(d),\varepsilon)},
\qquad
p=\frac{\langle d,R-B\rangle}{\max(\langle d,d\rangle,\varepsilon)},
\]

\[
\rho=u\exp[-2\operatorname{clip}(p,-1,1)],\qquad
g=0.25+0.75\min(1,\rho^{-2}),
\]

\[
\widehat{TVT}=B+\operatorname{clip}\!\left(
\operatorname{clip}\!\left(\frac{h}{250},0,1\right)
g\min\!\left(0.5,\frac{2.5}{\max(\operatorname{RMS}(d),\varepsilon)}\right)d,
-10,10\right).
\]

The comparator to beat is a fixed PF/GeoHMM blend. What HGRG adds on top is normalized disagreement, a directional consensus check against Ridge, and regret caps. Its honest limitation is that all three experts can share the same misspecification, in which case they agree confidently on the wrong geology and the gate opens wide. The caps bound the damage; they cannot detect the cause.

Time and memory after PF/HMM are both \(O(N)\) per well.
"""),
        code(r"""
source = inspect.getsource(apply_hgrg)
print("Executable HGRG implementation (first 45 lines):")
print("\n".join(source.splitlines()[:45]))
"""),
        md(r"""
## 5. Reimplemented Meta-State

HGRG only ever looks at one well at a time. Meta-State is where regional geology enters, and the design problem is that regional information is exactly the kind that leaks if handled carelessly.

The split I settled on separates the two sides. An RBF surface is fitted only from outer-training wells, so the structural shape comes from geology the model is allowed to see. The target well then uses its own visible prefix to estimate a vertical offset and a rolling-origin error, so the datum is calibrated locally. It never exposes its suffix TVT or its formation label. The regional model therefore contributes trend and shape without ever setting the target's absolute level.

For PF, HMM and structural observations, I form a correlated covariance matrix \(\Sigma\). Negative generalized least-squares weights are clipped before normalization:

\[
\widetilde w=\Sigma^{-1}\mathbf 1,\qquad
w_i=\frac{\max(\widetilde w_i,0)}{\sum_j\max(\widetilde w_j,0)}.
\]

The fused observation enters a constant-acceleration state model with position, slope and curvature,

\[
x_{t+1}=
\begin{bmatrix}1&\Delta&\tfrac12\Delta^2\\0&1&\Delta\\0&0&1\end{bmatrix}x_t+\eta_t,
\]

followed by a Rauch-Tung-Striebel backward pass. A prediction-time PF/HMM-to-state dispersion ratio then scales the bounded move away from HGRG. The result is a state-space fusion with one spatial observation.

The closest baseline is independent inverse-variance averaging followed by a generic smoother; the difference is that \(\Sigma\) here states the expert correlations explicitly instead of pretending they are absent. Failure is most likely in sparse or extrapolative XY regions, where prefix rolling error downweights a biased formation surface but cannot repair it.

One caveat on lineage: this is not byte-identical to the scored block. The historical path also used a q25 anchor, a horizon ramp, a 3-ft pre-consensus projection and different structural transport rules, so I call it a clean-room analogue rather than a reimplementation.
"""),
        md(r"""
## 6. Small corrections at the prefix boundary

The last two stages are deliberately small, and deliberately separate. A single unconstrained residual model would happily mix three physically different errors: a wrong datum, a discontinuity where prediction takes over from observation, and local shape that is misaligned but roughly centred. Those want different treatments and different budgets, so each gets its own correction and its own cap.

For Prefix-Boundary, I extrapolate the last visible slope of \(U=TVT+Z\), then decay its correction with horizon:

\[
\Delta U(h)=\operatorname{clip}\left(e^{-h/256}[U_0+s h-(B(h)+Z(h))],-10,10\right).
\]

The move is projected to a 2.5-ft RMS budget. It buys continuity at the hand-off and fades out where a constant tangent stops being credible. It fails in the obvious case: a real dip change immediately after the prefix, where tangent continuity is precisely the wrong inductive bias. The clean-room code fixes a 256-ft lookback, while the historical branch selected among 64/128/256/512/1024 ft using visible-prefix rolling error, so the two are not the same block.

For the shape correction, let \(d=H_{stride6}-P\). I remove its well mean before projection and retain local shape:

\[
A=\operatorname{RMS}(d),\quad
S=\frac{\operatorname{RMS}(\Delta d)}{\max(A,\varepsilon)},\quad
g=\frac{1}{1+(A/6.41825)^2}\frac{S}{S+0.0090056}.
\]

The raw move is \(0.25g(d-\bar d)\), projected to 0.75-ft RMS and a 2.5-ft row cap. Removing the mean is what keeps this branch confined to shape: it can sharpen local alignment without shifting the well's datum, which is Meta-State's job and not this one's. Note the precise claim is *mean removal before projection*, not zero-mean output, because row clipping afterwards can move the mean again.
"""),
        md(r"""
## 7. The 121-feature branch and the nested stack

The readable clean-room builder predicts a residual from PF using exactly 121 local-observable features: geometry, GR derivatives/rolling windows, type-well residual banks, visible-prefix calibration and target-free PF/HMM paths. Unknown feature names fail closed. It excludes global spatial interpolation, dense neighbours, well identity and suffix target values.

Alongside it I keep the audited historical 121-feature body, which preserves PF600/ANCC600, seven beams, multi-scale NCC, stable per-well seeds and float32 output. The original notebook-cell SHA-256 pins its source identity, and a suffix-truth poison test confirms its output stays byte-invariant when the target is corrupted.

The two builders share a column schema and nothing else. Their formulas genuinely differ, so merging them or quietly presenting one as the other would misstate what was actually run. Schema parity is asserted; numerical parity is claimed only for deterministic reruns of the frozen historical body.

The stack is fitted from **inner-OOF** expert moves. Its weights satisfy \(w_k\ge0\) and \(\sum_k w_k\le1\); the remaining mass stays on the parent:

\[
\min_w \sum_i q_i\left(y_i-B_i-\sum_k w_k(E_{ik}-B_i)\right)^2+\lambda\lVert w\rVert_2^2.
\]

The row weights \(q_i\) balance geological components. I used this stack as a control; it is not an explanation of the historical hidden score.
"""),
        code(r"""
families = pd.Series([name.split("_")[0] if "_" in name else ''.join(filter(str.isalpha, name)) for name in SAFE_FEATURES]).value_counts()
print("Feature count:", len(SAFE_FEATURES))
print("Historical builder source SHA-256:", HISTORICAL_ORIGINAL_BUILDER_SHA256)
display(pd.DataFrame({"feature": SAFE_FEATURES}).head(20))
display(pd.DataFrame({"family_token": families.index, "count": families.values}).head(15))
"""),
        md(r"""
## 8. Nested validation, starting with a small CI check

Within each outer fold, the 121-feature estimators and the structural RBF are refitted from outer-training components only. Inner component OOF predictions fit the stack. That fold-frozen pipeline then predicts the outer validation components. Only once every outer prediction exists do I freeze one development stack, refit the base models on all development components, and open the run-level holdout, once.

PF/HMM paths and target-local features are safe to cache inside this split, since each depends on a single well's visible prefix and type well and learns nothing across wells. The model-facing `PreparedWell` dataclass has no truth field at all, official mode refuses to reconstruct a broken prefix from truth, and caching never crosses a fold boundary.

The synthetic run below is a wiring check and nothing more. Its holdout contains only two components, and several overlays make it worse there. I am leaving that in rather than hiding it, because it is exactly why the real validation moved to the full official-data universe.
"""),
        code(r"""
display(Image(filename=str(ROOT / "evidence/rmse_by_stage.png")))
display(Image(filename=str(ROOT / "evidence/outer_fold_gains.png")))
"""),
        md(r"""
### An 18-well integration check

I also run the same pipeline on 18 official raw training wells with deliberately reduced compute (2 PF seeds × 32 particles and stride-24 GeoHMM). This exercises CSV discovery, the natural visible-prefix contract, feature construction, nested fitting and structural inference against real log shapes rather than synthetic ones.

It is **not** a leaderboard estimate, and I would rather say so plainly than let a small number look meaningful. The subset is tiny, the holdout has four components, the compute budget differs from the full policy, and the similarity graph is built only within those 18 wells instead of over the complete competition universe.
"""),
        code(r"""
real_path = ROOT / "artifacts/realdata_smoke/evidence/reproduction_summary.json"
if real_path.exists():
    real = json.loads(real_path.read_text())
    display(pd.DataFrame({
        "Outer OOF": real["outer_oof_rmse"],
        "Subset holdout": real["untouched_holdout_rmse"],
    }).loc[["incumbent", "hgrg", "meta_state", "prefix_boundary", "sequential_final", "nested_stack"]])
    print("Integration-only holdout components:", real["holdout_components"])
else:
    print("Run `make realdata-smoke DATA_ROOT=...` to populate this optional check.")
"""),
        md(r"""
## 9. The 320-well experiment

This is the experiment the rest of the notebook builds toward. I first construct the similarity graph over all 320 official training wells, which at the frozen thresholds yields 243 connected components. I then sample **whole components** into a primary 160-well panel, leaving the other 160 wells as a component-disjoint confirmation panel. Both run the same 5 outer × 3 inner protocol with reduced PF/HMM compute.

Two panels that share no component is as close as this dataset lets me get to an honest generalization test: whatever I learn on one, the other has never contributed a single row to. The 243-component count does depend on the graph thresholds, so threshold sensitivity is reported below rather than left implicit.
"""),
        code(r"""
if primary160:
    stages = ["incumbent", "ridge", "nonlinear", "hgrg", "meta_state", "prefix_boundary", "sequential_final", "nested_stack"]
    primary_table = pd.DataFrame({
        "Primary outer OOF": primary160["outer_oof_rmse"],
        "Primary run holdout": primary160["untouched_holdout_rmse"],
    }).loc[stages]
    display(primary_table)
    print(
        f"Graph scope: {primary160['component_graph_scope_wells']} wells / "
        f"{primary160['component_graph_universe_components']} components; "
        f"evaluated: {primary160['wells']} wells / {primary160['components']} components"
    )
    display(pd.DataFrame(primary160["component_bootstrap"]).T)
    ax = primary_table.plot.bar(figsize=(11, 5), rot=30)
    ax.set_ylabel("RMSE (ft; lower is better)")
    ax.set_title("Primary160: every stage on the same rows and folds")
    ax.grid(axis="y", alpha=.25)
    plt.tight_layout(); plt.show()
else:
    print("Run `make realdata-nested-160 DATA_ROOT=...` to populate the primary panel.")

if graph_audit:
    display(pd.DataFrame([graph_audit]).T.rename(columns={0: "value"}))
    sensitivity_path = ROOT / "artifacts/realdata_nested_160/graph_audit/component_graph_sensitivity.csv"
    if sensitivity_path.exists():
        display(pd.read_csv(sensitivity_path))
"""),
        md(r"""
### What I learned from the primary panel

The sequential physics path improved on the clean incumbent on both primary outer OOF and the run-level holdout. Ridge improved on it by considerably more, which is the sort of result that is tempting to accept and act on.

So I fitted an expanded convex policy on primary outer OOF alone and froze it, with the complementary panel still untouched at that point. That ordering is the whole point of the exercise: the policy had to be committed before the panel that would judge it could influence it. The fitted primary number below is therefore a selection statistic, not a performance estimate, and I report it as one.
"""),
        code(r"""
policy_path = ROOT / "artifacts/realdata_nested_160/frozen_primary_policy.json"
if policy_path.exists():
    frozen_policy = json.loads(policy_path.read_text())
    display(pd.DataFrame({
        "arm": ["incumbent", *frozen_policy["arms"]],
        "weight": [frozen_policy["parent_weight"], *frozen_policy["weights"]],
    }))
    print(frozen_policy["claim_note"])

if frozen_confirmation:
    rows = []
    for contract_name, result in (
        ("Complement160 outer OOF", frozen_confirmation["confirmation_outer_oof"]),
        ("Complement160 run holdout", frozen_confirmation["confirmation_run_holdout"]),
    ):
        for model, score in result["rmse"].items():
            rows.append({"contract": contract_name, "model": model, "rmse_ft": score})
    confirmation_table = pd.DataFrame(rows)
    confirmation_pivot = confirmation_table.pivot(index="model", columns="contract", values="rmse_ft")
    display(confirmation_pivot)
    ax = confirmation_pivot.plot.bar(figsize=(10, 5), rot=25)
    ax.set_ylabel("RMSE (ft; lower is better)")
    ax.set_title("Complement160: the Ridge-heavy frozen policy does not transfer")
    ax.grid(axis="y", alpha=.25)
    plt.tight_layout(); plt.show()
    display(pd.DataFrame({
        "expanded OOF bootstrap": frozen_confirmation["confirmation_outer_oof"]["expanded_component_bootstrap"],
        "ridge OOF bootstrap": frozen_confirmation["confirmation_outer_oof"]["ridge_component_bootstrap"],
    }).T)
    endpoint = frozen_confirmation["confirmation_outer_oof"]
    if endpoint["gain_vs_incumbent"]["expanded_frozen"] <= 0:
        print("I did not keep the Ridge-heavy policy: it failed on complementary-component OOF.")
else:
    print("The complementary 160-well confirmation has not been materialized yet.")
"""),
        md(r"""
### Post-hoc robustness check on the reused panels

Having now inspected both 160-well panels, I reused their four saved prediction
contracts for a group-robustness audit. That makes this section a post-hoc
diagnostic rather than another confirmation, and it is labelled that way for the
rest of the notebook. The registry fixes the model columns, horizon bins,
incumbent comparator, worst-10% component tail and 2,000 whole-component
bootstrap repeats per contract, so the aggregation choices were not made after
seeing which ones flattered the result.

The audit found sequential final improving on the incumbent in all ten outer
folds. Ridge won more individual folds, but with a far wider rank spread and one
severe complementary-component failure.

That contrast is the reason I report several grains side by side. Pooled row
RMSE, a typical component, and the worst component can each tell a different
story about the same predictions, and a promotion rule that consults only one of
them will eventually promote something like Ridge.

One limit remains that no amount of resampling fixes: every saved real-data
prediction came from training seed `20260806`. Bootstrap seeds vary which
components are sampled. They say nothing about what happens if the model is
refit from scratch. Claiming training-seed stability would need several new
registered refits, which I have not run.
"""),
        code(r"""
if robustness_summary is not None and robustness_metrics is not None and robustness_ranks is not None:
    selected = robustness_metrics.loc[
        robustness_metrics["model"].isin(["incumbent", "ridge", "sequential_final"]),
        [
            "contract_id", "model", "pooled_row_rmse", "macro_well_rmse",
            "macro_component_rmse", "harmed_well_rate",
            "worst_10pct_component_rmse_cvar",
        ],
    ].copy()
    selected["harmed_well_rate"] = selected["harmed_well_rate"].map(lambda value: f"{value:.1%}")
    display(selected)

    fold_stability = robustness_ranks.loc[
        (robustness_ranks["scope"] == "outer_fold")
        & (robustness_ranks["contract_id"] == "all_oof_folds")
        & robustness_ranks["model"].isin(["incumbent", "ridge", "sequential_final"]),
        [
            "model", "units", "mean_rank", "rank_std", "best_rate",
            "beats_baseline_rate", "worst_unit_gain_vs_baseline",
        ],
    ].sort_values("mean_rank")
    display(fold_stability)
    print(
        f"Quality checks: {robustness_summary['quality_checks']['passed']} passed / "
        f"{robustness_summary['quality_checks']['failed']} failed"
    )
    print(robustness_summary["training_seed_coverage"]["note"])
else:
    print("Run `make robustness-evaluation` after materializing both 160-well panels.")
"""),
        md(r"""
### Two model changes after the robustness audit

I tried one ambitious model and one deliberately modest correction, which turned out to be a useful pairing.

The ambitious one was a four-mode interacting state model over smooth, upward-switch, downward-switch and uncertain regimes. It was fully target-free and passed the synthetic step and disagreement controls, and it was still worse than the plain sequential path on both Primary160 OOF and its run-level holdout. The tempting move at that point is to tune it against the same panel until it wins. I left it default-off instead, because a model that only works after being fitted to its own test is not a model that works.

The modest one treats Ridge as a high-variance direction and strictly limits how much of it can get through:

\[
\widehat y_w=(1-w)\widehat y_{seq}+w\widehat y_{ridge},
\qquad 0\leq w\leq w_{max}.
\]

Repeated component folds on Complement OOF selected a 5% cap. Frozen at 5%, it improved row RMSE in all four reused contracts, though one component-bootstrap interval crossed zero and one tail metric regressed slightly.

This is a genuinely promising result, and it still does not get promoted. The 5% was found after I had already inspected both panels, so I cannot distinguish a real effect from a well-chosen number. It ships as an opt-in flag with the caveat attached rather than as a default.
"""),
        code(r"""
if switching_pilot is not None:
    display(pd.DataFrame([
        {
            "split": "Primary outer OOF",
            "sequential": switching_pilot["outer_oof_rmse"]["sequential_final"],
            "switching_state": switching_pilot["outer_oof_rmse"]["switching_state"],
        },
        {
            "split": "Primary run holdout",
            "sequential": switching_pilot["untouched_holdout_rmse"]["sequential_final"],
            "switching_state": switching_pilot["untouched_holdout_rmse"]["switching_state"],
        },
    ]))

if trust_region is not None:
    audit = trust_region["fixed_weight_0p05_audit"]
    labels = {
        "primary_outer_oof": "Primary OOF",
        "primary_holdout": "Primary holdout",
        "complement_outer_oof": "Complement OOF",
        "complement_holdout": "Complement holdout",
    }
    rows = []
    for key, label in labels.items():
        item = audit[key]
        rows.append({
            "contract": label,
            "sequential_rmse": item["sequential"]["row_rmse"],
            "trust_region_rmse": item["candidate"]["row_rmse"],
            "gain_ft": item["row_rmse_gain_ft"],
            "ci95_low": item["bootstrap"]["ci95_low"],
            "ci95_high": item["bootstrap"]["ci95_high"],
            "cvar_gain_ft": item["cvar10_gain_ft"],
        })
    trust_table = pd.DataFrame(rows)
    display(trust_table)
    lower = trust_table["gain_ft"] - trust_table["ci95_low"]
    upper = trust_table["ci95_high"] - trust_table["gain_ft"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(trust_table["contract"], trust_table["gain_ft"], color="#4C78A8", alpha=.85)
    ax.errorbar(
        np.arange(len(trust_table)), trust_table["gain_ft"],
        yerr=np.vstack([lower, upper]), fmt="none", ecolor="black", capsize=4,
    )
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("RMSE gain over sequential (ft)")
    ax.set_title("5% Ridge trust region: whole-component bootstrap")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=.2)
    plt.tight_layout()
    plt.show()
else:
    print("Run the post-hoc trust-region experiment to materialize this audit.")
"""),
        code(r"""
folds = pd.read_csv(ROOT / "evidence/outer_fold_metrics.csv")
display(folds)
stack = summary["final_stack"]
display(pd.DataFrame({"weight": {"parent": stack["parent_weight"], **stack["weights"]}}))
"""),
        md(r"""
## 10. Pairwise regret and a conformal-style abstention rule

If the experts disagree in an informative way, it should be possible to predict *in advance* which one will be right for a given well. That is worth testing directly, so I froze this control before the Primary160 run finished.

It predicts each expert's per-well MSE gain from target-free disagreement, movement, roughness and horizon summaries, fitted with component cross-fitting. I then subtract a group-weighted absolute-residual quantile and keep the incumbent unless some expert's lower confidence score is positive, which makes abstention the default rather than the exception:

\[
LCB_k(x)=\widehat{G}_k(x)-Q_q\left(\left|G_k-\widehat{G}^{OOF}_k\right|\right),
\qquad
k^*=\arg\max_k LCB_k(x).
\]

To be clear about what this is: an abstention rule over expert regret, not a calibrated TVT prediction interval. The only question it answers is whether moving away from the incumbent is justified for this particular well. The answer, as the results below show, was usually no.
"""),
        code(r"""
if regret_router:
    display(pd.DataFrame({
        "metric": [
            "development incumbent", "development routed", "holdout incumbent",
            "holdout routed", "holdout abstention rate"
        ],
        "value": [
            regret_router["development_fallback_rmse"],
            regret_router["development_selected_rmse"],
            regret_router["holdout_fallback_rmse"],
            regret_router["holdout_selected_rmse"],
            regret_router["holdout_abstention_rate"],
        ],
    }))
    display(pd.DataFrame([regret_router["holdout_component_bootstrap"]]))
    print("Selected arms:", regret_router["holdout_selected_arm_counts"])
    print("The uncertainty signal was useful, but the simple sequential expert was still slightly better.")
else:
    print("Run the predeclared regret-router script after Primary160.")
"""),
        md(r"""
### A small Meta-State ablation

Across the larger panels HGRG looked like the most transferable overlay, while Meta-State kept helping outer OOF and hurting a run-level holdout. That pattern is worth taking seriously, so rather than deleting the stage outright I audited a transparent family that contains both options

\[
\widehat y(\alpha,\gamma)=HGRG+\alpha(Meta-HGRG)+\gamma(Sequential-Meta),
\]

with \(\alpha\in\{0,.25,.5,.75,1\}\) and \(\gamma\in\{.5,.75,1\}\), selected on primary outer OOF. Setting \(\alpha=0\) removes Meta-State entirely, so the ablation is a point inside the family rather than a separate experiment.

I designed this check after seeing the complementary panel's stage scores, which makes it post-hoc no matter how clean the family looks. It would need a fresh component panel before I would change the main pipeline on the strength of it.
"""),
        code(r"""
if meta_shrinkage:
    rows = []
    for name in ("primary_outer_oof", "primary_run_holdout", "confirmation_outer_oof", "confirmation_run_holdout"):
        result = meta_shrinkage[name]
        rows.append({
            "contract": name,
            "sequential_rmse": result["sequential_rmse"],
            "candidate_rmse": result["candidate_rmse"],
            "gain_ft": result["gain_vs_sequential_ft"],
            "ci95_low": result["component_bootstrap"]["ci95_low"],
            "ci95_high": result["component_bootstrap"]["ci95_high"],
        })
    display(pd.DataFrame(rows))
    print("Status:", meta_shrinkage["status"], "|", meta_shrinkage["claim_note"])
"""),
        md(r"""
## 11. Remaining sources of bias

Bootstrapping complete connected components instead of rows fixes one thing and leaves others untouched. The resulting interval describes variation across sampled geological units for a policy that has *already been selected*. None of the uncertainty I introduced by trying many ideas is in that number.

The component split handles neighbour and similarity leakage. It does nothing about the three selection problems I actually ran into. A transfer panel stops being a transfer panel once I have looked at it enough times. A long series of PF variants can look like thorough search while really being one mechanism explored repeatedly. And a small, cheap runtime canary says very little about the expensive non-overlap wells that dominate the hidden workload.

Next time I would keep the component graph, set an explicit trial budget per model family before starting, hold one component panel genuinely in reserve, and profile runtime on long non-overlap wells before launching anything scored.
"""),
        code(r"""
boot = summary["component_bootstrap"]
display(pd.DataFrame(boot).T)
display(Image(filename=str(ROOT / "evidence/example_trajectory.png")))
"""),
        md(r"""
## 12. A tuning experiment that did not transfer

I also ran a fully nested optimizer, epoch and decoder sweep. Individual folds picked out high regularization, an adaptive learning rate, modified Huber loss, label smoothing, structured transitions, 300 epochs and a softened decoder, all of which sounds like a model that learned something.

It was not. The tuned candidate worsened 9.4531 to 9.4680 on its own Dev160 contract, improved only 2 of 5 folds, harmed 85 of 160 wells, and produced a component-bootstrap interval crossing zero. Having failed all four checks, I stopped there rather than opening the complementary labels to look for a better story.

The lesson was not that tuning is useless. It was that the fold-specific optima varied more than the benefit of averaging them, so the sweep was mostly measuring fold noise. No number of epochs fixes instability in which expert to trust.
"""),
        md(r"""
## 13. Reproduction status

| Reproduction check | Status |
|---|---|
| Clean-room algorithms and nested validation graph execute from source | **Yes: workflow + tests** |
| Historical 121 SAFE formulas execute from vendored source | **Yes: source hash + poison test** |
| Inference inputs → exact-order submission CSV graph executes | **Yes: deployment smoke** |
| Historical scored incumbent / Meta-State / Boundary are all byte-identical | **No** |
| One command rebuilds tracked smoke evidence, figures and notebook | **Yes** |
| Official raw data can replace the synthetic smoke data | **Yes, with `DATA_ROOT`** |
| Historical 9.091 `submission.csv` bytes reproduce without external artifacts | **No** |

Exact reproduction still depends on the third-party Kaggle feature and pretrained-model packages the historical parent loaded. They are listed in `configs/exact_artifacts.json`, and exact mode checks the real local files and their hashes rather than taking my word for it. Retraining the reimplemented pipeline is a useful thing to be able to do, but it answers a different question, and the table above keeps the two apart.
"""),
        code(r"""
if deployment_audit:
    display(pd.DataFrame([{
        "status": deployment_audit["status"],
        "rows": deployment_audit["rows"],
        "truth in targets": deployment_audit["targets_contained_tvt"],
        "CSV SHA-256": deployment_audit["csv_sha256"],
    }]))
"""),
        md(r"""
## 14. Looking back

The fixed final pipeline returned 6.536 on the visible partition and 9.091 on the hidden partition, inside the displayed bronze-score band. Without matched hidden ablations that is one observation of the whole path, not evidence for any individual overlay, and I have tried to resist reading more into it than that.

The run also completed after the cutoff, which made it ineligible. I had verified numerical correctness on a protected canary and never verified runtime on a workload that resembled the real one. That is the mistake I think about most, because it was not a modelling error at all. In a code competition the queue and the hidden execution time are part of the method, and I had left them out of the experimental plan entirely.

The project handed me a mathematics reading list along the way, each item arriving because something refused to work without it. The particle filter led back to Bayesian filtering, GeoHMM to dynamic programming, Meta-State to generalized least squares and RTS smoothing, and the leakage problem to graph theory. Signal processing and constrained optimization turned up every time I needed to separate local variation from long-range trend without letting a well drift too far. Filtering and constrained optimization are where I want to go deeper next.
"""),
        code(r"""
required = [
    "particle.py", "geohmm.py", "features.py", "historical_features.py",
    "hgrg.py", "meta_state.py", "overlays.py", "stack.py", "pipeline.py",
    "deployment.py", "parents.py", "regret_router.py", "evaluation.py"
]
source_root = ROOT / "src/rogii_portfolio"
audit = pd.DataFrame([
    {"module": name, "exists": (source_root / name).exists(), "bytes": (source_root / name).stat().st_size}
    for name in required
])
display(audit)
assert audit["exists"].all()
assert len(SAFE_FEATURES) == 121
assert summary["target_isolation"]["component_overlap_count"] == 0
assert summary["target_isolation"]["suffix_truth_in_prediction_api"] is False
assert deployment_audit is None or deployment_audit["targets_contained_tvt"] is False
print("Notebook checks passed.")
"""),
    ]
    output = ROOT / "portfolio_notebook.ipynb"
    nbf.write(nb, output)
    print(output)
    return output


if __name__ == "__main__":
    build()
