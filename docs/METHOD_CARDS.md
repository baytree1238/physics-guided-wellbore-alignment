# Method cards

One card per method. Each states what the method is, what it consumes and
returns, the simplest thing it has to beat, its cost, and the condition under
which it is expected to fail.

The last field is the one that matters most. A method whose failure mode I
cannot name is a method I do not understand well enough to bound, and every
block in this pipeline is allowed to move the prediction only within a budget.
Writing the failure condition down first is what makes that budget a decision
rather than a guess. Where a card reports an observed result, the status tag
(**HOLD**, opt-in, default path) records what the evidence actually licensed.

## Particle filter

- **Definition:** a sequential Monte Carlo tracker over structural position
  $U=TVT+Z$ and incidence rate, updated by type-well GR likelihood.
- **Input / output:** one target well's `MD,Z,GR,TVT_input` and its type-well
  `TVT,GR`; one suffix TVT path per seed plus predictive log evidence.
- **Closest baseline:** deterministic continuation of the last visible slope.
- **Complexity:** $O(SNP)$ for $S$ seeds, $N$ suffix rows and $P$
  particles; memory $O(P+SN)$.
- **Failure condition:** repeated GR motifs produce multimodal alignments and a
  globally dominant Monte Carlo seed can lock onto the wrong mode.

## GeoHMM

- **Definition:** a target-isolated grid smoother over TVT and local slope with
  robust GR emissions and checkpointed forward/backward inference.
- **Input / output:** the same target-local columns as PF; posterior mean TVT
  and uncertainty on the suffix.
- **Closest baseline:** a one-dimensional dynamic time-warping alignment.
- **Complexity:** $O((N/s)KV)$, where $s$ is stride, $K=25$ slope states,
  and $V$ is the 0.5-ft TVT grid; checkpointing reduces stored forward states.
- **Failure condition:** weak or periodic GR can support several equally
  plausible state paths; a hard replacement is unsafe.

## HGRG: Hierarchical Geology Regret Gate

- **One-sentence definition:** HGRG moves a protected parent coordinatewise
  toward the PF/GeoHMM bridge only when prediction-time dispersion and an
  auxiliary Ridge direction indicate tolerable regret.
- **Input / output:** parent, Ridge, PF, GeoHMM and horizon arrays; a bounded
  candidate and per-well gate diagnostics.
- **Closest baseline:** a fixed global PF/GeoHMM blend.
- **Gate rule:** disagreement is normalized by the requested move, consensus
  enters monotonically, and both RMS and row movement are bounded.
- **Complexity:** $O(N)$ time and memory per well after PF/HMM inference.
- **Failure condition:** all experts may agree on the same wrong geological
  mode; the gate limits movement but cannot detect shared misspecification.

Pseudocode:

```text
Q <- PF + beta * (HMM - PF)
d <- Q - parent
u <- RMS(HMM - PF) / max(RMS(d), epsilon)
p <- dot(d, Ridge - parent) / max(dot(d,d), epsilon)
risk <- u * exp(-2 * clip(p,-1,1))
gate <- 0.25 + 0.75 * min(1, risk^-2)
a <- gate * min(0.5, 2.5 / max(RMS(d), epsilon))
prediction <- parent + clip(clip(horizon/250,0,1) * a * d, -10, 10)
```

## Meta-State

- **Definition:** Meta-State combines PF, GeoHMM, and an outer-training-only
  RBF observation with correlated GLS. A constant-acceleration
  Rauch-Tung-Striebel smoother estimates position, slope, and curvature before
  the result is applied as a bounded overlay.
- **Input / output:** PF, GeoHMM, cross-fitted structural path, prefix CV error,
  HGRG parent and horizon; smoothed state plus bounded candidate.
- **Closest baseline:** independent inverse-variance averaging followed by a
  Savitzky-Golay smoother.
- **Regional fusion:** training-side structural geometry is separated from
  target-side prefix calibration. Correlated experts receive nonnegative GLS
  weights, and deployment confidence controls the move.
- **Complexity:** RBF fit is implementation-dependent (local neighbours cap it
  in practice); GLS is constant-size; RTS is $O(N/s)$ time and memory.
- **Failure condition:** sparse extrapolative XY regions or biased formation
  labels can corrupt the structural expert. Prefix rolling error downweights
  but cannot fully repair this.

Pseudocode:

```text
fit RBF(X,Y -> structural level) on outer-training wells only
calibrate target RBF level from the visible prefix
Sigma <- correlated covariance(PF, HMM, structural; prefix rolling error)
w <- clip_nonnegative(solve(Sigma, 1)); normalize(w)
z <- w_pf*PF + w_hmm*HMM + w_struct*structural
state <- constant-acceleration Kalman filter + RTS backward pass(z)
reliability <- bounded function of RMS(PF-HMM) / RMS(state-HGRG)
prediction <- radial_project(HGRG + reliability*(state-HGRG), 5-ft RMS, 10-ft row)
```

The historical Meta-State block also used a q25 anchor, a horizon ramp, a 3-ft
pre-consensus projection, and a different structural-source/transport
contract.

Two correctness variants are implemented but default-off. `constrained_weights`
solves the exact nonnegative simplex GLS problem instead of clipping a negative
unconstrained solution, and `exact_projection` computes the Euclidean
projection onto the joint RMS/row-cap set instead of preserving the proposed
move direction. They require nested empirical validation before promotion.

## Switching-state control

- **Definition:** a four-mode IMM/GPB2-style approximation over offset, rate
  and curvature, with smooth, upward-switch, downward-switch and uncertain
  regimes inferred from HGRG-relative PF/HMM/structural residuals.
- **Input / output:** target-free expert paths, horizon and prefix variability;
  one bounded trajectory plus regime probabilities.
- **Closest baseline:** the single-regime constant-acceleration RTS smoother.
- **Complexity:** linear in the number of stride checkpoints with four small
  state filters; about 0.29 seconds for a synthetic 1,800-row well on the audit
  CPU.
- **Failure condition:** GR outliers can resemble a changepoint and HGRG may
  absorb a jump before the regime model sees it. The probabilities are not
  calibrated geological-fault probabilities.
- **Observed result:** Primary160 OOF 15.2866 versus 14.8993 sequential, and
  holdout 14.6019 versus 13.9504. **HOLD**, excluded from the learned stack.

## Capped Ridge trust region

- **Definition:** retain the sequential trajectory as parent and permit only
  a small move toward Ridge,
  $\widehat y=(1-w)\widehat y_{seq}+w\widehat y_{ridge}$, with
  $0\leq w\leq w_{max}$.
- **Selection:** component folds minimize row error with macro-component,
  worst-component and $L_2$ guardrails; identity is an explicit abstention
  candidate.
- **Closest baseline:** an unrestricted convex residual stack.
- **Complexity:** $O(N+GJ)$ after the two expert paths exist, for $G$
  components and a small one-dimensional grid of $J$ weights.
- **Failure condition:** a shared regional bias can make even a small Ridge move
  harmful; one reused contract showed a 0.0731-ft CVaR regression.
- **Observed result:** a post-hoc 5% sensitivity improved row RMSE in all four
  reused contracts by 0.1344 to 0.3803 ft, but one bootstrap interval crossed
  zero. Status: opt-in research candidate.

## Prefix-Boundary

- **Definition:** extrapolate the last visible slope of $U=TVT+Z$, then apply
  an exponentially decaying, projected correction near the prefix boundary.
- **Closest baseline:** no boundary continuity correction.
- **Complexity:** $O(N)$.
- **Failure condition:** an abrupt real dip change immediately after the prefix
  makes tangent continuity the wrong inductive bias.

The executable clean-room version fixes a 256-ft lookback. The historical
scored branch selected 64/128/256/512/1024 ft using visible-prefix rolling
error, so the two implementations must not be called identical.

## Conditional GeoHMM shape

- **Definition:** remove the within-well mean from stride-6 GeoHMM minus PF,
  retain local shape only, gate it by amplitude and slope information, and
  project to 0.75-ft RMS / 2.5-ft row limits.
- **Closest baseline:** a fixed 20% to 25% HMM level blend.
- **Complexity:** $O(N)$ after the HMM path exists.
- **Failure condition:** row clipping can alter the mean after it was removed;
  the correct claim is “mean-removed before projection,” not “exactly
  zero-mean after every safety operation.”

## Nested residual stack (research control)

- **Definition:** nonnegative SLSQP weights with sum at most one and L2 shrink
  to the parent, learned only from inner-OOF expert moves.
- **Closest baseline:** equal averaging.
- **Complexity:** $O(NK^2)$ sufficient statistics plus a small constrained
  optimization for $K=4$ arms.
- **Failure condition:** highly correlated arms make weights unstable. A gain
  over the identity parent is not enough; it must also beat the best simple
  expert by enough to justify complexity.

## Pairwise-regret + conformal abstention (research control)

- **Definition:** predict each expert's MSE gain over the incumbent from
  well-level, target-free disagreement/movement/horizon summaries; subtract a
  component-cross-fitted absolute-residual quantile and abstain unless the best
  lower confidence score is positive.
- **Input / output:** OOF expert trajectories grouped by well and component;
  one selected expert (or incumbent) and a 25/50/100% bounded blend per well.
- **Closest baseline:** one global expert/blend for every well.
- **Complexity:** $O(WKF)$ small linear fits after expert inference, for $W$
  wells, $K$ candidates and $F$ component folds.
- **Failure condition:** uncertainty features may not identify shared expert
  misspecification; correlated candidates also make predicted regret noisy.
  The conformal margin is a group-weighted empirical guard, not a calibrated
  TVT prediction interval.

## Parameter settings

This pipeline carries more hard-coded constants than I would choose in a clean
design. Some are inherited from the historical baseline, some were fitted
inside outer training folds, and the rest are movement caps I set deliberately.
The table separates those three cases, so that a reproducible decimal is never
mistaken for a measured physical quantity.

| Block | Important constants | Source | Status |
|---|---|---|---|
| Historical incumbent | legacy mixture/projection/contact parameters | External scored lineage | Requires pinned parent artifact |
| Retrained clean parent | 0.30 Ridge + 0.70 PF; 0.00425 HGB cap | Fixed clean-room policy | Validation parent |
| HGRG | beta 0.5; 250-ft ramp; 2.5-ft RMS; 10-ft row | Frozen project policy | Default path |
| Clean Meta-State | 0.65/0.25/0.10; 0.32 coefficient cap; 5/10-ft projection | Fixed clean-room analogue | Default path |
| Prefix-Boundary | tau 256; fixed 256-ft lookback; 2.5/10-ft projection | Fixed clean-room analogue | Default path |
| Conditional shape | 6.41825; 0.0090056; alpha 0.25; 0.75/2.5-ft projection | Frozen prediction-only policy | Default path |
| Residual stack | fold-fitted nonnegative weights | Inner component OOF | Research control |
| Regret router | q in {0.50,0.75,0.90}; blend in {0.25,0.50,1.00} | Outer development OOF selection | Disabled |
| Switching state | four regimes; 5/10-ft movement caps | Synthetic controls + post-hoc Primary160 pilot | HOLD |
| Ridge trust region | 5% diagnostic weight; 25% search cap | Repeated component folds on reused panels | Opt-in |
