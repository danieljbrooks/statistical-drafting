# Learning draft policies by rollout-based policy iteration

The pieces for reinforcement learning on draft already exist in this repo. `DraftNet`
maps `(collection, pack)` to a distribution over cards in the pack — that is a policy.
`WinrateNet` maps a finished 42-card pool to a predicted per-match win probability — that
is a reward. What is missing is the loop that uses the second to improve the first, and
an environment to run it in.

## Name

**Approximate policy iteration with Monte-Carlo rollouts**, in the form usually called
**Expert Iteration** (AlphaZero's structure, with rollouts instead of tree search). The
update rule is **Advantage-Weighted Regression**, which is the same object as the E-step
of MPO, and is derived below as **KL-regularized policy improvement**.

One phrase: *KL-regularized policy improvement with Monte-Carlo advantage estimation,
projected back onto the network by weighted behaviour cloning.*

Lineage: Howard (1960) policy iteration · Tesauro & Galperin (1996) rollout improvement ·
Peters & Schaal (2007) reward-weighted regression · Abdolmaleki et al. (2018) MPO ·
Anthony, Tian & Barber (2017) expert iteration · Peng et al. (2019) AWR.

## Setup

A state `s = (pool, pack)` is one decision: the cards taken so far and the cards on
offer. An action `a` is a card in the pack. Reward is terminal only — there is no
intermediate signal, just the winrate model applied to the finished pool:

$$R = \sigma\big(\text{WinrateNet}(\text{pool}_{42})\big)$$

Write $\pi_\theta(a \mid s)$ for the policy. Then

$$Q^\pi(s,a) = \mathbb{E}\big[R \mid \text{take } a \text{ at } s, \text{ then follow } \pi\big]$$

$$V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s,a), \qquad A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

The expectation is over the environment: which packs are opened, and what the other
seven seats take.

## The improvement step

We want a better policy, but two things argue against taking a large step. $\hat{Q}$ is
estimated from finitely many rollouts, and the reward model is only trustworthy near the
pools it was fit to. So constrain the step:

$$\max_{\pi'} \ \sum_a \pi'(a) A^\pi(s,a) \quad \text{subject to} \quad \mathrm{KL}\big(\pi'(\cdot \mid s) \,\|\, \pi(\cdot \mid s)\big) \le \epsilon$$

With $\beta$ the multiplier on the KL and $\lambda$ on normalization:

$$\mathcal{L} = \sum_a \pi'(a) A(s,a) - \beta \sum_a \pi'(a) \log \frac{\pi'(a)}{\pi(a)} + \lambda\Big(\sum_a \pi'(a) - 1\Big)$$

$$\frac{\partial \mathcal{L}}{\partial \pi'(a)} = A(s,a) - \beta\left(\log \frac{\pi'(a)}{\pi(a)} + 1\right) + \lambda = 0$$

$$\log \frac{\pi'(a)}{\pi(a)} = \frac{A(s,a)}{\beta} + \frac{\lambda}{\beta} - 1$$

Normalizing away the constant gives the closed form:

$$\pi^*(a \mid s) = \frac{\pi(a \mid s)\, \exp\big(A^\pi(s,a)/\beta\big)}{\sum_{a'} \pi(a' \mid s)\, \exp\big(A^\pi(s,a')/\beta\big)}$$

This is the exact optimum of the constrained problem — no approximation yet. $\beta$ is
the trust region: small $\beta$ is aggressive, large $\beta$ barely moves.

## The projection step

$\pi^*$ is nonparametric: a reweighting of $\pi$ defined state by state. To get it back
into the network, minimize $\mathrm{KL}(\pi^* \,\|\, \pi_\theta)$, which up to a constant
in $\theta$ is

$$\mathcal{L}(\theta) = - \mathbb{E}_s \sum_a \pi^*(a \mid s) \log \pi_\theta(a \mid s)$$

Cross-entropy against a soft target. This is already the loss in
`statisticaldrafting/train.py`, which calls `CrossEntropyLoss` against `pick_vector`, a
probability vector rather than a class index. Swapping the one-hot human pick for
$\pi^*$ turns supervised imitation into reinforcement learning; nothing else in the
training code has to change.

## The estimator

For each sampled state $s$:

1. Enumerate every legal action. A booster holds 14 cards, so this is at most 14 — small
   enough that there is no reason to sample actions.
2. For each action, take it and roll the rest of the draft out to completion under the
   current policy. Score the resulting pool.
3. $\hat{Q}(s,a)$ is the mean over rollouts; $\hat{V}(s) = \sum_a \pi_\theta(a \mid s) \hat{Q}(s,a)$
   using the enumerated $\hat{Q}$; $\hat{A} = \hat{Q} - \hat{V}$.
4. Weight, normalize, and fit.

**Common random numbers are what make this affordable.** Advantage gaps between candidate
picks are on the order of $10^{-3}$ in winrate. Rollout-to-rollout variance from pack
randomness is around $5 \times 10^{-2}$ — fifty times larger. But only *differences*
$\hat{Q}(s,a) - \hat{Q}(s,b)$ enter $\hat{A}$, so if every action's rollout uses the same
future packs and the same opponent randomness, almost all of that variance cancels. In
this implementation the pack schedule is dealt up front and shared across the forked
worlds, and opponent sampling uses tiled Gumbel noise, so the *only* difference between
the worlds being compared is the deviation under test.

## Why not GRPO

GRPO samples $G$ completions per prompt and sets $A_i = (r_i - \text{mean})/\text{std}$,
then takes a clipped policy-gradient step. Three differences here, all favourable:

- **Enumeration beats sampling.** The action space is at most 14, so every action's $Q$ is
  available. There is no reason to accept $G$ noisy samples of one action instead.
- **No importance ratio, no clipping.** Because $\pi^*$ is a closed-form target rather
  than a gradient estimate, the off-policy correction and its clipping heuristic are not
  needed. Enumerating actions uniformly and *then* applying a policy-gradient ratio would
  in fact be wrong — the samples did not come from $\pi$.
- **Fixed $\beta$, not per-state std.** Dividing by the per-group standard deviation is
  known to be biased, and draft is a bad case for it: late picks have genuinely zero
  advantage spread, and normalizing by their std turns pure noise into a confident
  gradient.

## Three properties worth stating

**Monotone improvement.** As $\beta \to 0$ this becomes greedy policy iteration, where
Howard's policy improvement theorem gives $V^{\pi'} \ge V^\pi$ pointwise. Finite $\beta$
trades improvement rate for safety against estimation error.

**Graceful degradation — but only with an absolute $\beta$.** If every advantage at a state
is equal — pure noise, no signal — then $\exp(A/\beta)$ is constant, $\pi^* = \pi$, and the
policy does not move. Given how small real advantage gaps are, this matters more than the
improvement guarantee.

The catch is that it depends on how $\beta$ is set. Setting $\beta = c \cdot \mathrm{sd}(\hat{A})$
is convenient and scale-free, but it makes the step a fixed distance in advantage standard
deviations: the noise is divided back out, and the update is exactly as confident on a
weak reward as on a strong one. That is the per-group-std problem this document criticises
GRPO for, one level up. A $\beta$ fixed in winrate units (``--beta-absolute``) keeps the
property — when the reward weakens, $\hat{A}$ shrinks, $\hat{A}/\beta$ shrinks with it, and
the update collapses toward a no-op on its own.

**No critic.** The baseline $V^\pi(s)$ is computed exactly from the same enumerated
rollouts. There is no learned value network, no TD bootstrapping, and therefore no
staleness as the policy moves.

That last point is worth dwelling on, because this repo *has* a partial-pool value
model and it is tempting to use it. It should not be used as a bootstrapping target. It
was fit to the winrates that pools reached when **human drafters** finished them, so it
estimates $V^{\pi_{\text{human}}}$, not $V^*$ — a value function for the behaviour policy,
which becomes wrong as soon as the policy improves. It also carries a selection effect:
a strong-looking prefix correlates with an open lane and a lucky seat, not only with good
decisions. Comparing actions within a single state cancels both problems, which is the
reason the advantage is estimated the way it is above.

## Two anchors, and why both are needed

$\beta$ and the reference penalty $\lambda$ enter the target symmetrically, but they
anchor to different things, and only one of them bounds where the policy ends up.

$$\pi^*(a) \propto \pi_{\text{old}}(a)^{\frac{\beta}{\beta+\lambda}}\;\pi_{\text{ref}}(a)^{\frac{\lambda}{\beta+\lambda}}\exp\!\Big(\frac{A(s,a)}{\beta+\lambda}\Big)$$

$\beta$ anchors to the **previous iteration**, which is refreshed every round, so it bounds
step size and nothing else. A random walk with small steps still arrives anywhere. In a
60-iteration FDN run with $\beta = \mathrm{sd}(\hat{A})$, per-step KL averaged 0.119 nats
and never spiked — and the policy still ended 1.30 nats from where it started.

$\lambda$ anchors to a **frozen reference**, so it bounds total drift. The same run with
$\lambda = \beta$ reached KL 0.086 from the reference by iteration 5 and sat there for the
remaining 55, an equilibrium between the advantage pushing out and the anchor pulling back.

Choosing them: $\beta$ absolute, near one advantage sd for the reward in use (~0.02 for a
Diamond+ transfer reward, ~0.009 for the label-starved top-player one), so the update
tracks how much signal there actually is. Then $\lambda$ as a multiple of $\beta$, tuned so
the realized $\mathrm{KL}(\pi\|\pi_{\text{ref}})$ settles where you want it. Or solve the
MPO dual for $\beta$ by bisecting until $\mathbb{E}_s\,\mathrm{KL}(\pi^*\|\pi)$ hits a
target $\epsilon$.

## Where DraftNet is allowed to enter

Four independent switches, not one decision:

| Slot | Use DraftNet? | Reason |
|---|---|---|
| Reward | **No** | The `dn_*` winrate models take DraftNet's own scores as inputs (0.986 Spearman with its p1p1 order on one-card pools). Optimizing DraftNet against a restatement of DraftNet is circular. Use `mlp_400x2`. |
| Policy initialization | Optional | Free, and it starts the policy on the pool manifold the reward model actually covers. Turning it off is the tabula-rasa experiment. |
| Opponent field | Metagame-dependent | DraftNet opponents = best response to the human Arena field. Self-play opponents = the game-theoretic equilibrium. Different objectives; both valid. |
| KL anchor | Only if it drifts | The $\pi(a\mid s)$ factor in $\pi^*$ already anchors each step to the *previous* policy. An extra anchor to a frozen DraftNet bounds cumulative drift. |

## Known failure mode

The reward is a learned function with support only near human pools, and this loop
optimizes against it. Two measurements from `tabula_rasa_check.py` and `run_env_check.py`
on FIN:

- Unconstrained hill-climbing on `mlp_400x2` reaches 0.9996 (42 copies of one card). With
  a copy cap and a real rarity mix it still reaches 0.913, against 0.694 for the best of
  4705 real pools.
- Inside the simulator, where pack availability constrains what is reachable, the
  concentration exploit dies — greedy pools are structurally indistinguishable from real
  ones. But a one-step greedy maximizer still reaches 0.634 (above the 95th percentile of
  real pools) by hoarding commons the field does not contest and the reward overvalues:
  1.67 copies of `Adventurer's Inn` per pool against 0.22 in real pools.

Ensembling several winrate models and subtracting their disagreement does **not** fix
this. On adversarial pools the models spread ~20x wider than on real pools, so
disagreement is a usable *detector* — but they are trained on the same labels with the
same inputs, so all of them still rate the exploit far above any real pool. Correlated
errors; seed and architecture diversity do not buy independence.

The exploit that survives pack constraints lives on a measurable axis: how often the
policy takes a card, relative to how often any real pool holds it. That is the natural
place for a penalty.
