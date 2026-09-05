# Learning draft policies with reinforcement learning

Notes from a day of experiments on FDN. The method itself is written up separately in
[`rl_sandbox/METHOD.md`](../rl_sandbox/METHOD.md); this is what we ran, what happened, and
what we concluded.

**Headline.** A KL penalty anchoring the policy to DraftNet is what makes this work. At
`kl_lambda = 1` self-play learns small, safe adaptations on top of human behaviour —
DraftNet's pick order is preserved almost exactly (Spearman 0.986), human-like colour
flexibility is retained, and the policy still improves on DraftNet by the reward's own
measure. Without the anchor the same setup drifts into degenerate strategies within a few
dozen iterations, regardless of how it was initialized or which reward it was given.

## Setup

Three pieces, all of which already existed in the repo in some form:

| role | model |
|---|---|
| policy (what to pick) | `DraftNet` — `forward(collection, pack)` is already a distribution over cards in the pack |
| reward (how good is the result) | `WinrateNet` variants, applied to the finished 42-card pool |
| reference (what not to drift from) | frozen `DraftNet` |

New this round: a vectorized draft environment, the RL training loop, and an explicit
colour-structure term for the winrate models.

### The environment (`rl_sandbox/draft_env.py`)

Eight seats, three 14-card boosters, alternating pass direction, and the wheel. Packs are
not generated from a collation model — they are sampled from 300,000 real opening packs
extracted from the 17lands data (`rl_sandbox/packbank.py`), so pack contents are exactly
the observed distribution. Basic lands occupy the pad index at `num_cards`, matching
`CollectionWinrateDataset`, so simulated pools are directly comparable to training rows.

Invariants are asserted in `rl_sandbox/test_draft_env.py`: the wheel returns a seat's own
pack eight picks later with six cards left, pass direction alternates by booster (verified
on pack *contents*, since sizes cannot distinguish the two directions), cards are
conserved, no seat takes a card it was not offered, and basics are only taken last.

Validation: DraftNet playing all eight seats produces pools structurally
indistinguishable from real ones — 34.1 distinct cards vs 34.3, max copies 2.83 vs 2.76,
top-8 mass 14.8 vs 14.7 — and matches human colour behaviour closely (2.05 mean deck
colours vs 2.08, 97.9% vs 98.0% concentration).

### The method

Rollout-based policy iteration: enumerate every legal action at a state, fork the draft
once per candidate under common random numbers, play each to completion, score the
finished pools, and fit the policy to the closed-form KL-regularized improvement target.
The projection step is cross-entropy against a soft target, which is already the loss
`statisticaldrafting/train.py` runs — only the target vector changes.

## Reward models

### The colour term

Two-colour decks beat three-plus-colour decks, and a 281-wide count vector with ~17k
labelled drafts is an expensive way to learn that. `statisticaldrafting/winratecolour.py`
adds an explicit term **on the logit**:

```
logit = base_network(collection, n_picks) + colour_head(colour_features, n_picks)
```

16 permutation-invariant features (sorted colour shares, sorted counts, top-2 share,
Herfindahl, entropy, a soft colour count with learnable threshold and sharpness,
multicolour and colourless mass), 1,211 parameters, zero-initialized so an untrained
model is exactly its base network.

### Results, FDN Diamond+Mythic (17,393 train drafts, 3 seeds)

Ceiling: gain 0.00814, r 0.2475.

| architecture | params | gain | % ceiling | r@pool | % r ceiling |
|---|---|---|---|---|---|
| linear | 368 | 0.00243 | 29.9% | 0.1849 | 74.7% |
| mlp_400x2 | 282k | 0.00261 | 32.0% | 0.1953 | 78.9% |
| dn_all_features | 559k | 0.00259 | 31.8% | 0.1995 | 80.6% |
| mlp_400x2_colour | 284k | 0.00269 | 33.0% | 0.2052 | 82.9% |
| **dn_all_features_colour** | 560k | **0.00284** | **34.9%** | **0.2054** | **83.0%** |

Seed sd is 0.00004 on gain and 0.0021 on r@pool, so the colour term's +0.0099 on r@pool is
roughly 4 sigma. It stacks with DraftNet transfer rather than duplicating it.

### A top-player split

`create_winrate_dataset(player_filter="draftnet")` selects players the way DraftNet's own
training set does — 95% confidence the player's true winrate exceeds 0.55 — instead of by
rank. On FDN this keeps 4.7% of drafts against the rank filter's 20.3%, and the two
populations overlap in only 1,698 drafts: the winrate filter selects strong players across
*every* rank (platinum 32%, gold 16%, diamond 16%, mythic 14%, silver 12%, bronze 10%),
because a high-winrate player who does not grind ladder never leaves platinum.

With 3,548 training drafts the reward largely collapses (ceiling: gain 0.00875, r 0.2523):

| architecture | gain | % ceiling |
|---|---|---|
| mlp_400x2 | 0.00001 | **0.1%** |
| mlp_400x2_colour | 0.00050 | 5.7% |
| dn_all_features_colour | 0.00118 | 13.5% |

A 282k-parameter card-weight model learns *nothing* at this sample size. Essentially all
remaining power comes from the two low-dimensional priors — a sharper version of the
existing finding that transfer helps hardest where labels are scarce.

## Experiments

### 1. Is the reward exploitable? (yes)

Unconstrained hill-climbing on `mlp_400x2` reaches a predicted 0.9996 winrate with 42
copies of one card; with a copy cap and a real rarity mix it still reaches 0.913, against
0.694 for the best of 4,705 real pools. Inside the simulator, pack availability kills the
*concentration* exploit entirely — greedy pools are structurally indistinguishable from
real ones — but a one-step greedy maximizer still reaches 0.634 by hoarding commons the
field does not contest.

Ensembling several winrate models and subtracting their disagreement does **not** fix
this. Disagreement rises ~20x on adversarial pools, so it is a usable detector, but the
models are trained on the same labels with the same inputs and all of them still rate the
exploit far above any real pool.

### 2. Tabula rasa self-play

Random initialization, no anchor, self-play against seven copies of itself.

- **Self-play winrate is not a progress metric.** It was flat at 0.473 → 0.485 across 80
  iterations while the same run improved +0.283 against a frozen opponent. Eight seats
  improving together compete for the same cards and the gains cancel.
- **Card quality is learnable from win/loss records alone.** p1p1 correlation with
  DraftNet rose from random to 0.463, and by iteration 80 it independently first-picked
  Liliana, Dreadhorde General.
- **Colour discipline is not.** Decks stayed at 3.36 mean colours against a human 2.08,
  from iteration 10 through 80.

The p1p1 decomposition explained why: measured rollout advantage correlated **0.915** with
the reward's context-free one-card value. Rolling out 41 further picks told us almost
nothing the reward did not already say about the card alone, because context can only
appear in Q if the rollout policy responds to context — and this one did not. Colour
discipline is a multi-step commitment and was invisible to a one-action deviation.

### 3. Better rewards

| reward the agent trained against | 2c decks | mean deck colours | top-2 share |
|---|---|---|---|
| `mlp_400x2` (diamond) | 10.0% | 3.36 | 73.8% |
| `dn_all_features` (diamond) | 30.5% | 2.80 | 84.1% |
| `dn_all_features_colour` (diamond) | 61.9% | 2.32 | 92.6% |
| `dn_all_features_colour` (top-player) | **85.4%** | **2.00** | **98.5%** |
| DraftNet | 81.3% | 2.04 | 97.7% |
| real humans | 88.0% | 2.06 | 98.2% |

Colour discipline was solved by improving the reward — the top-player agent builds
two-colour decks at a *higher* rate than DraftNet. ρ(rollout, static) fell from 0.915 to
0.353 along the way, meaning the rollouts finally carried contextual information.

**Colour choice was not solved.** Every tabula-rasa agent forced green-red: pair entropy
1.43–1.85 against a human 2.17, with GR taken 29–42% of the time. The top-player agent did
reach the UWB archetypes 53% of the time — exactly the human rate — but its pair entropy
was the *lowest* of any agent (1.46), having gone bimodal rather than responsive.

That behaviour is not obviously wrong on the labels: GR is the **second-best** pair by
realized winrate in every population (0.681 top-player, behind UW's 0.686), because it is
underdrafted at 7.5% share against BU's 18%. The observational data cannot separate "GR is
good" from "GR is open," so no pool-based reward can either. The pathology is not choosing
GR; it is choosing GR in 100% of drafts, unable to notice when the lane is contested.

### 4. The KL penalty

An extra constraint to a frozen reference in the improvement step keeps the closed form:

```
pi*(a) ~ pi_old(a)^(beta/(beta+lambda)) pi_ref(a)^(lambda/(beta+lambda)) exp(A/(beta+lambda))
```

a geometric mixture of the two anchors. `kl_lambda` is lambda/beta, so 0 recovers the
unanchored update. All arms below: DraftNet initialization, top-player
`dn_all_features_colour` reward, 60 iterations, seven DraftNet opponents.

| kl_lambda | KL from DraftNet | training reward | vs real pools | held-out reward | p1p1 rho | top pair | pair entropy |
|---|---|---|---|---|---|---|---|
| real humans | — | 0.6864 | — | **0.5439** | — | BU 19% | 2.17 |
| DraftNet | 0.00 | 0.6807 | −0.0057 | 0.5089 | 1.000 | BW 15% | 2.27 |
| **0** | 1.30 | **0.7203** | +0.0339 | 0.5072 | 0.565 | **GR 34%** | **1.69** |
| **0.3** | 0.27 | 0.7092 | +0.0228 | **0.5292** | 0.950 | BW 23% | 2.03 |
| **1** | 0.09 | 0.6942 | +0.0078 | 0.5193 | 0.986 | BW 16% | 2.21 |
| **4** | 0.04 | 0.6867 | +0.0003 | 0.5160 | 0.994 | BW 18% | 2.22 |

The held-out column is a Diamond+ model that was never the training reward. It ranks the
arms in almost the reverse order of the training reward, which is the reward-hacking
signature: the unanchored run scores highest on what it optimized and *lowest* on the
impartial judge — below DraftNet.

**Initialization was not the lever; the anchor was.** DraftNet initialization with no
anchor drifted straight back to GR-forcing (entropy 1.69, p1p1 rho falling from 1.0 to
0.565). Sixty iterations of reward-chasing erased the signal-reading it started with.

**Beta and lambda are the same object pointed at different targets.** Beta anchors to the
*previous iteration*, refreshed every round, so it bounds step size and nothing else — a
random walk with small steps still arrives anywhere. Measured: per-step KL averaged 0.119
nats and never spiked, and the policy still ended 1.30 nats from where it started. Lambda
anchors to a *frozen* reference, so it bounds total drift; the same run at lambda = beta
reached 0.086 by iteration 5 and sat there for the remaining 55.

There is also a flaw worth recording. `beta = beta_scale * sd(advantage)` makes the step a
fixed distance in advantage standard deviations, so the noise divides back out and the
update is exactly as confident on a weak reward as a strong one — the per-group-std
problem this project criticises GRPO for, one level up. `--beta-absolute` (now the
default) fixes beta in winrate units, so the update genuinely degrades toward a no-op as
the advantage shrinks.

## What the anchored agent does differently

Against DraftNet, `kl_lambda = 1` agrees on ~84% of picks, evenly spread across the draft.
On-colour rates are identical, so it is not changing colour discipline — DraftNet already
had that. The differences are in card evaluation: it takes cards DraftNet ranks lower
(mean rank 172 vs 157), downgrades gold cards at pack-1 pick-1, and — the largest single
change — takes much less cheap removal (Bake into a Pie 21% vs 48%).

At `kl_lambda = 0.3`, agreement falls to 73% and the pattern sharpens: more expensive
top-end (Exsanguinate 25% vs 9%, Painful Quandary 25% vs 11%), much less cheap interaction
(Bake into a Pie 8.5% vs 42%), and almost no mana fixing (Tranquil Cove 0.4% vs 13%).

Measuring the reward's *contextual* marginal value splits those into two different
failures. Exsanguinate sits at the 95th percentile of contextual value despite being
DraftNet's 273rd card — the agent is faithfully optimizing a reward that is simply wrong.
Bake into a Pie sits at the 86th percentile — the reward likes it and the policy dropped
it anyway, which is the policy failing to fit its own training signal.

The removal-and-fixing pattern points at a structural gap: the reward reads a vector of
card identities and has **no representation of mana cost or curve**. A pool of 23 six-drops
and a pool with a sensible curve are the same kind of object to it. That is the same shape
of problem colour was, and it wants the same shape of fix — a small set of learnable curve
features — but `data/cards/*.csv` carries no mana cost or type line, so the data would
have to come from elsewhere first.

## Conclusion

**A KL penalty to DraftNet is what makes self-play useful here, and `kl_lambda = 1` is the
setting at which it learns small, genuine adaptations on top of human behaviour.**

At that setting the policy keeps DraftNet's pick order almost exactly (Spearman 0.986) and
its human-like colour flexibility (pair entropy 2.21 against a human 2.17), changes about
one pick in six, and still improves on DraftNet by the reward's own measure (0.6942 vs
0.6807). It is also the least reward-hacked agent trained: +0.008 over real human pools,
against +0.034 for the unanchored run and +0.03 to +0.10 for every earlier agent.

The reason the anchor is necessary is not only that the reward is weak. A terminal
pool-based reward is *incapable in principle* of distinguishing "I read that green was open
and ended in GR" from "I always draft GR" — both produce the same 42-card pool. The
information that separates them lives in the pick sequence relative to what was passed,
which never enters the reward at any strength. DraftNet has it, because it was trained on
picks conditioned on packs. The anchor supplies a class of knowledge the reward cannot
represent, so how far the policy can usefully drift is bounded by how much of drafting the
reward can express.

That is why the gain is small and why it should be. Within the space the reward can judge,
the policy improves; everything the reward is blind to is held fixed by the anchor. That
is the honest kind of improvement, and it is what `kl_lambda = 1` buys.

## Caveats

- **No ground truth anywhere.** Every quality judgement is the training reward (circular),
  a held-out reward (correlated — same architecture family, same colour term), agreement
  with DraftNet (circular with the anchor), or domain judgement. Realized winrate would
  settle it and cannot be measured here.
- **One seed per arm.** Held-out gaps between kl_lambda arms (0.5292 vs 0.5193) are smaller
  than seed-to-seed variation seen elsewhere in this project, so the ordering
  0 < DraftNet < 4 < 1 < 0.3 is suggestive, not established.
- **The peak is guessed from four points.** kl_lambda in {0.2, 0.3, 0.5} at two seeds each
  would locate it properly.
- **One set.** Everything here is FDN Premier.
- **The new defaults are smoke-tested only.** Every result above used
  `beta_scale = 1.0`; `--beta-absolute 0.02` has not been run end to end.

## Next steps, in rough order of expected value

1. **Curve features in the reward**, mirroring the colour term. Needs mana cost data.
2. **Locate the kl_lambda peak** with multiple seeds.
3. **A signal-reading test**: seat the agent against fields that deliberately vacate
   different colours and measure whether its pair choice moves. This is the direct test of
   whether it responds to what it is passed, and it has not been run.
4. **A reward that sees the pack, not just the pool** — the only way to raise the ceiling
   on how far the policy can usefully drift.
5. **Pooling sets** with a per-set bias, since labels are the binding constraint
   everywhere.

## Where things live

| path | what |
|---|---|
| `rl_sandbox/METHOD.md` | the method, derived |
| `rl_sandbox/draft_env.py` | 8-seat vectorized draft, policies, forking for CRN |
| `rl_sandbox/packbank.py` | real opening packs from 17lands |
| `rl_sandbox/test_draft_env.py` | environment invariants |
| `rl_sandbox/train_policy.py` | the training loop |
| `rl_sandbox/inspect_policy.py` | p1p1 order, colour profile, cross-model check |
| `rl_sandbox/p1p1_decompose.py` | policy vs rollout advantage vs static reward value |
| `rl_sandbox/policy_diff.py` | pick-by-pick comparison against DraftNet or another policy |
| `rl_sandbox/colour_distribution.py` | pool and deck colour counts |
| `rl_sandbox/colour_probe.py` | forced two-colour counterfactual |
| `rl_sandbox/tabula_rasa_check.py` | synthetic pool families, adversarial hill-climb |
| `rl_sandbox/run_env_check.py` | environment validation, greedy-value probe |
| `statisticaldrafting/winratecolour.py` | the learnable colour term |
