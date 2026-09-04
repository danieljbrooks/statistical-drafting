# Deck winrate prediction (`WinrateNet`)

Predicts the per-match win probability of the deck a draft collection becomes, from the
collection vector alone. `expected_match_wins(p)` converts that to expected match wins
for a 7-win / 3-loss Arena event.

## Pipeline

```bash
# 1. Build train/val/test splits from raw 17lands data (~1 min per set).
python -c "import statisticaldrafting as sd; sd.create_winrate_dataset('FIN', 'Premier', min_rank='diamond')"

# 2. Compare architectures (regularization grid on val, 3 seeds scored on test).
python winrate_sandbox/run_architecture_comparison.py FIN

# 3. Per-pick-count skill and calibration for one model.
python winrate_sandbox/analyze_winrate_model.py FIN mlp_400x2

# 4. How much signal is even there.
python winrate_sandbox/noise_ceiling.py FIN
```

All commands run from the repo root.

## Data

- Label is the draft's event record: `event_match_wins / (wins + losses)`.
- Filter is Diamond + Mythic only, first 2 days of the format dropped, drafts with zero
  games dropped. No filtering on player winrate or games played.
- Basic lands are real picks but not part of the collection, so they are routed to a pad
  slot. Dropping those rows instead would truncate ~55% of drafts (basics wheel to the
  last pick of a pack) and silently bias the sample.
- Collections are stored as pick sequences, so every prefix length 1..42 is available for
  free; `build_collections` materializes count vectors per batch on device.
- Split is **by draft** into 70/15/15 train/val/test. Val drives early stopping and
  architecture selection; test is only ever read for reporting.

FIN Premier, Diamond+Mythic: 31,371 drafts (22.4% of all FIN drafts), 21,960 / 4,706 /
4,705, 358 cards, mean 5.55 games per draft, population winrate 0.538.

## Objective

Binomial NLL per game, `-(wins·log p + losses·log(1-p)) / games`. Arena events stop at a
fixed win or loss count, and that stopping rule does not depend on `p`, so the record's
likelihood is still `p^wins · (1-p)^losses` up to a constant — this is the right
objective, and it automatically weights a 7-2 more than an 0-3. Regressing MSE on the
observed winrate instead costs about 80% of the signal (see ablations).

## Results (FIN Premier, test split, mean of 3 seeds)

`gain` = nats/game better than predicting the population winrate for every deck.
`@pool` = evaluated on completed 42-card pools only.

| architecture | params | test NLL/game | gain | r (all prefixes) | gain@pool | r@pool |
|---|---|---|---|---|---|---|
| **mlp_400x2** | 313k | **0.68732** | **0.00296** | **0.172** | **0.00498** | **0.221** |
| mlp_400x2_no_pick_cond | 306k | 0.68744 | 0.00284 | 0.168 | 0.00467 | 0.213 |
| mlp_256x2 | 164k | 0.68747 | 0.00281 | 0.169 | 0.00466 | 0.216 |
| mlp_1024x2 | 1.44M | 0.68753 | 0.00275 | 0.167 | 0.00449 | 0.212 |
| embed_32 | 15k | 0.68762 | 0.00267 | 0.163 | 0.00431 | 0.204 |
| mlp_1024x3_res | 2.49M | 0.68762 | 0.00266 | 0.166 | 0.00449 | 0.216 |
| mlp_2048x2 | 4.98M | 0.68769 | 0.00259 | 0.165 | 0.00454 | 0.215 |
| linear | 445 | 0.68785 | 0.00244 | 0.157 | 0.00382 | 0.193 |
| mlp_400x2_full_pool_only | 313k | 0.68794 | 0.00234 | 0.164 | 0.00457 | 0.213 |
| embed_128 | 82k | 0.68811 | 0.00217 | 0.151 | 0.00352 | 0.192 |
| mlp_400x2_mse_loss | 313k | 0.68971 | 0.00057 | 0.151 | 0.00374 | 0.201 |

Seed-to-seed sd of test NLL is 0.0001 or less, so the ordering above is stable.

Every model needed `dropout=0.5, input_dropout=0.2, weight_decay=1e-4` (the middle of the
grid) except `embed_128`, which preferred the heaviest setting.

## What the numbers mean

There are only ~22k labelled drafts and each is worth ~5.5 games, so the labels are
extremely noisy. Method of moments on the test split (`noise_ceiling.py`):

- Observed winrate sd: 0.221, of which 0.212 is pure binomial noise.
- Implied **true** winrate sd across drafts: **0.065** — and that mixes deck quality with
  player skill, which the collection cannot see.
- Ceiling on log loss gain for a perfect model of the true winrate: **0.0084 nats/game**.
- Ceiling on Pearson r against the observed winrate: **0.25**.

`mlp_400x2` reaches 0.0050 of the 0.0084 gain ceiling and r=0.22 of the 0.25 r ceiling on
completed pools. Most of the remaining headroom is player skill, not deck quality.

## Findings

1. **Capacity does not help.** 400x400 (DraftNet's size) is the sweet spot; 1024x2,
   1024x3-residual and 2048x2 are all *worse* on test. The binding constraint is 22k
   labels, not model expressiveness. Without heavy regularization the wide nets overfit
   inside the first epoch — train NLL 0.626 against val 0.719 — which is why training
   evaluates every 25 batches rather than once per epoch.
2. **Nonlinearity buys a real but modest amount.** The 445-parameter linear model (one
   weight per card, i.e. "every card has a fixed value") already gets most of the way:
   gain@pool 0.0038 vs 0.0050. Synergy is worth roughly 30% on top of card quality.
3. **Training on all prefix lengths beats training on completed pools only**, even when
   scored on completed pools: gain@pool 0.0050 vs 0.0046. The intermediate collections
   act as augmentation.
4. **Binomial NLL matters a lot.** Swapping to MSE on the observed winrate drops overall
   gain from 0.0030 to 0.0006, because MSE treats an 0-3 and a 2-3 as equally informative
   and chases binomial noise.
5. **Pick-count conditioning is a small win** (gain 0.00296 vs 0.00284). The collection's
   L1 norm already leaks the pick count, so the embedding mostly helps calibration.

## Calibration and usable output (`mlp_400x2`, test split, 42-card pools)

| decile | predicted | actual | E[match wins] |
|---|---|---|---|
| 1 | 0.4402 | 0.4439 | 2.32 |
| 5 | 0.5281 | 0.5105 | 3.15 |
| 10 | 0.6248 | 0.6279 | 4.24 |

Predicted winrate spans 0.447 (p5) to 0.619 (p95). Top-decile pools actually won 3.87
matches on average against 2.18 for bottom-decile pools — a 1.7 match spread that the
model identifies from the collection alone.

Skill rises monotonically through the draft: r = 0.05 after one pick, 0.15 by pick 10,
0.22 by pick 35, then flat. The last ~7 picks add nothing, which is what you would expect
from picks that mostly do not make the deck.

`expected_match_wins` assumes the event is played to completion. 13.8% of events in the
data are abandoned early, so observed mean wins (2.99) sits below the completion-implied
value (3.26) at the same winrate.

## Sanity check on learned card values

Top linear weights for FIN: Esper Origins, Smuggler's Copter, Adeline Resplendent Cathar,
Nibelheim Aflame, Ardyn the Usurper, Summon: Primal Odin, Sephiroth Fabled SOLDIER, Dion
Bahamut's Dominant, Traveling Chocobo. Bottom weights are dominated by off-set bonus-sheet
cards (Traxos, Rhystic Study, Nyxbloom Ancient, Chromatic Lantern) — plausible, since those
are usually off-color speculative picks rather than deck cards.

## Known limitations / next steps

- The input is the **pool**, not the deck. `pick_maindeck_rate` in the raw data would let
  us weight cards by how often they were maindecked, which should be a straight upgrade.
- Player skill is baked into the label. `user_game_win_rate_bucket` is deliberately not
  filtered on, but it could be used as an auxiliary input at training time and zeroed at
  inference to strip skill out of the deck estimate.
- Only one set is trained here. Pooling sets (with a shared per-set bias) would multiply
  the label count, which is the binding constraint.
- Lower ranks would roughly 4x the data at the cost of a different skill population.
