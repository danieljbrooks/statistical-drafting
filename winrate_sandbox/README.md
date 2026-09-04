# Deck winrate prediction (`WinrateNet`)

Predicts the per-match win probability of the deck a draft collection becomes, from the
collection vector alone. `expected_match_wins(p)` converts that to expected match wins
for a 7-win / 3-loss Arena event.

## Pipeline

```bash
# 1. Build train/val/test splits from raw 17lands data (~1 min per set).
#    min_rank=None keeps every rank, including unranked drafts.
python -c "import statisticaldrafting as sd; sd.create_winrate_dataset('FIN', 'Premier', min_rank='diamond')"

# 2. Compare architectures (regularization grid on val, several seeds scored on test).
python winrate_sandbox/run_architecture_comparison.py FIN --rank-tag diamond

# 3. Per-pick-count skill and calibration for one model.
python winrate_sandbox/analyze_winrate_model.py FIN dn_all_features --dropout 0.7 --input-dropout 0.4

# 4. How much signal is even there.
python winrate_sandbox/noise_ceiling.py FIN --rank-tag diamond

# 5. Both comparisons, normalized by each dataset's noise ceiling.
python winrate_sandbox/summarize_results.py FIN

# 6. Train one model for actual use, with a JSON sidecar describing it.
python winrate_sandbox/train_final_model.py FIN dn_all_features --rank-tag diamond

# 7. Score a pool. Edit POOL at the top of the script, or pass a file.
python winrate_sandbox/predict_pool.py
python winrate_sandbox/predict_pool.py --per-card

# 8. Rank every card in a set by the winrate of a one-card pool holding it.
python winrate_sandbox/p1p1_ratings.py FDN
python winrate_sandbox/p1p1_ratings.py FDN --min-support 20 --csv fdn_p1p1.csv
```

All commands run from the repo root.

## Scoring a pool

`predict_pool.py` takes a list of cardnames and prints the predicted match winrate,
expected match wins, and where the pool sits against held-out pools. Set `SET` at the top
of the file (or pass `--set`) to choose which set's model to score with; the cards in
`POOL` have to belong to it.

The default is a typical completed FDN pool — all 42 picks, no basics taken — which the
model puts at 0.514, dead on the median. Real pools from the held-out splits are included
commented out for both FDN (strong / weak / mid-draft) and FIN (typical / strong), so
predictions can be sanity-checked against decks with known records.

Unknown cardnames are reported with close-match suggestions rather than silently dropped;
if *nothing* matches, it says the pool is probably from a different set rather than
printing 42 individual errors.
Partial pools are supported: pass fewer cards and the estimate is pulled toward the
population average, which is correct, because an 8-card pool genuinely says less than a
finished one.

## Ranking a set by one-card pools

`p1p1_ratings.py` scores every card in a set on its own — a collection holding just that
card, one pick in — and ranks the set by the result. It is the winrate model's answer to
"how good is this card as a pack-1 pick-1", expressed directly in winrate.

The `p1p1s` column counts how often each card was *actually* taken first in the training
split, and it matters: 22% of FDN cards and 16% of FIN cards were never first-picked at
all, so they never appear as a one-card pool during training and their estimates are
extrapolated from the rest of the draft. The top of the ranking rests on 150–300 real
first picks per card; the bottom rests on none. `--min-support 20` restricts the table to
the well-supported part. The extreme low end is where this bites hardest. FIN's
lowest card is `Wastes` at 0.124: it is a basic land, but not one of the five the
pipeline filters out, so it counts as draftable while nobody ever takes it first — an
unsupported estimate landing far outside anything the model observed.

`--all` prints every card, `--csv` writes the full table.

## Per-card contribution

`--per-card` on `predict_pool.py` leaves each card out in turn to show what it is worth *in this pool*. Off-colour
cards score near zero or negative even when they are strong in the abstract, which is the
synergy signal doing its job.

Checkpoints are named `<SET>_<mode>_<rank_tag>_<architecture>.pt`; the rank tag matters,
because the same architecture trained on Diamond+Mythic and on all ranks are different
models.

## Data

- Label is the draft's event record: `event_match_wins / (wins + losses)`.
- Rank filter is configurable (`min_rank`); first 2 days of the format are dropped, as
  are drafts with zero games played. No filtering on player winrate or games played.
- Basic lands are real picks but not part of the collection, so they are routed to a pad
  slot. Dropping those rows instead would truncate ~55% of drafts (basics wheel to the
  last pick of a pack) and silently bias the sample.
- Collections are stored as pick sequences, so every prefix length 1..42 is available for
  free; `build_collections` materializes count vectors per batch on device.
- Split is **by draft** into 70/15/15 train/val/test. Val drives early stopping and
  architecture selection; test is only ever read for reporting.

Datasets are tagged by rank filter so several can coexist:

| set | rank filter | drafts | share of set | population winrate | mean games |
|---|---|---|---|---|---|
| FIN | `diamond` (Diamond + Mythic) | 31,371 | 22.4% | 0.538 | 5.55 |
| FIN | `allranks` (every rank, incl. unranked) | 118,958 | 84.8% | 0.551 | 5.47 |
| FDN | `diamond` | 24,847 | 18.8% | 0.538 | 5.53 |

358 cards, 42 picks. All-ranks has a slightly *lower* true winrate spread (0.061 vs
0.065) — Arena matchmaking is rank-based, so each player faces similar opponents and the
spread compresses rather than widening.

## Objective

Binomial NLL per game, `-(wins·log p + losses·log(1-p)) / games`. Arena events stop at a
fixed win or loss count, and that stopping rule does not depend on `p`, so the record's
likelihood is still `p^wins · (1-p)^losses` up to a constant — this is the right
objective, and it automatically weights a 7-2 more than an 0-3. Regressing MSE on the
observed winrate instead costs most of the signal (see finding 8).

## Transferring DraftNet into the winrate model

DraftNet is fit to millions of picks by strong drafters. The winrate model has tens of
thousands of labelled drafts. Anything DraftNet already knows about card quality and
synergy is signal the winrate model should not have to rediscover.

`DraftNetFeatures` exposes a frozen DraftNet three ways:

- **power** — scores with an *empty* pool: context-free card power (DraftNet's p1p1 order).
- **context** — scores with the *current collection* as the pool: what DraftNet wants
  next, which also says how well the collection's own cards fit each other.
- **hidden** — the 400-d penultimate activation: DraftNet's representation of the
  collection itself.

Eight scalars distil these: `power_sum/mean/max`, `context_sum/mean`, `synergy_sum`
(context minus power), and **`power_top23` / `context_top23`** — the sum over the best 23
cards, counting duplicates. A 42-card pool becomes a ~23-spell deck, so the top-23 sum is
a much better proxy for what actually gets played than the whole pool, and ranking is a
nonlinearity the winrate model cannot express over raw counts.

Frozen features are precomputed once per dataset (`precompute_features`) rather than
recomputed every batch; on the all-ranks split that is the difference between a 10-minute
and an hour-long run.

## Results

Raw log loss is not comparable across rank filters — different base winrates, different
amounts of binomial noise. What is comparable is the fraction of the achievable signal
each model captures, so both tables are normalized by that split's own ceiling.
`gain` = nats/game better than predicting the population winrate. `@pool` = completed
42-card pools only. `dn_*` are DraftNet transfer models.

### Diamond + Mythic (21,960 train drafts, 3 seeds)

Ceiling: gain 0.00839, r 0.2515.

| architecture | params | gain | % of ceiling | gain@pool | r@pool | % of r ceiling |
|---|---|---|---|---|---|---|
| **dn_all_features** | 621k | **0.00377** | **44.9%** | **0.00587** | **0.2415** | **96.0%** |
| dn_scalars_collection | 316k | 0.00366 | 43.6% | 0.00549 | 0.2357 | 93.7% |
| dn_context_collection | 457k | 0.00364 | 43.3% | 0.00559 | 0.2349 | 93.4% |
| dn_hidden_collection | 474k | 0.00360 | 42.9% | 0.00556 | 0.2358 | 93.7% |
| dn_trunk_finetune | 307k | 0.00348 | 41.5% | 0.00547 | 0.2312 | 91.9% |
| dn_scalars_mlp | 6.8k | 0.00335 | 40.0% | 0.00503 | 0.2203 | 87.6% |
| dn_scalars_only | **729** | 0.00320 | 38.1% | 0.00510 | 0.2270 | 90.2% |
| mlp_400x2 | 313k | 0.00296 | 35.3% | 0.00498 | 0.2211 | 87.9% |
| dn_hidden_only | 331k | 0.00288 | 34.3% | 0.00447 | 0.2110 | 83.9% |
| mlp_400x2_no_pick_cond | 306k | 0.00284 | 33.9% | 0.00467 | 0.2131 | 84.7% |
| mlp_256x2 | 164k | 0.00281 | 33.5% | 0.00466 | 0.2162 | 85.9% |
| mlp_1024x2 | 1.44M | 0.00275 | 32.8% | 0.00449 | 0.2124 | 84.4% |
| embed_32 | 15k | 0.00267 | 31.8% | 0.00431 | 0.2038 | 81.0% |
| mlp_1024x3_res | 2.49M | 0.00266 | 31.7% | 0.00449 | 0.2155 | 85.7% |
| mlp_2048x2 | 4.98M | 0.00259 | 30.9% | 0.00454 | 0.2152 | 85.6% |
| linear | 445 | 0.00244 | 29.0% | 0.00382 | 0.1934 | 76.9% |
| mlp_400x2_full_pool_only | 313k | 0.00234 | 27.9% | 0.00457 | 0.2134 | 84.8% |
| embed_128 | 82k | 0.00217 | 25.9% | 0.00352 | 0.1917 | 76.2% |
| dn_trunk_frozen | 1.1k | 0.00210 | 25.0% | 0.00322 | 0.1985 | 78.9% |
| mlp_400x2_mse_loss | 313k | 0.00057 | 6.8% | 0.00374 | 0.2011 | 79.9% |

### All ranks (83,271 train drafts, 2 seeds)

Ceiling: gain 0.00761, r 0.2326.

| architecture | params | gain | % of ceiling | gain@pool | r@pool | % of r ceiling |
|---|---|---|---|---|---|---|
| **dn_scalars_collection** | 316k | **0.00386** | **50.7%** | **0.00576** | **0.2257** | **97.0%** |
| dn_trunk_finetune | 307k | 0.00378 | 49.7% | 0.00570 | 0.2215 | 95.2% |
| mlp_400x2 | 313k | 0.00366 | 48.1% | 0.00551 | 0.2217 | 95.3% |
| mlp_1024x2 | 1.44M | 0.00353 | 46.4% | 0.00540 | 0.2188 | 94.0% |
| mlp_2048x2 | 4.98M | 0.00351 | 46.2% | 0.00539 | 0.2183 | 93.9% |
| linear | 445 | 0.00297 | 39.1% | 0.00411 | 0.1926 | 82.8% |
| dn_scalars_mlp | 6.8k | 0.00295 | 38.7% | 0.00437 | 0.1981 | 85.1% |
| dn_scalars_only | 729 | 0.00284 | 37.3% | 0.00443 | 0.2035 | 87.5% |

Seed-to-seed sd of test NLL is 0.0002 or less throughout, so these orderings are stable.
`dn_all_features`, `dn_context_collection` and `dn_hidden_collection` were not run on the
all-ranks split — their 400-d and 358-d features are too large to cache, and recomputing
them per batch over 3.5M examples per epoch was not worth the hours. Given how close they
sit to `dn_scalars_collection` on Diamond+Mythic, that is a gap in the table rather than a
result.

### A second set: FDN (Foundations)

`dn_all_features` trained the same way on FDN Premier, Diamond+Mythic (17,393 train
drafts, 281 cards). Its noise ceiling is nearly identical to FIN's — gain 0.00814,
r 0.2475 — but the model captures less of it:

| set | train drafts | cards | gain | % of ceiling | r@pool | % of r ceiling |
|---|---|---|---|---|---|---|
| FIN | 21,960 | 358 | 0.00377 | 44.9% | 0.2415 | 96.0% |
| FDN | 17,393 | 281 | 0.00273 | 33.5% | 0.2081 | 84.1% |

The true winrate spread is the same in both (0.064), so the signal is there — FDN just has
21% fewer labelled drafts, which is the same bottleneck as everywhere else in this
project. Whether Foundations is also a genuinely flatter format (it is designed as an
entry-point set, and its predictions span 0.42–0.60 against FIN's 0.45–0.61) is not
separable from the label count without an all-ranks FDN run.

## Findings

1. **DraftNet transfer works, and it works hardest where labels are scarce.** On
   Diamond+Mythic it lifts the best model from 35.3% to 44.9% of the ceiling — a 27%
   relative gain — with no new labels. On all-ranks, where there are 3.8x more labels, the
   same idea buys only 48.1% → 50.7%.
2. **Transfer and data are substitutes.** Expanding to all ranks alone lifts `mlp_400x2`
   from 35.3% to 48.1%; transfer alone lifts it to 44.9%. Together, 50.7%. Both attack the
   same bottleneck, which is why they do not stack cleanly — and which confirms the
   bottleneck really is labels.
3. **Both of the intuitions behind the transfer idea pay off, and combining them pays
   most.** Empty-pool power and current-pool context each carry signal on their own; the
   full feature set (`dn_all_features`) beats every single-view variant.
4. **Eight numbers beat a 313k-parameter network.** `dn_scalars_only` — 729 trainable
   parameters, no raw collection at all, just 8 scalars read off a frozen DraftNet — gets
   38.1% of the ceiling against `mlp_400x2`'s 35.3%. Individually, `context_top23`
   correlates at r≈0.18 with realized winrate using *zero* winrate labels.
5. **DraftNet's representation alone is not the win; the combination is.** `dn_hidden_only`
   (34.3%) is no better than the raw collection, and `dn_trunk_frozen` (25.0%) is much
   worse. DraftNet's features have to sit *alongside* the collection, or the trunk has to
   be fine-tuned, to help.
6. **Capacity still does not help, at either data scale.** 400x400 stays the sweet spot;
   1024x2 and 2048x2 are worse on test even with 83k training drafts.
7. **Training on all prefix lengths beats training on completed pools only**, even when
   scored on completed pools (gain@pool 0.00498 vs 0.00457). Intermediate collections act
   as augmentation.
8. **Binomial NLL matters more than any architecture choice.** Swapping to MSE on the
   observed winrate drops gain from 35.3% to 6.8% of ceiling, because MSE treats an 0-3
   and a 2-3 as equally informative and chases binomial noise.

## Calibration and usable output (`dn_all_features`, Diamond+Mythic test, 42-card pools)

| decile | predicted | actual | E[match wins] |
|---|---|---|---|
| 1 | 0.4452 | 0.4216 | 2.36 |
| 5 | 0.5275 | 0.5276 | 3.15 |
| 10 | 0.6191 | 0.6202 | 4.18 |

Predicted winrate spans 0.451 (p5) to 0.614 (p95). Top-decile pools actually won 3.85
matches on average against 1.97 for bottom-decile pools — a 1.9 match spread that the
model identifies from the collection alone (`mlp_400x2` separates the same deciles by
1.7 matches).

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
  us weight cards by how often they were maindecked, which should be a straight upgrade —
  and the `*_top23` features working as well as they do is evidence for it.
- Player skill is baked into the label. `user_game_win_rate_bucket` is deliberately not
  filtered on, but it could be used as an auxiliary input at training time and zeroed at
  inference to strip skill out of the deck estimate.
- Only one set is trained here. Pooling sets (with a shared per-set bias) would multiply
  the label count, which is the binding constraint.
- The transferred DraftNet is trained on *top-player* picks, which suits the
  Diamond+Mythic population better than the all-ranks one. That may be part of why
  transfer buys less on all-ranks; a like-for-like comparison would need an all-ranks
  DraftNet.
- `dn_all_features` needs an all-ranks run (see the note under Results).
