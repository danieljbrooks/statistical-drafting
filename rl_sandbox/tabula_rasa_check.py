"""
Does the winrate model have a usable gradient off the human pool manifold?

Tabula-rasa RL only works if the reward can tell a garbage pool from a good one. The
reward model is fit to ~22k pools that human drafters actually built; a randomly
initialized policy produces pools nowhere near that manifold, where the model is purely
extrapolating. This scores several families of synthetic pool against real held-out
pools to see whether the ordering survives.

Families, weakest-looking to strongest:
  uniform_random   42 cards drawn uniformly from the set - ignores rarity entirely.
  random_draft     rarity mix of a real draft, colors ignored. This is what a randomly
                   initialized policy drafting from real packs would actually produce.
  five_color       random_draft forced to spread evenly over all five colors.
  two_color        random_draft restricted to two colors plus colorless.
  real             completed held-out pools from the test split.
  hill_climb       greedy ascent on the reward itself: the adversarial upper bound.

Run from the repo root:  python rl_sandbox/tabula_rasa_check.py FIN
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "winrate_sandbox"))
import statisticaldrafting as sd
from architectures import build_network

MODEL_FOLDER = "data/winrate_models/"
PICKS = 42


def load_model(set_abbreviation, draft_mode, rank_tag, architecture):
    name = f"{set_abbreviation}_{draft_mode}_{rank_tag}_{architecture}"
    with open(os.path.join(MODEL_FOLDER, name + ".json")) as f:
        meta = json.load(f)
    network = build_network(
        architecture,
        cardnames=meta["cardnames"],
        picks_per_draft=meta["picks_per_draft"],
        base_winrate=meta["base_winrate"],
        dropout=meta["dropout"],
        input_dropout=meta["input_dropout"],
        set_abbreviation=meta["set"],
        draft_mode=meta["draft_mode"],
    )
    network.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, name + ".pt"), weights_only=True))
    network.eval()
    return network, meta


@torch.no_grad()
def score(network, collections, n_picks=PICKS, batch=2048):
    """Predicted per-match winrate for a (n, num_cards) batch of count vectors."""
    out = []
    for i in range(0, collections.shape[0], batch):
        chunk = collections[i : i + batch].float()
        counts = torch.full((chunk.shape[0],), n_picks, dtype=torch.long)
        out.append(torch.sigmoid(network(chunk, counts)))
    return torch.cat(out).numpy()


def counts_from_indices(indices, num_cards):
    """
    (n, picks) int index array -> (n, num_cards) float count matrix.

    Index ``num_cards`` is the dataset's pad slot for basic lands, so real pools can
    hold fewer than 42 draftable cards; ``build_collections`` drops it.
    """
    indices = torch.from_numpy(np.asarray(indices)).long()
    n_picks = torch.full((indices.shape[0],), indices.shape[1], dtype=torch.long)
    return sd.build_collections(indices, n_picks, num_cards)


def sample_pools(rng, n, weights, rarity_plan, rarity_index):
    """
    n pools of PICKS cards. ``rarity_plan`` gives how many cards of each rarity, and
    ``weights`` is a per-card sampling weight (zero to exclude a card).
    """
    pools = np.zeros((n, PICKS), dtype=np.int64)
    for rarity, k in rarity_plan.items():
        if k == 0:
            continue
        pool_of = rarity_index[rarity]
        w = weights[pool_of]
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()
        column = sum(v for r, v in rarity_plan.items() if r < rarity)
        pools[:, column : column + k] = rng.choice(pool_of, size=(n, k), p=w)
    return pools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FIN")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument("--rank-tag", default="diamond")
    parser.add_argument("--architecture", default="mlp_400x2")
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    network, meta = load_model(
        args.set_abbreviation, args.draft_mode, args.rank_tag, args.architecture
    )
    cardnames = meta["cardnames"]
    num_cards = len(cardnames)

    cards = pd.read_csv(f"data/cards/{args.set_abbreviation}.csv").set_index("name")
    cards = cards.reindex(cardnames)
    rarity = cards["rarity"].fillna("common").to_numpy()
    color = cards["color_identity"].fillna("Colorless").to_numpy()

    # Rarity composition of real completed pools, so synthetic pools are not trivially
    # distinguishable by having 42 mythics.
    dataset = torch.load(
        sd.winrate_dataset_path(
            args.set_abbreviation, args.draft_mode, "test", args.rank_tag
        ),
        weights_only=False,
    )
    real_indices = dataset.pick_sequences[: args.n]
    # Index num_cards is the basic-land pad slot; it is not a draftable card.
    real_rarities = np.where(real_indices < num_cards, rarity[np.minimum(real_indices, num_cards - 1)], "pad")
    plan_float = {r: (real_rarities == r).sum() / real_indices.shape[0] for r in set(rarity)}
    rarity_plan = {r: int(round(v)) for r, v in sorted(plan_float.items())}
    # Force the plan to sum to exactly PICKS.
    while sum(rarity_plan.values()) != PICKS:
        key = max(rarity_plan, key=lambda r: plan_float[r] - rarity_plan[r]) \
            if sum(rarity_plan.values()) < PICKS else \
            max(rarity_plan, key=lambda r: rarity_plan[r] - plan_float[r])
        rarity_plan[key] += 1 if sum(rarity_plan.values()) < PICKS else -1

    rarity_index = {r: np.where(rarity == r)[0] for r in rarity_plan}
    ones = np.ones(num_cards)

    families = {}

    # 1. Uniform over every card, rarity ignored.
    families["uniform_random"] = rng.choice(num_cards, size=(args.n, PICKS))

    # 2. Rarity-realistic, colors ignored: a random policy drafting from real packs.
    families["random_draft"] = sample_pools(rng, args.n, ones, rarity_plan, rarity_index)

    # 3. Forced five-colour spread.
    mono = ["W", "U", "B", "R", "G"]
    # Rarity plan split evenly across all five colours, so this differs from two_color
    # only in colour spread and not in card power.
    five = np.zeros((args.n, PICKS), dtype=np.int64)
    for i in range(args.n):
        columns, offset = [], 0
        for r, k in rarity_plan.items():
            if k == 0:
                continue
            for j in range(k):
                c = mono[j % 5]
                allowed = np.where((rarity == r) & ((color == c) | (color == "Colorless")))[0]
                if len(allowed) == 0:
                    allowed = rarity_index[r]
                columns.append(rng.choice(allowed))
            offset += k
        five[i] = np.array(columns)
    families["five_color"] = five

    # 4. Two colours plus colourless, rarity-realistic.
    two = np.zeros((args.n, PICKS), dtype=np.int64)
    for i in range(args.n):
        pair = rng.choice(mono, size=2, replace=False)
        allowed = np.isin(color, list(pair)) | (color == "Colorless")
        two[i] = sample_pools(rng, 1, allowed.astype(float), rarity_plan, rarity_index)[0]
    families["two_color"] = two

    # 5. Real held-out pools.
    families["real_pools"] = real_indices

    print(f"{args.set_abbreviation} {args.draft_mode} [{args.rank_tag}] reward={args.architecture}")
    print(f"rarity plan per 42-card pool: {rarity_plan}")
    print(f"n per family = {args.n}\n")
    print(f"{'family':<16} {'mean':>7} {'sd':>7} {'p5':>7} {'p50':>7} {'p95':>7} {'max':>7}")
    print("-" * 60)

    results = {}
    for name, indices in families.items():
        winrates = score(network, counts_from_indices(indices, num_cards))
        results[name] = winrates
        print(
            f"{name:<16} {winrates.mean():>7.4f} {winrates.std():>7.4f} "
            f"{np.percentile(winrates, 5):>7.4f} {np.percentile(winrates, 50):>7.4f} "
            f"{np.percentile(winrates, 95):>7.4f} {winrates.max():>7.4f}"
        )

    # 6. Adversarial hill-climbs on the reward itself, at three levels of realism.
    climbs = {
        "climb_free": dict(max_copies=PICKS, respect_rarity=False),
        "climb_4copy": dict(max_copies=4, respect_rarity=False),
        "climb_packable": dict(max_copies=4, respect_rarity=True),
    }
    climbed = {}
    for name, kwargs in climbs.items():
        values, pools = hill_climb(
            network, num_cards, rng, rarity, rarity_plan, rarity_index, **kwargs
        )
        climbed[name] = (values, pools)
        print(f"{name:<16} {values.mean():>7.4f} {values.std():>7.4f} "
              f"{'-':>7} {'-':>7} {'-':>7} {values.max():>7.4f}")

    # Does an independent model agree these pools are good? Disagreement between models
    # is the cheapest available detector for "we have left the data".
    others = {}
    for architecture in ["linear", "dn_all_features"]:
        try:
            others[architecture] = load_model(
                args.set_abbreviation, args.draft_mode, args.rank_tag, architecture
            )[0]
        except FileNotFoundError:
            pass
    if others:
        print(f"\nCross-model check ({args.architecture} vs models held out of the reward):")
        print(f"{'pool set':<16} {args.architecture:>14}" + "".join(f"{a:>18}" for a in others))
        rows = {"real_pools": families["real_pools"], "random_draft": families["random_draft"]}
        rows.update({k: v[1] for k, v in climbed.items()})
        for name, indices in rows.items():
            collections = counts_from_indices(indices, num_cards)
            line = f"{name:<16} {score(network, collections).mean():>14.4f}"
            for architecture, other in others.items():
                line += f"{score(other, collections).mean():>18.4f}"
            print(line)

    for name, (values, pools) in climbed.items():
        print(f"\nTop {name} pool:")
        top = pools[int(np.argmax(values))]
        counter = pd.Series([cardnames[i] for i in top]).value_counts()
        for cardname, count in counter.head(10).items():
            print(f"  {count}x {cardname}  [{cards.loc[cardname, 'rarity']}, {cards.loc[cardname, 'color_identity']}]")
        if len(counter) > 10:
            print(f"  ... {len(counter) - 10} more distinct cards")


def hill_climb(
    network, num_cards, rng, rarity, rarity_plan, rarity_index,
    restarts=6, steps=15, max_copies=PICKS, respect_rarity=False,
):
    """
    Greedy card swaps that maximize predicted winrate, from random starts.

    ``max_copies`` caps how many copies of one card a pool may hold, and
    ``respect_rarity`` forces each slot to keep its rarity, so the resulting pool is one
    a drafter could actually open. Together they bound how far a policy constrained by
    real packs could push the reward.
    """
    if respect_rarity:
        slot_rarity = np.concatenate(
            [np.full(k, r) for r, k in rarity_plan.items() if k > 0]
        )
        pools = np.stack([
            np.array([rng.choice(rarity_index[r]) for r in slot_rarity])
            for _ in range(restarts)
        ])
    else:
        slot_rarity = None
        pools = rng.choice(num_cards, size=(restarts, PICKS))

    for _ in range(steps):
        base = score(network, counts_from_indices(pools, num_cards))
        for slot in range(PICKS):
            # Try replacing this slot with every legal card, for every restart at once.
            candidates = np.repeat(pools[:, None, :], num_cards, axis=1)
            candidates[:, :, slot] = np.arange(num_cards)[None, :]
            values = score(
                network, counts_from_indices(candidates.reshape(-1, PICKS), num_cards)
            ).reshape(restarts, num_cards)
            if slot_rarity is not None:
                values[:, rarity != slot_rarity[slot]] = -np.inf
            if max_copies < PICKS:
                # Copies already held in the other 41 slots.
                held = np.zeros((restarts, num_cards))
                other = np.delete(pools, slot, axis=1)
                np.add.at(held, (np.arange(restarts)[:, None], np.minimum(other, num_cards - 1)), 1)
                values[held >= max_copies] = -np.inf
            pools[:, slot] = values.argmax(axis=1)
        new = score(network, counts_from_indices(pools, num_cards))
        if new.mean() <= base.mean() + 1e-6:
            break
    return score(network, counts_from_indices(pools, num_cards)), pools


if __name__ == "__main__":
    main()
