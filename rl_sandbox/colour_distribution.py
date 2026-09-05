"""
Colour distribution of the pools the RL policy drafts, seated against seven DraftNet bots.

Reported two ways, because they answer different questions:

  pool   what the 42 picks look like. This is what the policy directly controls, and
         includes every speculative and off-colour pick that never makes the deck.
  deck   the 23-card deck ``IterativeDeckBuilder`` builds from that pool. This is the
         thing that is "typically two colours", and it is the fairer test: a pool with
         eight stray green cards is fine if the deck ignores them.

A colour counts toward a deck's colour count when the deck holds at least ``--min-cards``
cards of it, so a single splashed card does not turn a two-colour deck into three.

Caveat on the card data: ``data/cards/*.csv`` gives ``color_identity`` as one of
W/U/B/R/G/Multicolor/Colorless, so gold cards cannot be attributed to a specific pair.
They are counted separately rather than assigned, which slightly understates colour
counts for pools heavy in gold cards.

Run from the repo root:
    python rl_sandbox/colour_distribution.py FDN --n-drafts 1000
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "winrate_sandbox"))

import statisticaldeckbuild as sdb
import statisticaldrafting as sd
from draft_env import N_SEATS, DraftEnv, DraftNetPolicy, run_drafts
from packbank import PackBank
from run_env_check import load_winrate_model
from train_policy import PickPolicy, score_pools

MONO = ["W", "U", "B", "R", "G"]


class Adapter:
    def __init__(self, policy):
        self.policy = policy.eval()

    @torch.no_grad()
    def pick(self, pools, packs):
        return self.policy.logits(pools, packs).argmax(dim=1)


def colour_counts(counts_matrix, colour):
    """(n, 5) mono-colour card counts, plus multicolour and colourless columns."""
    mono = np.stack([counts_matrix[:, colour == c].sum(1) for c in MONO], axis=1)
    multi = counts_matrix[:, colour == "Multicolor"].sum(1)
    colourless = counts_matrix[:, colour == "Colorless"].sum(1)
    return mono, multi, colourless


def summarize(label, mono, multi, colourless, min_cards):
    n_colours = (mono >= min_cards).sum(axis=1)
    n_colours = np.clip(n_colours, 1, 5)
    share = Counter(n_colours.tolist())
    total_mono = mono.sum(axis=1)
    top2 = np.sort(mono, axis=1)[:, -2:].sum(1)
    concentration = np.divide(top2, np.maximum(total_mono, 1))
    row = {
        "label": label,
        "mean_colours": n_colours.mean(),
        "top2_share": 100 * concentration.mean(),
        "multi": multi.mean(),
        "colourless": colourless.mean(),
    }
    for k in range(1, 6):
        row[f"c{k}"] = 100 * share.get(k, 0) / len(n_colours)
    return row


def print_table(rows, title, min_cards):
    print(f"\n{title}  (a colour counts at >= {min_cards} cards)")
    print(f"{'source':<26}{'1c':>7}{'2c':>7}{'3c':>7}{'4c':>7}{'5c':>7}"
          f"{'mean':>7}{'top2%':>8}{'gold':>7}{'clrls':>7}")
    print("-" * 90)
    for row in rows:
        print(f"{row['label']:<26}{row['c1']:>6.1f}%{row['c2']:>6.1f}%{row['c3']:>6.1f}%"
              f"{row['c4']:>6.1f}%{row['c5']:>6.1f}%{row['mean_colours']:>7.2f}"
              f"{row['top2_share']:>7.1f}%{row['multi']:>7.1f}{row['colourless']:>7.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FDN")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument("--rank-tag", default="diamond")
    parser.add_argument("--reward", default="mlp_400x2")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--n-drafts", type=int, default=1000)
    parser.add_argument("--opponent-temperature", type=float, default=0.5)
    parser.add_argument("--min-cards-pool", type=int, default=5)
    parser.add_argument("--min-cards-deck", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20250904)
    args = parser.parse_args()

    bank = PackBank(args.set_abbreviation, args.draft_mode)
    num_cards, picks = bank.num_cards, bank.pack_size * bank.n_boosters
    reward, _ = load_winrate_model(
        args.set_abbreviation, args.draft_mode, args.rank_tag, args.reward
    )
    draftnet = sd.load_draftnet(
        bank.cardnames, args.set_abbreviation, args.draft_mode, "data/models/"
    )
    policy = PickPolicy(bank.cardnames, dropout=0.1)
    policy.load_state_dict(torch.load(
        args.policy or f"data/rl_models/{args.set_abbreviation}_{args.draft_mode}_tabularasa_selfplay.pt",
        weights_only=True,
    ))
    policy.eval()

    cards = pd.read_csv(f"data/cards/{args.set_abbreviation}.csv").set_index("name")
    cards = cards.reindex(bank.cardnames)
    colour = cards["color_identity"].fillna("Colorless").to_numpy()

    print(f"{args.set_abbreviation}: RL policy in seat 0 against 7 DraftNet bots "
          f"(temperature {args.opponent_temperature}), {args.n_drafts} drafts\n")

    env = DraftEnv(bank, n_drafts=args.n_drafts, seed=args.seed)
    opponent = DraftNetPolicy(draftnet, args.opponent_temperature)
    run_drafts(env, [Adapter(policy)] + [opponent] * (N_SEATS - 1))
    collections = env.collections()

    rng = np.random.default_rng(args.seed)
    # One opponent seat per draft, so the DraftNet sample is the same size as the agent's.
    opponent_seat = rng.integers(1, N_SEATS, size=args.n_drafts)
    pools = {
        "RL policy (seat 0)": collections[:, 0],
        "DraftNet (1 seat/draft)": collections[torch.arange(args.n_drafts), torch.from_numpy(opponent_seat)],
    }
    dataset = torch.load(
        sd.winrate_dataset_path(args.set_abbreviation, args.draft_mode, "test", args.rank_tag),
        weights_only=False,
    )
    sequences = torch.from_numpy(dataset.pick_sequences[: args.n_drafts]).long()
    pools["real human pools"] = sd.build_collections(
        sequences, torch.full((len(sequences),), picks), num_cards
    )

    print(f"{'source':<26}{'reward_wr':>11}{'pool cards':>12}")
    print("-" * 49)
    for label, collection in pools.items():
        print(f"{label:<26}{score_pools(reward, collection, picks).mean():>11.4f}"
              f"{collection.sum(1).mean():>12.1f}")

    pool_rows = []
    for label, collection in pools.items():
        mono, multi, colourless = colour_counts(collection.numpy(), colour)
        pool_rows.append(summarize(label, mono, multi, colourless, args.min_cards_pool))
    print_table(pool_rows, "POOL colours (all 42 picks)", args.min_cards_pool)

    print("\nMean cards per colour in the pool:")
    print(f"{'source':<26}" + "".join(f"{c:>7}" for c in MONO) + f"{'gold':>7}{'clrls':>7}")
    for label, collection in pools.items():
        mono, multi, colourless = colour_counts(collection.numpy(), colour)
        print(f"{label:<26}" + "".join(f"{mono[:, i].mean():>7.1f}" for i in range(5))
              + f"{multi.mean():>7.1f}{colourless.mean():>7.1f}")

    # --- decks ------------------------------------------------------------
    builder = sdb.IterativeDeckBuilder(
        args.set_abbreviation, args.draft_mode,
        model_folder="data/models/", cards_folder="data/cards/",
    )
    deck_rows, deck_sizes = [], {}
    for label, collection in pools.items():
        built = np.zeros((len(collection), num_cards), dtype=np.float32)
        counts = collection.numpy()
        for i in range(len(collection)):
            pool_list = []
            for index in np.nonzero(counts[i])[0]:
                pool_list.extend([bank.cardnames[index]] * int(counts[i, index]))
            for name, n in builder.build_deck(pool_list)["deck_counts"].items():
                built[i, bank.cardnames.index(name)] = n
            if (i + 1) % 250 == 0:
                print(f"  built {i + 1}/{len(collection)} decks for {label}", flush=True)
        mono, multi, colourless = colour_counts(built, colour)
        deck_rows.append(summarize(label, mono, multi, colourless, args.min_cards_deck))
        deck_sizes[label] = built.sum(1).mean()

    print_table(deck_rows, "DECK colours (23 spells built from the pool)", args.min_cards_deck)
    print("\nMean deck size: " + "   ".join(f"{k}={v:.1f}" for k, v in deck_sizes.items()))


if __name__ == "__main__":
    main()
