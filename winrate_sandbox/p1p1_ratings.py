"""
Estimated winrate of a one-card pool, for every card in a set.

Scores each card on its own - a collection holding just that card, one pick in - with a
trained winrate model, and ranks the set by the result.

Run from the repo root:
    python winrate_sandbox/p1p1_ratings.py FDN
    python winrate_sandbox/p1p1_ratings.py FIN --all
    python winrate_sandbox/p1p1_ratings.py FDN --csv fdn_p1p1.csv

Note on support: the `p1p1s` column counts how often each card was actually taken first
in the training data. Cards nobody first-picks never appear as a one-card pool during
training, so their estimates are extrapolation from the rest of the draft and should be
read as much softer than the well-supported rows. Use --min-support to hide them.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statisticaldrafting as sd
from predict_pool import load_model


def one_card_winrates(network, cardnames, device, batch_size=2048):
    """Predicted winrate of the pool {card}, for every card, at one pick in."""
    num_cards = len(cardnames)
    predictions = np.zeros(num_cards, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, num_cards, batch_size):
            stop = min(start + batch_size, num_cards)
            collection = torch.zeros(stop - start, num_cards, device=device)
            collection[torch.arange(stop - start), torch.arange(start, stop)] = 1.0
            n_picks = torch.ones(stop - start, dtype=torch.long, device=device)
            predictions[start:stop] = (
                torch.sigmoid(network(collection, n_picks)).cpu().numpy()
            )
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FDN")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument("--rank-tag", default="diamond")
    parser.add_argument("--architecture", default="dn_all_features")
    parser.add_argument("--model", default=None, help="Exact model name, overriding the above.")
    parser.add_argument("--top", type=int, default=25, help="Rows to show at each end.")
    parser.add_argument(
        "--min-support", type=int, default=0,
        help="Hide cards taken first fewer than this many times in training.",
    )
    parser.add_argument("--all", action="store_true", help="Print every card.")
    parser.add_argument("--csv", default=None, help="Write the full table here.")
    args = parser.parse_args()

    model_name = args.model or (
        f"{args.set_abbreviation}_{args.draft_mode}_{args.rank_tag}_{args.architecture}"
    )
    device = sd.get_device()
    network, sidecar = load_model(model_name, device)
    cardnames = sidecar["cardnames"]

    winrates = one_card_winrates(network, cardnames, device)

    # How often each card was actually a real pack-1 pick-1 in the training split.
    train_dataset = torch.load(
        sd.winrate_dataset_path(
            sidecar["set"], sidecar["draft_mode"], "train", sidecar["rank_tag"]
        ),
        weights_only=False,
    )
    first_picks = train_dataset.pick_sequences[:, 0]
    support = np.bincount(first_picks, minlength=len(cardnames) + 1)[: len(cardnames)]

    table = pd.DataFrame({"card": cardnames, "winrate": winrates, "p1p1s": support})
    card_csv = f"data/cards/{sidecar['set']}.csv"
    if os.path.exists(card_csv):
        info = pd.read_csv(card_csv)[["name", "rarity", "color_identity"]]
        table = table.merge(info, left_on="card", right_on="name", how="left")
        table = table.drop(columns=["name"])
    table = table.sort_values("winrate", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    shown = table[table["p1p1s"] >= args.min_support]

    print(f"\nModel: {sidecar['name']}")
    print(
        f"  trained on {sidecar['set']} {sidecar['draft_mode']} [{sidecar['rank_tag']}], "
        f"population winrate {sidecar['base_winrate']:.4f}"
    )
    print(
        f"  {len(cardnames)} cards, one-card-pool winrates span "
        f"{winrates.min():.4f} to {winrates.max():.4f} (sd {winrates.std():.4f})"
    )
    never = int((support == 0).sum())
    print(
        f"  {never} cards ({100 * never / len(cardnames):.0f}%) were never taken first in "
        f"the {len(first_picks)} training drafts;\n  their estimates are extrapolated - "
        "see the p1p1s column."
    )
    if args.min_support:
        print(f"  showing only cards with at least {args.min_support} real first picks")

    header = (
        f"{'rank':>6}  {'winrate':<9}{'p1p1s':>6}  {'card':<42}{'rarity':<10}color"
    )

    def show(rows):
        for row in rows.itertuples():
            rarity = (row.rarity or "")[:8] if hasattr(row, "rarity") else ""
            color = (row.color_identity or "") if hasattr(row, "color_identity") else ""
            print(
                f"  {row.rank:>4}  {row.winrate:.4f} {row.p1p1s:>7}  "
                f"{row.card:<42}{rarity:<10}{color}"
            )

    if args.all:
        print("\n" + header)
        show(shown)
    else:
        print(f"\nBest one-card pools:\n{header}")
        show(shown.head(args.top))
        print(f"\nWorst one-card pools:\n{header}")
        show(shown.tail(args.top))

    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nWrote all {len(table)} cards to {args.csv}")


if __name__ == "__main__":
    main()
