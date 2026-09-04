"""
How much of the variation in match records is real, and how much is coin flips?

Method of moments on the held-out drafts: if every deck had the same true winrate, the
weighted spread of observed winrates would be exactly binomial. The excess over that is
the real spread in per-match win probability, which bounds how well ANY model of these
labels can do.

Run from the repo root:  python winrate_sandbox/noise_ceiling.py FIN
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FIN")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument("--split", default="test")
    parser.add_argument("--rank-tag", default="diamond")
    args = parser.parse_args()

    import statisticaldrafting as sd

    path = sd.winrate_dataset_path(
        args.set_abbreviation, args.draft_mode, args.split, args.rank_tag
    )
    dataset = torch.load(path, weights_only=False)
    wins = dataset.wins.astype(np.float64)
    losses = dataset.losses.astype(np.float64)
    games = wins + losses
    observed = wins / games

    n_drafts = len(wins)
    total_games = games.sum()
    mean_winrate = wins.sum() / total_games

    weighted_sq = (games * (observed - mean_winrate) ** 2).sum()
    binomial_expectation = n_drafts * mean_winrate * (1 - mean_winrate)
    true_variance = max((weighted_sq - binomial_expectation) / total_games, 0.0)
    true_sd = np.sqrt(true_variance)

    print(f"{args.set_abbreviation} {args.draft_mode} [{args.split}]")
    print(f"  drafts={n_drafts}  games={int(total_games)}  mean games/draft={games.mean():.2f}")
    print(f"  population winrate={mean_winrate:.4f}")
    print(f"  observed winrate sd (weighted)={np.sqrt(weighted_sq / total_games):.4f}")
    print(f"  binomial noise sd (per draft)={np.sqrt(binomial_expectation / total_games):.4f}")
    print(f"  implied TRUE winrate sd across drafts={true_sd:.4f}")

    # Ceiling on log loss gain for a perfect model of the true winrate.
    gain_ceiling = true_variance / (2 * mean_winrate * (1 - mean_winrate))
    print(f"  ceiling on log loss gain per game ~= {gain_ceiling:.5f} nats")

    # Ceiling on Pearson r against the noisy observed winrate.
    observed_variance = np.var(observed)
    r_ceiling = np.sqrt(min(true_variance / observed_variance, 1.0))
    print(f"  ceiling on pearson r vs observed winrate ~= {r_ceiling:.4f}")
    print(
        "\n  Note: this true spread mixes deck quality with player skill; a model that "
        "sees only the collection cannot reach all of it."
    )


if __name__ == "__main__":
    main()
