"""
Combined view of the architecture comparisons, normalized by each dataset's noise ceiling.

Raw log loss is not comparable across rank filters - the populations have different base
winrates and different amounts of binomial noise. What is comparable is the fraction of
the achievable signal each model captures.

Run from the repo root:  python winrate_sandbox/summarize_results.py
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statisticaldrafting as sd


def noise_ceiling(set_abbreviation, draft_mode, rank_tag):
    """Method-of-moments ceiling on log loss gain and Pearson r for a test split."""
    dataset = torch.load(
        sd.winrate_dataset_path(set_abbreviation, draft_mode, "test", rank_tag),
        weights_only=False,
    )
    wins = dataset.wins.astype(np.float64)
    losses = dataset.losses.astype(np.float64)
    games = wins + losses
    observed = wins / games
    mean_winrate = wins.sum() / games.sum()

    weighted_sq = (games * (observed - mean_winrate) ** 2).sum()
    binomial = len(wins) * mean_winrate * (1 - mean_winrate)
    true_variance = max((weighted_sq - binomial) / games.sum(), 0.0)
    return {
        "drafts": len(wins),
        "true_sd": float(np.sqrt(true_variance)),
        "gain_ceiling": float(true_variance / (2 * mean_winrate * (1 - mean_winrate))),
        "r_ceiling": float(np.sqrt(min(true_variance / np.var(observed), 1.0))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FIN")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "diamond:winrate_sandbox/results/architecture_comparison.json",
            "allranks:winrate_sandbox/results/architecture_comparison_allranks.json",
        ],
        help="rank_tag:path pairs.",
    )
    args = parser.parse_args()

    for run in args.runs:
        rank_tag, path = run.split(":", 1)
        if not os.path.exists(path):
            print(f"(missing {path})")
            continue
        with open(path) as f:
            results = json.load(f)
        ceiling = noise_ceiling(args.set_abbreviation, args.draft_mode, rank_tag)

        print(f"\n=== {args.set_abbreviation} {args.draft_mode} [{rank_tag}] ===")
        print(
            f"{ceiling['drafts']} test drafts, true winrate sd={ceiling['true_sd']:.4f}, "
            f"gain ceiling={ceiling['gain_ceiling']:.5f}, r ceiling={ceiling['r_ceiling']:.4f}"
        )
        header = (
            f"{'architecture':<26}{'params':>11}{'gain':>9}{'% ceiling':>11}"
            f"{'gain@pool':>11}{'r@pool':>8}{'% r ceiling':>13}"
        )
        print(header)
        print("-" * len(header))
        for r in sorted(results, key=lambda x: -x["test_gain_mean"]):
            print(
                f"{r['architecture']:<26}{r['n_parameters']:>11,}"
                f"{r['test_gain_mean']:>9.5f}"
                f"{100 * r['test_gain_mean'] / ceiling['gain_ceiling']:>10.1f}%"
                f"{r['test_gain_pool_mean']:>11.5f}{r['test_r_pool_mean']:>8.4f}"
                f"{100 * r['test_r_pool_mean'] / ceiling['r_ceiling']:>12.1f}%"
            )


if __name__ == "__main__":
    main()
