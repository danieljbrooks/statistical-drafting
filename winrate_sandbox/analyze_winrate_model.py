"""
Detailed report for one trained winrate model: calibration, per-pick-count skill,
and the spread of expected match wins it predicts.

Run from the repo root:
    python winrate_sandbox/analyze_winrate_model.py FIN mlp_1024x2
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statisticaldrafting as sd
from architectures import build_network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FIN")
    parser.add_argument("architecture", nargs="?", default="mlp_1024x2")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--input-dropout", type=float, default=0.2)
    args = parser.parse_args()

    prefix = f"data/winrate_sets/{args.set_abbreviation}_{args.draft_mode}_winrate"
    train_dataset = torch.load(f"{prefix}_train.pth", weights_only=False)
    eval_dataset = torch.load(f"{prefix}_{args.split}.pth", weights_only=False)

    device = sd.get_device()
    train_data = sd.DeviceTensors(train_dataset, device)
    eval_data = sd.DeviceTensors(eval_dataset, device)
    picks_per_draft = train_dataset.picks_per_draft
    base_winrate = float(
        (train_data.wins.sum() / (train_data.wins + train_data.losses).sum()).item()
    )

    network = build_network(
        args.architecture, train_dataset.cardnames, picks_per_draft, base_winrate,
        args.dropout, args.input_dropout,
    ).to(device)
    weights_path = f"data/winrate_models/{args.set_abbreviation}_{args.architecture}.pt"
    network.load_state_dict(torch.load(weights_path))
    network.eval()
    print(f"Loaded {weights_path}; reporting on the {args.split} split.\n")

    # Skill as a function of how far into the draft we are.
    metrics = sd.evaluate_model(
        network, eval_data, base_winrate, report_picks=list(range(1, picks_per_draft + 1))
    )
    print(f"{'picks':>6}{'NLL/game':>11}{'gain':>10}{'pearson r':>12}{'pred sd':>10}")
    print("-" * 49)
    for n in range(1, picks_per_draft + 1):
        row = metrics["by_n_picks"][n]
        print(
            f"{n:>6}{row['log_loss_per_game']:>11.5f}"
            f"{row['baseline_log_loss_per_game'] - row['log_loss_per_game']:>10.5f}"
            f"{row['pearson_r']:>12.4f}{row['skill_spread']:>10.4f}"
        )

    # Calibration on the completed pool.
    out = sd.predict(network, eval_data, [picks_per_draft])
    p, wins, losses = out["p"], out["wins"], out["losses"]
    expected_wins = sd.expected_match_wins(torch.from_numpy(p)).numpy()

    print(f"\nCalibration on full {picks_per_draft}-card pools (held-out decks):")
    print(f"{'decile':>7}{'decks':>8}{'predicted':>12}{'actual':>10}{'games':>9}{'E[wins]':>10}")
    print("-" * 56)
    order = np.argsort(p)
    for i, chunk in enumerate(np.array_split(order, 10)):
        games = (wins[chunk] + losses[chunk]).sum()
        print(
            f"{i + 1:>7}{len(chunk):>8}{p[chunk].mean():>12.4f}"
            f"{wins[chunk].sum() / games:>10.4f}{int(games):>9}"
            f"{expected_wins[chunk].mean():>10.2f}"
        )

    print(
        f"\nPredicted winrate: mean={p.mean():.4f} sd={p.std():.4f} "
        f"p5={np.percentile(p, 5):.4f} p95={np.percentile(p, 95):.4f}"
    )
    print(
        f"Predicted expected match wins: mean={expected_wins.mean():.2f} "
        f"p5={np.percentile(expected_wins, 5):.2f} p95={np.percentile(expected_wins, 95):.2f}"
    )
    actual_wins = wins.mean()
    print(f"Actual mean match wins in held-out set: {actual_wins:.2f}")

    # How well do predictions rank decks by realized wins?
    top = p >= np.percentile(p, 90)
    bottom = p <= np.percentile(p, 10)
    print(
        f"\nTop decile pools: {wins[top].mean():.2f} actual wins, "
        f"{wins[top].sum() / (wins[top] + losses[top]).sum():.4f} winrate"
    )
    print(
        f"Bottom decile pools: {wins[bottom].mean():.2f} actual wins, "
        f"{wins[bottom].sum() / (wins[bottom] + losses[bottom]).sum():.4f} winrate"
    )


if __name__ == "__main__":
    main()
