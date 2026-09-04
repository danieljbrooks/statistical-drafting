"""
Compare architectures for predicting deck winrate from a draft collection.

For each architecture a small regularization grid is run on the validation split;
the best setting is then retrained over several seeds and scored on the untouched
test split, so the reported numbers are not the ones that were selected on.

Run from the repo root:
    python winrate_sandbox/run_architecture_comparison.py FIN
"""
import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statisticaldrafting as sd
from architectures import (
    ARCHITECTURES, build_network, cacheable_feature_modes, learning_rate, train_overrides,
)

# (dropout, input_dropout, weight_decay) - collections are dropped card-by-card because
# there are only ~22k labelled drafts to fit.
REGULARIZATION_GRID = [
    (0.3, 0.0, 0.0),
    (0.5, 0.2, 1e-4),
    (0.7, 0.4, 1e-3),
    (0.8, 0.5, 3e-3),
]


def run_one(name, cardnames, picks_per_draft, base_winrate, regularization,
            train_data, val_data, seed, args):
    dropout, input_dropout, weight_decay = regularization
    network = build_network(
        name, cardnames, picks_per_draft, base_winrate, dropout, input_dropout,
        args.set_abbreviation, args.draft_mode,
    )
    settings = dict(
        learning_rate=learning_rate(name),
        weight_decay=weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        eval_every_batches=args.eval_every,
        val_n_picks=args.val_n_picks,
        experiment_name=(
            f"{args.set_abbreviation}_{args.draft_mode}_{args.rank_tag}_{name}"
        ),
        model_folder="data/winrate_models/",
        seed=seed,
        verbose=args.verbose,
    )
    settings.update(train_overrides(name, picks_per_draft))
    info = sd.train_winrate_model(network, train_data, val_data, **settings)
    return network, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FIN")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument("--rank-tag", default="diamond", help='Rank filter the dataset was built with ("diamond" or "allranks").')
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=30, help="Evaluations without improvement.")
    parser.add_argument("--eval-every", type=int, default=25, help="Batches between evaluations.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--only", default=None, help="Comma separated architecture names.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--results", default="winrate_sandbox/results/architecture_comparison.json")
    args = parser.parse_args()

    datasets = {
        split: torch.load(
            sd.winrate_dataset_path(
                args.set_abbreviation, args.draft_mode, split, args.rank_tag
            ),
            weights_only=False,
        )
        for split in ["train", "val", "test"]
    }
    device = sd.get_device()
    data = {split: sd.DeviceTensors(ds, device) for split, ds in datasets.items()}
    train_data, val_data, test_data = data["train"], data["val"], data["test"]

    picks_per_draft = datasets["train"].picks_per_draft
    cardnames = datasets["train"].cardnames
    base_winrate = float(
        (train_data.wins.sum() / (train_data.wins + train_data.losses).sum()).item()
    )
    # Model selection uses a subset of prefix lengths, purely to keep evaluation cheap.
    args.val_n_picks = list(range(1, picks_per_draft + 1, 3))
    report_picks = [1, 7, 14, 21, 28, 35, picks_per_draft]

    print(
        f"{args.set_abbreviation} {args.draft_mode}: "
        f"{train_data.n_drafts} train / {val_data.n_drafts} val / {test_data.n_drafts} test drafts, "
        f"{train_data.num_cards} cards, {picks_per_draft} picks, "
        f"base winrate={round(base_winrate, 4)}"
    )

    chosen = list(ARCHITECTURES)
    if args.only:
        wanted = set(args.only.split(","))
        chosen = [name for name in chosen if name in wanted]

    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    results = []
    if os.path.exists(args.results):
        with open(args.results) as f:
            results = json.load(f)

    for name in chosen:
        print(f"\n########## {name} ##########", flush=True)

        # Frozen DraftNet features do not change during training, so compute them once
        # per architecture instead of once per batch (about 5x faster end to end).
        feature_modes = cacheable_feature_modes(name)
        for split_data in data.values():
            split_data.extra_features = None
        sd.free_device_memory(device)
        if feature_modes is not None:
            module = sd.DraftNetFeatures(
                cardnames, args.set_abbreviation, args.draft_mode, feature_modes
            )
            for split_data in data.values():
                split_data.extra_features = sd.precompute_features(split_data, module)
            print(f"  cached {module.output_dim} frozen features per collection", flush=True)

        # 1. Pick regularization on the validation split.
        best_regularization, best_val = None, float("inf")
        for regularization in REGULARIZATION_GRID:
            _, info = run_one(name, cardnames, picks_per_draft, base_winrate,
                              regularization, train_data, val_data, 0, args)
            val_log_loss = info["metrics"]["log_loss_per_game"]
            print(
                f"  dropout={regularization[0]} input_dropout={regularization[1]} "
                f"wd={regularization[2]}  val NLL/game={round(val_log_loss, 5)}",
                flush=True,
            )
            if val_log_loss < best_val:
                best_val, best_regularization = val_log_loss, regularization

        # 2. Retrain the winner over several seeds and score on the test split.
        seed_metrics, last_info = [], None
        for seed in range(args.seeds):
            network, last_info = run_one(name, cardnames, picks_per_draft, base_winrate,
                                         best_regularization, train_data, val_data,
                                         seed, args)
            test_metrics = sd.evaluate_model(
                network, test_data, base_winrate, report_picks=report_picks
            )
            out = sd.predict(network, test_data, [picks_per_draft])
            test_metrics["calibration_full_pool"] = sd.calibration_table(
                out["p"], out["wins"], out["losses"]
            )
            seed_metrics.append(test_metrics)
            print(
                f"  seed {seed}: test NLL/game={round(test_metrics['log_loss_per_game'], 5)} "
                f"gain={round(test_metrics['log_loss_improvement'], 5)} "
                f"r={round(test_metrics['pearson_r'], 4)}",
                flush=True,
            )

        record = {
            "architecture": name,
            "set": args.set_abbreviation,
            "draft_mode": args.draft_mode,
            "n_parameters": last_info["n_parameters"],
            "loss": last_info["loss"],
            "learning_rate": learning_rate(name),
            "dropout": best_regularization[0],
            "input_dropout": best_regularization[1],
            "weight_decay": best_regularization[2],
            "val_log_loss": best_val,
            "baseline_winrate": base_winrate,
            "test_metrics_by_seed": seed_metrics,
            "test_log_loss_mean": statistics.mean(m["log_loss_per_game"] for m in seed_metrics),
            "test_log_loss_sd": (
                statistics.stdev(m["log_loss_per_game"] for m in seed_metrics)
                if len(seed_metrics) > 1 else 0.0
            ),
            "test_gain_mean": statistics.mean(m["log_loss_improvement"] for m in seed_metrics),
            "test_r_mean": statistics.mean(m["pearson_r"] for m in seed_metrics),
            "test_r_pool_mean": statistics.mean(
                m["by_n_picks"][picks_per_draft]["pearson_r"] for m in seed_metrics
            ),
            "test_gain_pool_mean": statistics.mean(
                m["by_n_picks"][picks_per_draft]["baseline_log_loss_per_game"]
                - m["by_n_picks"][picks_per_draft]["log_loss_per_game"]
                for m in seed_metrics
            ),
            "test_sd_pool_mean": statistics.mean(
                m["by_n_picks"][picks_per_draft]["skill_spread"] for m in seed_metrics
            ),
        }
        sd.free_device_memory(device)
        results = [r for r in results if r["architecture"] != name]
        results.append(record)
        with open(args.results, "w") as f:
            json.dump(results, f, indent=2)

    print_summary(results)


def print_summary(results):
    header = (
        f"{'architecture':<26}{'params':>11}{'drop':>6}{'in-drop':>8}"
        f"{'test NLL/game':>15}{'±sd':>9}{'gain':>9}{'r(all)':>8}"
        f"{'gain@pool':>11}{'r@pool':>8}{'sd@pool':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x["test_log_loss_mean"]):
        print(
            f"{r['architecture']:<26}{r['n_parameters']:>11,}{r['dropout']:>6}"
            f"{r['input_dropout']:>8}{r['test_log_loss_mean']:>15.5f}"
            f"{r['test_log_loss_sd']:>9.5f}{r['test_gain_mean']:>9.5f}"
            f"{r['test_r_mean']:>8.4f}{r['test_gain_pool_mean']:>11.5f}"
            f"{r['test_r_pool_mean']:>8.4f}{r['test_sd_pool_mean']:>9.4f}"
        )


if __name__ == "__main__":
    main()
