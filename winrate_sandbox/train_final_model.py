"""
Train one winrate model and save it with everything needed to use it later.

Writes ``data/winrate_models/<SET>_<mode>_<rank_tag>_<architecture>.pt`` plus a JSON
sidecar holding the cardnames, regularization and base winrate, so predict_pool.py can
rebuild the network without being told how it was configured.

Run from the repo root:
    python winrate_sandbox/train_final_model.py FIN dn_all_features --rank-tag diamond
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statisticaldrafting as sd
from architectures import (
    build_network, cacheable_feature_modes, learning_rate, train_overrides,
)

MODEL_FOLDER = "data/winrate_models/"


def model_name(set_abbreviation, draft_mode, rank_tag, architecture):
    return f"{set_abbreviation}_{draft_mode}_{rank_tag}_{architecture}"


def sidecar_path(name):
    return os.path.join(MODEL_FOLDER, name + ".json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FIN")
    parser.add_argument("architecture", nargs="?", default="dn_all_features")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument("--rank-tag", default="diamond")
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--input-dropout", type=float, default=0.4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
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
    cardnames = datasets["train"].cardnames
    picks_per_draft = datasets["train"].picks_per_draft
    base_winrate = float(
        (data["train"].wins.sum() / (data["train"].wins + data["train"].losses).sum()).item()
    )

    feature_modes = cacheable_feature_modes(args.architecture)
    if feature_modes is not None:
        module = sd.DraftNetFeatures(
            cardnames, args.set_abbreviation, args.draft_mode, feature_modes
        )
        for split_data in data.values():
            split_data.extra_features = sd.precompute_features(split_data, module)

    network = build_network(
        args.architecture, cardnames, picks_per_draft, base_winrate,
        args.dropout, args.input_dropout, args.set_abbreviation, args.draft_mode,
    )
    name = model_name(
        args.set_abbreviation, args.draft_mode, args.rank_tag, args.architecture
    )
    settings = dict(
        learning_rate=learning_rate(args.architecture),
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        eval_every_batches=args.eval_every,
        val_n_picks=list(range(1, picks_per_draft + 1, 3)),
        experiment_name=name,
        model_folder=MODEL_FOLDER,
        seed=args.seed,
    )
    settings.update(train_overrides(args.architecture, picks_per_draft))
    sd.train_winrate_model(network, data["train"], data["val"], **settings)

    test_metrics = sd.evaluate_model(
        network, data["test"], base_winrate,
        report_picks=[1, 7, 14, 21, 28, 35, picks_per_draft], verbose=True,
    )

    # Reference distribution of predictions over held-out completed pools, so a single
    # pool can be placed against the population without reloading the dataset.
    final = sd.predict(network, data["test"], [picks_per_draft])
    percentiles = list(range(1, 100))
    reference = {
        str(q): float(value)
        for q, value in zip(percentiles, np.percentile(final["p"], percentiles))
    }

    sidecar = {
        "name": name,
        "set": args.set_abbreviation,
        "draft_mode": args.draft_mode,
        "rank_tag": args.rank_tag,
        "architecture": args.architecture,
        "dropout": args.dropout,
        "input_dropout": args.input_dropout,
        "picks_per_draft": picks_per_draft,
        "base_winrate": base_winrate,
        "cardnames": cardnames,
        "test_metrics": {
            key: value for key, value in test_metrics.items() if key != "by_n_picks"
        },
        "prediction_percentiles": reference,
    }
    with open(sidecar_path(name), "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"\nSaved {MODEL_FOLDER}{name}.pt and its sidecar {sidecar_path(name)}")


if __name__ == "__main__":
    main()
