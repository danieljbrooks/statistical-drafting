import json
import math
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .winratedataset import CollectionWinrateDataset, build_collections


def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DeviceTensors:
    """Whole dataset held on device - it is small enough that a DataLoader is pure overhead."""

    def __init__(self, dataset: CollectionWinrateDataset, device: torch.device):
        self.pick_sequences = torch.from_numpy(dataset.pick_sequences).to(device)
        self.wins = torch.from_numpy(dataset.wins).to(device)
        self.losses = torch.from_numpy(dataset.losses).to(device)
        self.n_picks_options = torch.tensor(
            dataset.n_picks_options, dtype=torch.long, device=device
        )
        self.num_cards = len(dataset.cardnames)
        self.n_drafts = dataset.n_drafts
        self.device = device


def iterate_batches(
    data: DeviceTensors,
    batch_size: int,
    shuffle: bool,
    n_picks_values: Optional[List[int]] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Yields (collection, n_picks, wins, losses) batches over all (draft, n_picks) pairs.
    """
    options = (
        data.n_picks_options
        if n_picks_values is None
        else torch.tensor(n_picks_values, dtype=torch.long, device=data.device)
    )
    n_options = options.shape[0]
    total = data.n_drafts * n_options

    if shuffle:
        order = torch.randperm(total, device=data.device, generator=generator)
    else:
        order = torch.arange(total, device=data.device)

    for start in range(0, total, batch_size):
        flat = order[start : start + batch_size]
        draft_index = flat // n_options
        n_picks = options[flat % n_options]
        sequences = data.pick_sequences[draft_index]
        collection = build_collections(sequences, n_picks, data.num_cards)
        yield collection, n_picks, data.wins[draft_index], data.losses[draft_index]


def binomial_nll(logits, wins, losses):
    """
    Negative log likelihood per game of a match record under a per-match win probability.

    Arena events stop at a fixed number of wins or losses; that stopping rule does not
    depend on p, so the record's likelihood is still p^wins * (1-p)^losses up to a
    constant, and this is the correct objective.
    """
    nll = -(wins * F.logsigmoid(logits) + losses * F.logsigmoid(-logits))
    return nll.sum() / (wins + losses).sum()


def winrate_mse(logits, wins, losses):
    """Alternative objective: squared error against the observed (noisy) winrate."""
    observed = wins / (wins + losses)
    return F.mse_loss(torch.sigmoid(logits), observed)


LOSS_FUNCTIONS = {"binomial_nll": binomial_nll, "mse": winrate_mse}


@torch.no_grad()
def predict(network, data: DeviceTensors, n_picks_values: List[int], batch_size: int = 16384):
    """Returns numpy arrays of (predicted p, wins, losses, n_picks) over the given prefixes."""
    network.eval()
    chunks = {"p": [], "wins": [], "losses": [], "n_picks": []}
    for collection, n_picks, wins, losses in iterate_batches(
        data, batch_size, shuffle=False, n_picks_values=n_picks_values
    ):
        logits = network(collection, n_picks)
        chunks["p"].append(torch.sigmoid(logits).float().cpu())
        chunks["wins"].append(wins.cpu())
        chunks["losses"].append(losses.cpu())
        chunks["n_picks"].append(n_picks.cpu())
    return {k: torch.cat(v).numpy() for k, v in chunks.items()}


def _log_loss_per_game(p, wins, losses, eps: float = 1e-7):
    p = np.clip(p, eps, 1 - eps)
    nll = -(wins * np.log(p) + losses * np.log(1 - p))
    return float(nll.sum() / (wins + losses).sum())


def evaluate_model(
    network,
    data: DeviceTensors,
    baseline_winrate: float,
    n_picks_values: Optional[List[int]] = None,
    report_picks: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict:
    """
    Evaluate on validation collections.

    Primary metric is log loss per game against the observed match records; the
    baseline is predicting the population winrate for every deck. ``skill_spread`` is
    the standard deviation of predicted winrates, i.e. how much the model believes
    decks actually differ.
    """
    if n_picks_values is None:
        n_picks_values = list(data.n_picks_options.cpu().numpy())
    out = predict(network, data, n_picks_values)
    p, wins, losses, n_picks = out["p"], out["wins"], out["losses"], out["n_picks"]

    observed = wins / (wins + losses)
    log_loss = _log_loss_per_game(p, wins, losses)
    baseline_log_loss = _log_loss_per_game(
        np.full_like(p, baseline_winrate), wins, losses
    )

    metrics = {
        "log_loss_per_game": log_loss,
        "baseline_log_loss_per_game": baseline_log_loss,
        "log_loss_improvement": baseline_log_loss - log_loss,
        "pearson_r": float(np.corrcoef(p, observed)[0, 1]),
        "rmse_vs_observed": float(np.sqrt(np.mean((p - observed) ** 2))),
        "skill_spread": float(np.std(p)),
        "mean_prediction": float(np.mean(p)),
    }

    # Per-prefix-length breakdown.
    if report_picks is None:
        report_picks = [n for n in [1, 7, 14, 21, 28, 35, 42] if n in set(n_picks_values)]
    by_picks = {}
    for n in report_picks:
        mask = n_picks == n
        if mask.sum() == 0:
            continue
        by_picks[int(n)] = {
            "log_loss_per_game": _log_loss_per_game(p[mask], wins[mask], losses[mask]),
            "baseline_log_loss_per_game": _log_loss_per_game(
                np.full(int(mask.sum()), baseline_winrate), wins[mask], losses[mask]
            ),
            "pearson_r": float(np.corrcoef(p[mask], observed[mask])[0, 1]),
            "skill_spread": float(np.std(p[mask])),
        }
    metrics["by_n_picks"] = by_picks

    if verbose:
        print(
            f"  val log loss/game={round(log_loss, 5)} "
            f"(baseline {round(baseline_log_loss, 5)}, "
            f"improvement {round(baseline_log_loss - log_loss, 5)})  "
            f"r={round(metrics['pearson_r'], 4)}  "
            f"pred sd={round(metrics['skill_spread'], 4)}"
        )
    return metrics


def calibration_table(p, wins, losses, n_bins: int = 10) -> List[Dict]:
    """Bucket predictions into quantile bins and compare predicted vs realized winrate."""
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, n_bins):
        games = (wins[chunk] + losses[chunk]).sum()
        rows.append(
            {
                "n_decks": int(len(chunk)),
                "mean_predicted": float(p[chunk].mean()),
                "actual_winrate": float(wins[chunk].sum() / games),
                "games": int(games),
            }
        )
    return rows


def train_winrate_model(
    network,
    train_data: DeviceTensors,
    val_data: DeviceTensors,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 4096,
    max_epochs: int = 200,
    patience: int = 10,
    eval_every_batches: int = 50,
    loss_name: str = "binomial_nll",
    lr_gamma: float = 0.97,
    experiment_name: str = "winrate_test",
    model_folder: str = "data/winrate_models/",
    train_n_picks: Optional[List[int]] = None,
    val_n_picks: Optional[List[int]] = None,
    verbose: bool = True,
    seed: int = 0,
) -> Dict:
    """
    Train a winrate model with early stopping on validation log loss per game.

    Evaluation runs every ``eval_every_batches`` optimizer steps rather than once per
    epoch: with few labelled drafts the wider networks overfit well inside a single
    epoch, so per-epoch checkpoints miss the best model entirely.
    """
    torch.manual_seed(seed)
    device = train_data.device
    network = network.to(device)

    loss_fn = LOSS_FUNCTIONS[loss_name]
    optimizer = torch.optim.Adam(
        network.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

    baseline_winrate = float(
        (train_data.wins.sum() / (train_data.wins + train_data.losses).sum()).item()
    )

    os.makedirs(model_folder, exist_ok=True)
    weights_path = os.path.join(model_folder, experiment_name + ".pt")

    n_params = sum(param.numel() for param in network.parameters())
    if verbose:
        print(
            f"\n=== {experiment_name} === {n_params:,} params, loss={loss_name}, "
            f"lr={learning_rate}, batch={batch_size}, device={device}"
        )

    best = evaluate_model(
        network, val_data, baseline_winrate, val_n_picks, verbose=verbose
    )
    best_log_loss, best_step, best_metrics = best["log_loss_per_game"], 0, best
    best_epoch = -1
    torch.save(network.state_dict(), weights_path)

    t0 = time.time()
    step, evals_since_best, stop = 0, 0, False
    for epoch in range(max_epochs):
        network.train()
        epoch_loss, n_batches = 0.0, 0
        for collection, n_picks, wins, losses in iterate_batches(
            train_data, batch_size, shuffle=True, n_picks_values=train_n_picks
        ):
            optimizer.zero_grad()
            logits = network(collection, n_picks)
            loss = loss_fn(logits, wins, losses)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            step += 1

            if step % eval_every_batches == 0:
                metrics = evaluate_model(network, val_data, baseline_winrate, val_n_picks)
                network.train()
                if metrics["log_loss_per_game"] < best_log_loss - 1e-6:
                    best_log_loss, best_step, best_metrics = (
                        metrics["log_loss_per_game"],
                        step,
                        metrics,
                    )
                    best_epoch, evals_since_best = epoch, 0
                    torch.save(network.state_dict(), weights_path)
                else:
                    evals_since_best += 1
                    if evals_since_best >= patience:
                        stop = True
                        break
        scheduler.step()

        if verbose:
            print(
                f"epoch {epoch:3d} step {step:5d}  train {loss_name}="
                f"{round(epoch_loss / max(n_batches, 1), 5)}  "
                f"best val log loss/game={round(best_log_loss, 5)}  "
                f"r={round(best_metrics['pearson_r'], 4)}  "
                f"pred sd={round(best_metrics['skill_spread'], 4)}  "
                f"lr={round(scheduler.get_last_lr()[0], 6)}"
            )
        if stop:
            break

    network.load_state_dict(torch.load(weights_path))
    train_time = time.time() - t0

    # Calibration on the full 42-card pools.
    final_picks = [int(train_data.pick_sequences.shape[1])]
    final_out = predict(network, val_data, final_picks)
    calibration = calibration_table(
        final_out["p"], final_out["wins"], final_out["losses"]
    )

    info = {
        "experiment_name": experiment_name,
        "n_parameters": n_params,
        "loss": loss_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "eval_every_batches": eval_every_batches,
        "train_drafts": train_data.n_drafts,
        "val_drafts": val_data.n_drafts,
        "baseline_winrate": baseline_winrate,
        "train_seconds": round(train_time, 1),
        "metrics": best_metrics,
        "calibration_full_pool": calibration,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if verbose:
        print(
            f"Done in {round(train_time)}s. Best epoch {best_epoch} (step {best_step}), "
            f"val log loss/game={round(best_log_loss, 5)} "
            f"(baseline {round(best_metrics['baseline_log_loss_per_game'], 5)})"
        )
    return info
