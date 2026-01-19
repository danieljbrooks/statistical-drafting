#!/usr/bin/env python3
"""
Recalculate 5-holdout validation accuracy as % of cards correctly predicted.
"""

import torch
from torch.utils.data import DataLoader
import statisticaldeckbuild as sdb

def evaluate_5holdout_properly(val_dataloader, network, device):
    """
    Evaluate 5-holdout model by counting how many of the 5 held-out cards
    are correctly predicted (not just at least 1).

    Returns the average % of held-out cards correctly predicted.
    """
    total_correct = 0
    total_holdout = 0

    network.eval()
    with torch.no_grad():
        for partial_deck, available, label in val_dataloader:
            # Move to device
            partial_deck = partial_deck.to(device)
            available = available.to(device)
            label = label.to(device)

            # Get model predictions
            predictions = network(partial_deck.float(), available.float())

            # For each example in batch
            for i in range(predictions.shape[0]):
                pred = predictions[i]
                lbl = label[i]
                avail = available[i]

                # Get the held-out card indices (should be 5 cards)
                actual_indices = torch.where(lbl > 0)[0]
                n_holdout = len(actual_indices)

                # Mask out unavailable cards
                masked_pred = pred.clone()
                masked_pred[avail == 0] = float('-inf')

                # Get top N predictions (where N = number of held-out cards)
                top_n_indices = torch.topk(masked_pred, k=n_holdout).indices

                # Count how many of the held-out cards are in the top N predictions
                num_correct_this_example = 0
                for idx in actual_indices:
                    if idx in top_n_indices:
                        num_correct_this_example += 1

                total_correct += num_correct_this_example
                total_holdout += n_holdout

    accuracy = 100 * total_correct / total_holdout if total_holdout > 0 else 0
    return accuracy, total_correct, total_holdout

def main():
    print("=" * 80)
    print("RECALCULATING 5-HOLDOUT TRAINING VALIDATION ACCURACY")
    print("=" * 80)
    print()

    SET_ABBREVIATION = "FDN"
    DRAFT_MODE = "Premier"
    BATCH_SIZE = 10000

    # Load the 5-holdout validation dataset
    print("Loading 5-holdout validation dataset...")
    val_path = f"../data/training_sets_5holdout/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_val.pth"
    val_dataset = torch.load(val_path, weights_only=False)
    print(f"  ✓ Validation examples: {len(val_dataset):,}")
    print(f"  ✓ Number of cards: {len(val_dataset.cardnames)}")
    print(f"  ✓ Cards held out per example: {val_dataset.n_holdout}")
    print()

    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load the trained 5-holdout model
    print("Loading 5-holdout model...")
    network = sdb.DeckbuildNet(
        cardnames=val_dataset.cardnames,
        hidden_dims=[len(val_dataset.cardnames), 400, 400],
        dropout_rate=0.6
    )
    network.load_state_dict(
        torch.load(f"../data/models/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_5holdout.pt",
                   weights_only=True)
    )

    device = sdb.get_device()
    network = network.to(device)
    network.eval()
    print(f"  ✓ Model loaded on device: {device}")
    print()

    # Old evaluation method (at least 1 correct)
    print("=" * 80)
    print("OLD METHOD: At least 1 of 5 correct")
    print("=" * 80)
    old_accuracy = sdb.evaluate_deckbuild_model(val_dataloader, network, device)
    print()

    # New evaluation method (average % correct)
    print("=" * 80)
    print("NEW METHOD: Average % of held-out cards correctly predicted")
    print("=" * 80)
    print()

    new_accuracy, total_correct, total_holdout = evaluate_5holdout_properly(
        val_dataloader, network, device
    )

    print(f"Total held-out cards: {total_holdout:,}")
    print(f"Correctly predicted: {total_correct:,}")
    print(f"Validation accuracy: {new_accuracy:.2f}%")
    print()

    # Comparison
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Method':<40} {'Accuracy':>12}")
    print("-" * 80)
    print(f"{'Old (at least 1 of 5 correct)':<40} {old_accuracy:>11.2f}%")
    print(f"{'New (avg % of 5 cards correct)':<40} {new_accuracy:>11.2f}%")
    print()
    print("The new method is more comparable to the 1-holdout model's 71.42%")
    print("since it measures the average recovery rate of held-out cards.")
    print()

if __name__ == "__main__":
    main()
