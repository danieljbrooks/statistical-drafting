#!/usr/bin/env python3
"""
Train a 3-layer deckbuilding model and compare with the 2-layer baseline.
"""

import torch
from torch.utils.data import DataLoader
import statisticaldeckbuild as sdb

def main():
    print("=" * 80)
    print("TRAINING 3-LAYER DECKBUILDING MODEL")
    print("=" * 80)
    print()

    # Configuration
    SET_ABBREVIATION = "FDN"
    DRAFT_MODE = "Premier"
    BATCH_SIZE = 10000

    # Load existing datasets (don't recreate)
    print("Loading datasets...")
    train_path = f"../data/training_sets/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_train.pth"
    val_path = f"../data/training_sets/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_val.pth"

    train_dataset = torch.load(train_path, weights_only=False)
    val_dataset = torch.load(val_path, weights_only=False)

    print(f"  ✓ Training examples: {len(train_dataset):,}")
    print(f"  ✓ Validation examples: {len(val_dataset):,}")
    print(f"  ✓ Number of cards: {len(train_dataset.cardnames)}")
    print()

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create 3-layer network
    num_cards = len(train_dataset.cardnames)
    hidden_dims_3layer = [num_cards, 400, 400, 400]

    print("Creating 3-layer network...")
    print(f"  Architecture: {hidden_dims_3layer} → {num_cards}")
    print(f"  Total hidden layers: 3")
    print()

    network_3layer = sdb.DeckbuildNet(
        cardnames=train_dataset.cardnames,
        hidden_dims=hidden_dims_3layer,
        dropout_rate=0.6
    )

    # Count parameters
    total_params = sum(p.numel() for p in network_3layer.parameters())
    print(f"  Total parameters: {total_params:,}")
    print()

    # Train the model
    print("Starting training...")
    print("-" * 80)
    network_3layer, training_info = sdb.train_deckbuild_model(
        train_dataloader,
        val_dataloader,
        network_3layer,
        experiment_name=f"{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_3layer",
        learning_rate=0.03,
        patience=40,
    )

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation accuracy: {training_info['validation_accuracy']:.2f}%")
    print(f"Number of epochs: {training_info['num_epochs']}")
    print(f"Model saved to: ../data/models/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_3layer.pt")
    print()

    # Now evaluate on actual deckbuilding task
    print("=" * 80)
    print("EVALUATING ON ACTUAL DECKBUILDING (100 examples)")
    print("=" * 80)
    print()

    # Load the 3-layer model
    print("Testing 3-layer model...")
    builder_3layer = sdb.IterativeDeckBuilder(
        set_abbreviation=SET_ABBREVIATION,
        draft_mode=DRAFT_MODE,
        model_folder="../data/models/",
        cards_folder="../data/cards/",
    )
    # Replace with 3-layer model
    builder_3layer.network = network_3layer

    results_3layer = sdb.evaluate_deckbuilder(
        val_dataset=val_dataset,
        builder=builder_3layer,
        max_examples=100,
        verbose=False,
        save_results=True,
        output_path=f"evaluation_results/eval_{SET_ABBREVIATION}_{DRAFT_MODE}_3layer.json",
    )

    print()
    print("3-LAYER MODEL RESULTS:")
    print("-" * 80)
    print(f"Card accuracy: {results_3layer['card_accuracy']['accuracy_percentage']:.2f}%")
    print(f"Mean cards different: {results_3layer['summary_stats']['mean_cards_different']:.2f}")
    print()

    # Load and test the 2-layer baseline
    print("Testing 2-layer baseline model...")
    builder_2layer = sdb.IterativeDeckBuilder(
        set_abbreviation=SET_ABBREVIATION,
        draft_mode=DRAFT_MODE,
        model_folder="../data/models/",
        cards_folder="../data/cards/",
    )

    results_2layer = sdb.evaluate_deckbuilder(
        val_dataset=val_dataset,
        builder=builder_2layer,
        max_examples=100,
        verbose=False,
        save_results=True,
        output_path=f"evaluation_results/eval_{SET_ABBREVIATION}_{DRAFT_MODE}_2layer.json",
    )

    print()
    print("2-LAYER MODEL RESULTS:")
    print("-" * 80)
    print(f"Card accuracy: {results_2layer['card_accuracy']['accuracy_percentage']:.2f}%")
    print(f"Mean cards different: {results_2layer['summary_stats']['mean_cards_different']:.2f}")
    print()

    # Comparison
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)

    acc_diff = results_3layer['card_accuracy']['accuracy_percentage'] - results_2layer['card_accuracy']['accuracy_percentage']
    diff_diff = results_2layer['summary_stats']['mean_cards_different'] - results_3layer['summary_stats']['mean_cards_different']

    print(f"{'Metric':<30} {'2-Layer':>12} {'3-Layer':>12} {'Difference':>12}")
    print("-" * 80)
    print(f"{'Card Accuracy':<30} {results_2layer['card_accuracy']['accuracy_percentage']:>11.2f}% {results_3layer['card_accuracy']['accuracy_percentage']:>11.2f}% {acc_diff:>11.2f}%")
    print(f"{'Mean Cards Different':<30} {results_2layer['summary_stats']['mean_cards_different']:>12.2f} {results_3layer['summary_stats']['mean_cards_different']:>12.2f} {diff_diff:>12.2f}")
    print()

    if acc_diff > 0:
        print(f"✓ 3-layer model is BETTER by {acc_diff:.2f}% card accuracy")
    elif acc_diff < 0:
        print(f"✗ 3-layer model is WORSE by {abs(acc_diff):.2f}% card accuracy")
    else:
        print("= Models perform equally")

    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
