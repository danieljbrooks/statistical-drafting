#!/usr/bin/env python3
"""
Train a deckbuilding model with 5 cards held out instead of 1.
"""

import torch
from torch.utils.data import DataLoader
import statisticaldeckbuild as sdb

def main():
    print("=" * 80)
    print("TRAINING DECKBUILDING MODEL WITH 5 CARDS HELD OUT")
    print("=" * 80)
    print()

    # Configuration
    SET_ABBREVIATION = "FDN"
    DRAFT_MODE = "Premier"
    N_HOLDOUT = 5
    BATCH_SIZE = 10000

    # Create dataset with n_holdout=5
    print(f"Creating dataset with n_holdout={N_HOLDOUT}...")
    print("This will hold out 5 cards at a time during training")
    print()

    train_path, val_path = sdb.create_deckbuild_dataset(
        set_abbreviation=SET_ABBREVIATION,
        draft_mode=DRAFT_MODE,
        overwrite=True,  # Overwrite to create new dataset
        n_holdout=N_HOLDOUT,
        data_folder_17lands="../data/17lands/",
        data_folder_training_set="../data/training_sets_5holdout/",
        data_folder_cards="../data/cards/",
    )

    print()

    # Load datasets
    print("Loading datasets...")
    train_dataset = torch.load(train_path, weights_only=False)
    val_dataset = torch.load(val_path, weights_only=False)

    print(f"  ✓ Training examples: {len(train_dataset):,}")
    print(f"  ✓ Validation examples: {len(val_dataset):,}")
    print(f"  ✓ Number of cards: {len(train_dataset.cardnames)}")
    print(f"  ✓ Holdout cards: {N_HOLDOUT}")
    print()

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create network (same architecture as baseline)
    num_cards = len(train_dataset.cardnames)
    hidden_dims = [num_cards, 400, 400]

    print("Creating network...")
    print(f"  Architecture: {hidden_dims} → {num_cards}")
    print(f"  Total hidden layers: 2")
    print()

    network = sdb.DeckbuildNet(
        cardnames=train_dataset.cardnames,
        hidden_dims=hidden_dims,
        dropout_rate=0.6
    )

    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    print(f"  Total parameters: {total_params:,}")
    print()

    # Train the model
    print("Starting training...")
    print("-" * 80)
    network, training_info = sdb.train_deckbuild_model(
        train_dataloader,
        val_dataloader,
        network,
        experiment_name=f"{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_5holdout",
        learning_rate=0.03,
        patience=40,
        model_folder="../data/models/",
    )

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation accuracy: {training_info['validation_accuracy']:.2f}%")
    print(f"Number of epochs: {training_info['num_epochs']}")
    print(f"Model saved to: ../data/models/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_5holdout.pt")
    print()

    print("Note: This model was trained to predict 5 held-out cards at once.")
    print("      We'll evaluate it later on the actual deckbuilding task.")
    print()

if __name__ == "__main__":
    main()
