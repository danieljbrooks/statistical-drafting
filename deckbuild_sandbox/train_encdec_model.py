#!/usr/bin/env python3
"""
Train encoder/decoder deckbuilding model and compare with MLP baseline.
"""

import torch
from torch.utils.data import DataLoader
import statisticaldeckbuild as sdb

def main():
    print("=" * 80)
    print("TRAINING ENCODER/DECODER DECKBUILDING MODEL")
    print("=" * 80)
    print()

    # Configuration
    SET_ABBREVIATION = "FDN"
    DRAFT_MODE = "Premier"
    BATCH_SIZE = 10000

    # Load datasets
    train_path = f"../data/training_sets/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_train.pth"
    val_path = f"../data/training_sets/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_val.pth"

    print("Loading datasets...")
    train_dataset = torch.load(train_path, weights_only=False)
    val_dataset = torch.load(val_path, weights_only=False)

    print(f"  ✓ Training examples: {len(train_dataset):,}")
    print(f"  ✓ Validation examples: {len(val_dataset):,}")
    print(f"  ✓ Number of cards: {len(train_dataset.cardnames)}")
    print()

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create encoder/decoder network with attention
    embed_dim = 64
    num_heads = 4
    dropout_rate = 0.1

    print("Creating attention-based encoder/decoder network...")
    print(f"  Architecture:")
    print(f"    - Embedding dim: {embed_dim}")
    print(f"    - Attention heads: {num_heads}")
    print(f"    - Dropout rate: {dropout_rate}")
    print(f"    - Encoder: Multi-head attention pooling")
    print(f"    - Decoder: Direct dot product scoring")
    print()

    network = sdb.DeckbuildEncDecNet(
        cardnames=train_dataset.cardnames,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
    )

    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  (MLP baseline: ~387,000 parameters)")
    print(f"  Parameter reduction: {100 * (1 - total_params / 387000):.1f}%")
    print()

    # Train the model
    print("Starting training...")
    print("-" * 80)
    network, training_info = sdb.train_deckbuild_model(
        train_dataloader,
        val_dataloader,
        network,
        experiment_name=f"{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_encdec",
        learning_rate=0.01,  # Lower LR for smaller model
        patience=40,
        model_folder="../data/models/",
    )

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation accuracy: {training_info['validation_accuracy']:.2f}%")
    print(f"Number of epochs: {training_info['num_epochs']}")
    print(f"Model saved to: ../data/models/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_encdec.pt")
    print()

    # Compare with MLP baseline
    print("Baseline MLP validation accuracy: 71.42%")
    print(f"Encoder/decoder validation accuracy: {training_info['validation_accuracy']:.2f}%")
    diff = training_info['validation_accuracy'] - 71.42
    print(f"Difference: {diff:+.2f}%")
    print()

if __name__ == "__main__":
    main()
