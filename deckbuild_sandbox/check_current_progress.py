#!/usr/bin/env python3
"""
Quick check of current model validation accuracy during training.
"""

import torch
from torch.utils.data import DataLoader
import statisticaldeckbuild as sdb

# Configuration
SET_ABBREVIATION = "FDN"
DRAFT_MODE = "Premier"

# Load validation dataset
val_path = f"../data/training_sets/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_val.pth"
val_dataset = torch.load(val_path, weights_only=False)
val_dataloader = DataLoader(val_dataset, batch_size=10000, shuffle=False)

# Load current model checkpoint
network = sdb.DeckbuildEncDecNet(
    cardnames=val_dataset.cardnames,
    embed_dim=64,
    num_heads=4,
    dropout_rate=0.1,
)

try:
    network.load_state_dict(
        torch.load(f"../data/models/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_encdec.pt",
                   weights_only=True)
    )
    print("Model loaded successfully!")
    print()

    # Quick evaluation
    accuracy = sdb.evaluate_deckbuild_model(val_dataloader, network)

    print()
    print(f"Current best validation accuracy: {accuracy:.2f}%")
    print(f"MLP baseline: 71.42%")
    print(f"Difference: {accuracy - 71.42:+.2f}%")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Training may still be in early stages...")
