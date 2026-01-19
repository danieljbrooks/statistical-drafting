#!/usr/bin/env python3
"""
Debug the attention model to see if there are any issues.
"""

import torch
import statisticaldeckbuild as sdb

# Load validation dataset
val_path = "../data/training_sets/FDN_Premier_deckbuild_val.pth"
val_dataset = torch.load(val_path, weights_only=False)

# Create model
network = sdb.DeckbuildEncDecNet(
    cardnames=val_dataset.cardnames,
    embed_dim=64,
    num_heads=4,
    dropout_rate=0.1,
)

# Load weights
network.load_state_dict(
    torch.load("../data/models/FDN_Premier_deckbuild_encdec.pt", weights_only=True)
)
network.eval()

print("Model loaded. Testing forward pass...")
print()

# Test with a single example
partial_deck, available, label = val_dataset[0]
partial_deck_batch = partial_deck.unsqueeze(0)
available_batch = available.unsqueeze(0)

print(f"Partial deck cards: {partial_deck.sum().item()}")
print(f"Available cards: {available.sum().item()}")
print()

# Forward pass
with torch.no_grad():
    scores = network(partial_deck_batch, available_batch)

print(f"Output scores shape: {scores.shape}")
print(f"Min score: {scores.min().item():.4f}")
print(f"Max score: {scores.max().item():.4f}")
print(f"Mean score (available): {scores[0][available.bool()].mean().item():.4f}")
print(f"Std score (available): {scores[0][available.bool()].std().item():.4f}")
print()

# Check if model is just predicting uniform distribution
top_5_indices = scores[0].argsort(descending=True)[:5]
print("Top 5 predicted cards:")
for i, idx in enumerate(top_5_indices):
    card_name = val_dataset.cardnames[idx]
    score = scores[0][idx].item()
    is_available = available[idx].item()
    print(f"  {i+1}. {card_name:30s} score={score:7.4f} available={bool(is_available)}")

print()
print("Actual label:")
label_indices = torch.where(label > 0)[0]
for idx in label_indices:
    card_name = val_dataset.cardnames[idx]
    score = scores[0][idx].item()
    print(f"  {card_name:30s} score={score:7.4f}")

# Check temperature
print()
print(f"Learned temperature: {network.temperature.item():.4f}")
