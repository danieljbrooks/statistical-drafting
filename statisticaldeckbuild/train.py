import time
import warnings
import json
import os
from datetime import datetime
from typing import Dict, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import DeckbuildNet
from .trainingset import DeckbuildDataset, create_deckbuild_dataset


def get_device() -> torch.device:
    """Get the best available device (MPS for Mac, CUDA for NVIDIA, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_deckbuild_model(val_dataloader: DataLoader, network: DeckbuildNet, device: torch.device = None) -> float:
    """
    Evaluate deck completion model accuracy on validation dataset.

    For n_holdout=1: checks if the highest-scoring available card matches the held-out card.

    Args:
        val_dataloader: Validation data loader
        network: Trained DeckbuildNet model
        device: Device to run on (auto-detected if None)

    Returns:
        Accuracy as a percentage
    """
    if device is None:
        device = get_device()

    num_correct = 0
    num_total = 0

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

                # Get the predicted card (highest score among available)
                # Mask out unavailable cards with large negative value
                masked_pred = pred.clone()
                masked_pred[avail == 0] = float('-inf')
                predicted_idx = torch.argmax(masked_pred).item()

                # Get the actual held-out card(s)
                actual_indices = torch.where(lbl > 0)[0]

                # Check if prediction matches any held-out card
                if predicted_idx in actual_indices:
                    num_correct += 1
                num_total += 1

    accuracy = 100 * num_correct / num_total if num_total > 0 else 0
    print(f"Validation accuracy = {round(accuracy, 2)}% ({num_correct}/{num_total})")
    return accuracy


def train_deckbuild_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    network: DeckbuildNet,
    learning_rate: float = 0.03,
    experiment_name: str = "deckbuild_test",
    model_folder: str = "../data/models/",
    patience: int = 40,
    device: torch.device = None,
) -> Tuple[DeckbuildNet, Dict]:
    """
    Train deck completion model.

    Args:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        network: DeckbuildNet model to train
        learning_rate: Initial learning rate
        experiment_name: Name for saving model
        model_folder: Folder to save model weights
        patience: Number of epochs without improvement before stopping
        device: Device to train on (auto-detected if None)

    Returns:
        Tuple of (trained network, training_info dict)
    """
    # Auto-detect device
    if device is None:
        device = get_device()
    print(f"Using device: {device}")

    # Move network to device
    network = network.to(device)

    # Loss function - BCE with logits for multi-label classification
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    # Initial evaluation
    print(f"Starting training. learning_rate={learning_rate}")
    best_accuracy = evaluate_deckbuild_model(val_dataloader, network, device)
    best_epoch = 0

    # Training loop
    t0 = time.time()
    epoch = 0
    weights_path = model_folder + experiment_name + ".pt"

    while (epoch - best_epoch) <= patience:
        network.train()
        epoch_losses = []
        print(f"\nEpoch {epoch}  lr={round(scheduler.get_last_lr()[0], 5)}")

        for i, (partial_deck, available, label) in enumerate(train_dataloader):
            # Move to device
            partial_deck = partial_deck.to(device)
            available = available.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = network(partial_deck.float(), available.float())

            # Compute loss only on available cards
            loss_per_position = loss_fn(predictions, label)
            # Mask to only compute loss on available cards
            masked_loss = loss_per_position * available
            # Average over available positions
            num_available = available.sum(dim=1, keepdim=True).clamp(min=1)
            loss_per_example = masked_loss.sum(dim=1) / num_available.squeeze()
            loss = loss_per_example.mean()

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f"Training loss: {round(np.mean(epoch_losses), 4)}")

        # Evaluate every 2 epochs
        if epoch % 2 == 0 and epoch > 0:
            network.eval()
            accuracy = evaluate_deckbuild_model(val_dataloader, network, device)

            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                os.makedirs(model_folder, exist_ok=True)
                print(f"Saving model weights to {weights_path}")
                torch.save(network.state_dict(), weights_path)

        epoch += 1
        scheduler.step()

    print(f"\nTraining complete. Best accuracy={round(best_accuracy, 2)}% at epoch {best_epoch}")
    print(f"Total time: {round(time.time() - t0)}s")

    # Return training info
    training_info = {
        "experiment_name": experiment_name,
        "training_examples": len(train_dataloader.dataset),
        "validation_examples": len(val_dataloader.dataset),
        "validation_accuracy": best_accuracy,
        "num_epochs": best_epoch,
        "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    return network, training_info


def default_deckbuild_pipeline(
    set_abbreviation: str,
    draft_mode: str = "Premier",
    overwrite_dataset: bool = True,
    n_holdout: int = 1,
    batch_size: int = 10000,
) -> Dict:
    """
    End-to-end deck completion model training pipeline.

    Args:
        set_abbreviation: Three letter abbreviation of set.
        draft_mode: Draft mode ("Premier", "Trad", etc.)
        overwrite_dataset: If False, won't overwrite existing dataset.
        n_holdout: Number of cards to hold out during training.
        batch_size: Training batch size.

    Returns:
        Training info dictionary
    """
    # Create dataset
    train_path, val_path = create_deckbuild_dataset(
        set_abbreviation=set_abbreviation,
        draft_mode=draft_mode,
        overwrite=overwrite_dataset,
        n_holdout=n_holdout,
    )

    # Load datasets
    train_dataset = torch.load(train_path, weights_only=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.load(val_path, weights_only=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create network
    network = DeckbuildNet(cardnames=train_dataset.cardnames)

    # Train
    network, training_info = train_deckbuild_model(
        train_dataloader,
        val_dataloader,
        network,
        experiment_name=f"{set_abbreviation}_{draft_mode}_deckbuild",
    )

    return training_info
