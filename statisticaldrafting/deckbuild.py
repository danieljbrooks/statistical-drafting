"""
Deckbuilding functionality for the statistical drafting library.

This module provides tools for analyzing game data and extracting deckbuilding
information from Magic: The Gathering draft data.
"""

import math
import os
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from .trainingset import get_min_winrate, create_card_csv
from .model import DraftNet
from .train import _log_training_info


def remove_basics_from_games(game_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Remove basic lands from raw game dataset.
    
    Args:
        game_chunk: DataFrame containing game data
        
    Returns:
        DataFrame with basic land columns removed
    """
    basic_names = ["Forest", "Island", "Mountain", "Plains", "Swamp"]
    columns_to_drop = []
    
    # Add deck_ and sideboard_ basic columns to drop list
    for basic in basic_names:
        columns_to_drop.extend([
            f"deck_{basic}",
            f"sideboard_{basic}",
            f"opening_hand_{basic}",
            f"drawn_{basic}",
            f"tutored_{basic}"
        ])
    
    # If basics are in the dataset, drop them
    existing_basics = set(game_chunk.columns).intersection(columns_to_drop)
    if len(existing_basics) > 0:
        game_chunk = game_chunk.drop(columns=list(existing_basics))
        print(f"Removed {len(existing_basics)} basic land columns")
    
    return game_chunk


class DeckbuildDataset(Dataset):
    """Dataset for deckbuilding training data."""
    
    def __init__(self, pools, decks, cardnames):
        """
        Initialize the deckbuilding dataset.
        
        Args:
            pools: Array of pool vectors (cards available for deckbuilding)
            decks: Array of deck vectors (cards chosen for main deck)
            cardnames: List of card names corresponding to vector indices
        """
        self.pools = pools
        self.decks = decks
        self.cardnames = cardnames

    def __len__(self):
        return len(self.pools)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.pools[index]).float(),
            torch.from_numpy(self.decks[index]).float(),
        )


def etl_game(df_game: pd.DataFrame) -> pd.DataFrame:
    """
    Extract relevant rows from df_game for deckbuilding analysis.
    
    This function processes game data to extract the most relevant information
    for deckbuilding analysis by:
    1. Filtering to relevant columns (excluding opening_hand_, tutored_, drawn_ columns)
    2. Selecting the final build for each deck (latest match_number per draft_id)
    
    Args:
        df_game (pd.DataFrame): Raw game data DataFrame
        
    Returns:
        pd.DataFrame: Processed DataFrame with relevant columns and final builds only
    """
    df_game = df_game.copy()
    
    # Filter to relevant columns - exclude hand/draw/tutor tracking columns
    cols_of_interest = [
        c for c in df_game.columns 
        if not (c.startswith("opening_hand_") or 
                c.startswith("tutored_") or 
                c.startswith("drawn_"))
    ]
    df_game = df_game[cols_of_interest]

    # Select final build for each deck (highest match_number per draft_id)
    df_game = df_game.loc[
        df_game.groupby(["draft_id"])["match_number"].idxmax()
    ].reset_index(drop=True)

    return df_game


def create_deckbuild_dataset(
    set_abbreviation: str,
    draft_mode: str = "Premier",
    overwrite: bool = False,
    train_fraction: float = 0.8,
    data_folder_games: str = "../data/games/",
    data_folder_training_set: str = "../data/training_deckbuild/",
    data_folder_cards: str = "../data/cards/",
) -> Tuple[str, str]:
    """
    Creates clean training and validation datasets from raw game data for deckbuilding.

    Args:
        set_abbreviation (str): Three letter abbreviation of set to create training set of.
        draft_mode (str): Use either "Premier" or "Trad" draft data.
        overwrite (bool): If False, won't overwrite an existing dataset for the set and draft mode.
        train_fraction (float): Fraction of dataset to use for training.
        data_folder_games (str): Folder where raw game files are stored.
        data_folder_training_set (str): Folder where processed training & validation sets are stored.
        data_folder_cards (str): Folder where card info is stored.
    
    Returns:
        Tuple[str, str]: Paths to training and validation dataset files.
    """
    # Check if training set exists.
    train_filename = f"{set_abbreviation}_{draft_mode}_deckbuild_train.pth"
    val_filename = f"{set_abbreviation}_{draft_mode}_deckbuild_val.pth"
    train_path = data_folder_training_set + train_filename
    val_path = data_folder_training_set + val_filename
    if overwrite == False and os.path.exists(train_path) and os.path.exists(val_path):
        print("Deckbuilding training and validation sets already exist. Skipping.")
        return train_path, val_path

    # Validate input file.
    csv_path = f"{data_folder_games}game_data_public.{set_abbreviation}.{draft_mode}Draft.csv.gz"
    if os.path.exists(csv_path):
        print(f"Using input file {csv_path}")
    else:
        raise FileNotFoundError(f"Did not find file {csv_path}")

    # Load and process game data
    print("Loading game data...")
    df_game = pd.read_csv(csv_path, compression="gzip")
    print(f"Loaded {len(df_game)} game records")
    
    # Remove basic lands
    df_game = remove_basics_from_games(df_game)
    
    # Apply ETL processing
    df_processed = etl_game(df_game)
    print(f"After ETL processing: {len(df_processed)} records")
    
    # Filter for good players (similar to draft dataset)
    min_winrate = df_processed["user_n_games_bucket"].apply(get_min_winrate, p=0.55, stdev=1.5)
    df_processed = df_processed[df_processed["user_game_win_rate_bucket"] >= min_winrate]
    print(f"After filtering for good players: {len(df_processed)} records")
    
    # Get card columns for deck and pool
    deck_cols = [col for col in df_processed.columns if col.startswith("deck_")]
    # Note: Game data doesn't have pool columns directly, we need to reconstruct from deck + sideboard
    sideboard_cols = [col for col in df_processed.columns if col.startswith("sideboard_")]
    
    # Extract card names (remove prefixes)
    cardnames = [col[5:] for col in sorted(deck_cols)]  # Remove "deck_" prefix
    
    # Verify we have matching sideboard columns
    expected_sideboard_cols = ["sideboard_" + name for name in cardnames]
    missing_sideboard = set(expected_sideboard_cols) - set(sideboard_cols)
    if missing_sideboard:
        print(f"Warning: Missing sideboard columns for some cards: {len(missing_sideboard)} cards")
    
    # Create pool vectors (deck + sideboard)
    deck_data = df_processed[sorted(deck_cols)].astype(np.uint8)
    sideboard_data = df_processed[sorted(sideboard_cols)].astype(np.uint8)
    
    # Pool = deck + sideboard (total cards available)
    pool_data = deck_data.values + sideboard_data.values
    deck_vectors = deck_data.values
    
    print(f"Created vectors for {len(cardnames)} cards")
    print(f"Pool shape: {pool_data.shape}, Deck shape: {deck_vectors.shape}")
    
    # Make sure we have a card csv
    create_card_csv(
        set_abbreviation=set_abbreviation, 
        cardnames=cardnames,
        data_folder_cards=data_folder_cards
    )
    
    # Create train and validation datasets
    pools_train, pools_test, decks_train, decks_test = train_test_split(
        pool_data, deck_vectors, test_size=0.2, random_state=42
    )
    
    deckbuild_train_dataset = DeckbuildDataset(
        pools_train, decks_train, cardnames
    )
    deckbuild_val_dataset = DeckbuildDataset(
        pools_test, decks_test, cardnames
    )
    
    # Create output directory if it doesn't exist
    if not os.path.exists(data_folder_training_set):
        os.makedirs(data_folder_training_set)
    
    # Save datasets
    torch.save(deckbuild_train_dataset, train_path)
    print(f"A total of {len(deckbuild_train_dataset)} deckbuilding examples in the training set.")
    print(f"Saved training set to {train_path}")
    
    torch.save(deckbuild_val_dataset, val_path)
    print(f"A total of {len(deckbuild_val_dataset)} deckbuilding examples in the validation set.")
    print(f"Saved validation set to {val_path}")
    
    return train_path, val_path


def create_deckbuild_training_sample(pool_vector, deck_vector):
    """
    Create a training sample for deckbuilding by removing a random card from the deck.
    
    Args:
        pool_vector: Available cards (deck + sideboard)
        deck_vector: Current deck composition
        
    Returns:
        current_deck: Deck with one card removed (input to model)
        available_cards: Cards that can be added (pool - current_deck)
        target_card: The removed card (target for prediction)
    """
    # Find cards that are in the deck (have count > 0)
    deck_card_indices = torch.nonzero(deck_vector > 0, as_tuple=True)[0]
    
    if len(deck_card_indices) == 0:
        # Edge case: empty deck
        return deck_vector, pool_vector - deck_vector, torch.zeros_like(deck_vector)
    
    # Randomly select a card to remove from deck
    random_idx = torch.randint(0, len(deck_card_indices), (1,)).item()
    target_card_idx = deck_card_indices[random_idx]
    
    # Create current deck (with target card removed)
    current_deck = deck_vector.clone()
    current_deck[target_card_idx] = max(0, current_deck[target_card_idx] - 1)
    
    # Available cards = pool - current_deck (cards that could be added)
    available_cards = pool_vector - current_deck
    available_cards = torch.clamp(available_cards, min=0)  # Ensure no negative values
    
    # Create target vector (one-hot for the removed card)
    target_card = torch.zeros_like(deck_vector)
    target_card[target_card_idx] = 1
    
    return current_deck, available_cards, target_card


def evaluate_deckbuild_model(val_dataloader, network):
    """
    Evaluate deckbuilding model accuracy on validation dataset.
    """
    num_correct, num_total = 0, 0
    
    # Set model to eval mode and disable batchnorm updates
    network.eval()
    
    # Process in larger batches to avoid BatchNorm issues
    eval_samples = []
    eval_targets = []
    
    for pool_batch, deck_batch in val_dataloader:
        batch_size = pool_batch.shape[0]
        
        for i in range(batch_size):
            pool_vector = pool_batch[i]
            deck_vector = deck_batch[i]
            
            # Create training sample
            current_deck, available_cards, target_card = create_deckbuild_training_sample(
                pool_vector, deck_vector
            )
            
            eval_samples.append((current_deck.float(), available_cards.float()))
            eval_targets.append(torch.argmax(target_card).item())
    
    # Process in batches of at least 2 to avoid BatchNorm issues
    batch_size = max(2, min(32, len(eval_samples)))
    
    with torch.no_grad():
        for i in range(0, len(eval_samples), batch_size):
            batch_end = min(i + batch_size, len(eval_samples))
            batch_samples = eval_samples[i:batch_end]
            batch_targets = eval_targets[i:batch_end]
            
            # Stack samples into batch
            current_decks = torch.stack([s[0] for s in batch_samples])
            available_cards = torch.stack([s[1] for s in batch_samples])
            
            # Get predictions
            predictions = network(current_decks, available_cards)
            predicted_indices = torch.argmax(predictions, dim=1).tolist()
            
            # Count correct predictions
            for pred_idx, target_idx in zip(predicted_indices, batch_targets):
                if pred_idx == target_idx:
                    num_correct += 1
                num_total += 1
    
    percent_correct = 100 * num_correct / num_total if num_total > 0 else 0
    print(f"Deckbuild validation accuracy = {round(percent_correct, 2)}%")
    return percent_correct


def train_deckbuild_model(
    train_dataloader,
    val_dataloader,
    network: torch.nn.Module,
    learning_rate: float = 0.03,
    experiment_name: str = "test_deckbuild",
    model_folder: str = "../data/models/",
):
    """
    Train and evaluate deckbuilding model.
    
    Args:
        train_dataloader: DataLoader with (pool, deck) pairs
        val_dataloader: DataLoader with (pool, deck) pairs  
        network: DraftNet model to train
        learning_rate: Learning rate for optimizer
        experiment_name: Name for saving model (should include "_deckbuild")
        model_folder: Directory to save model weights
    """
    import time
    import torch.optim as optim
    from datetime import datetime
    
    # Optimizer parameters
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    
    # Initial evaluation
    print(f"Starting deckbuild model training. learning_rate={learning_rate}")
    best_percent_correct, best_epoch = evaluate_deckbuild_model(val_dataloader, network), 0
    
    # Training loop
    t0 = time.time()
    epoch = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    
    while (epoch - best_epoch) <= 40:
        network.train()
        epoch_training_loss = []
        print(f"\nStarting epoch {epoch}  lr={round(scheduler.get_last_lr()[0], 5)}")
        
        for batch_idx, (pool_batch, deck_batch) in enumerate(train_dataloader):
            batch_size = pool_batch.shape[0]
            
            # Collect training samples for this batch
            batch_current_decks = []
            batch_available_cards = []
            batch_targets = []
            
            for i in range(batch_size):
                pool_vector = pool_batch[i]
                deck_vector = deck_batch[i]
                
                # Create training sample
                current_deck, available_cards, target_card = create_deckbuild_training_sample(
                    pool_vector, deck_vector
                )
                
                batch_current_decks.append(current_deck.float())
                batch_available_cards.append(available_cards.float())
                batch_targets.append(target_card.float())
            
            # Stack into proper batch tensors
            current_decks_batch = torch.stack(batch_current_decks)
            available_cards_batch = torch.stack(batch_available_cards)
            targets_batch = torch.stack(batch_targets)
            
            # Forward pass on entire batch
            optimizer.zero_grad()
            predicted_cards = network(current_decks_batch, available_cards_batch)
            loss_per_sample = loss_fn(predicted_cards, targets_batch)
            loss = loss_per_sample.mean()
            
            loss.backward()
            optimizer.step()
            epoch_training_loss.append(loss.item())
            
            # Progress updates
            if batch_idx % 10 == 0:
                examples_processed = batch_idx * batch_size
                print(f"Processed {examples_processed} examples, time={round(time.time() - t0, 1)}s")
        
        print(f"Training loss: {round(np.mean(epoch_training_loss), 4)}")
        
        # Evaluate every 2 epochs
        if epoch % 2 == 0 and epoch > 0:
            percent_correct = evaluate_deckbuild_model(val_dataloader, network)
            
            # Save best model
            if percent_correct > best_percent_correct:
                best_percent_correct = percent_correct
                best_epoch = epoch
                weights_path = model_folder + experiment_name + ".pt"
                print(f"Saving deckbuild model weights to {weights_path}")
                torch.save(network.state_dict(), weights_path)
        
        epoch += 1
        scheduler.step()
    
    print(f"Deckbuild training complete for {weights_path}. Best performance={round(best_percent_correct, 2)}% Time={round(time.time()-t0)}s\n")
    
    # Return training information
    training_info = {
        "experiment_name": experiment_name,
        "model_type": "deckbuild",
        "training_examples": len(train_dataloader.dataset),
        "validation_examples": len(val_dataloader.dataset), 
        "validation_accuracy": best_percent_correct,
        "num_epochs": best_epoch,
        "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return network, training_info


def default_deckbuild_training_pipeline(
    set_abbreviation: str,
    draft_mode: str = "Premier",
    overwrite_dataset: bool = True,
    dropout_input: float = 0.6,
) -> dict:
    """
    End-to-end deckbuilding training pipeline using default values.
    
    Args:
        set_abbreviation: Three letter abbreviation of set
        draft_mode: Use either "Premier" or "Trad" draft data
        overwrite_dataset: Whether to overwrite existing dataset
        dropout_input: Dropout rate for input layer
        
    Returns:
        Dictionary with training information
    """
    from torch.utils.data import DataLoader
    
    # Create deckbuild dataset
    train_path, val_path = create_deckbuild_dataset(
        set_abbreviation=set_abbreviation,
        draft_mode=draft_mode,
        overwrite=overwrite_dataset,
    )
    
    # Load datasets
    train_dataset = torch.load(train_path, weights_only=False)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = torch.load(val_path, weights_only=False)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create and train network
    network = DraftNet(cardnames=train_dataset.cardnames, dropout_input=dropout_input)
    
    experiment_name = f"{set_abbreviation}_{draft_mode}_deckbuild"
    network, training_info = train_deckbuild_model(
        train_dataloader,
        val_dataloader,
        network,
        experiment_name=experiment_name,
    )
    
    # Log training information
    _log_training_info(training_info)
    
    return training_info
