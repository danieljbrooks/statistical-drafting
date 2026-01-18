import math
import os
import time
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


def remove_basics_deckbuild(df: pd.DataFrame) -> pd.DataFrame:
    """Remove basic lands from deck and sideboard columns."""
    basic_names = ["Forest", "Island", "Mountain", "Plains", "Swamp"]

    # Build list of columns to drop
    columns_to_drop = []
    for basic in basic_names:
        columns_to_drop.extend([
            f"deck_{basic}",
            f"sideboard_{basic}",
        ])

    # Drop columns that exist
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)

    return df


class DeckbuildDataset(Dataset):
    """
    Dataset for deck completion prediction.

    Given a partial deck and available cards (sideboard + held-out cards),
    predicts which cards were held out from the original deck.
    """

    def __init__(self, decks, sideboards, cardnames, rarities, n_holdout=1):
        """
        Args:
            decks: (num_examples, num_cards) - full deck counts
            sideboards: (num_examples, num_cards) - sideboard counts
            cardnames: List of card names
            rarities: List of card rarities
            n_holdout: Number of cards to hold out from deck
        """
        self.decks = decks
        self.sideboards = sideboards
        self.cardnames = cardnames
        self.rarities = rarities
        self.n_holdout = n_holdout

    def __len__(self):
        return len(self.decks)

    def __getitem__(self, index):
        deck = self.decks[index].copy()
        sideboard = self.sideboards[index].copy()

        # Randomly remove n_holdout cards from deck
        held_out = self._random_sample_cards(deck, self.n_holdout)
        partial_deck = deck - held_out

        # Available = sideboard + held_out cards (binary mask)
        available = (sideboard + held_out) > 0

        # Label = the held_out cards (binary mask for BCE loss)
        label = held_out > 0

        return (
            torch.from_numpy(partial_deck),
            torch.from_numpy(available.astype(np.float32)),
            torch.from_numpy(label.astype(np.float32)),
        )

    def _random_sample_cards(self, deck: np.ndarray, n: int) -> np.ndarray:
        """
        Randomly sample n cards from deck, respecting card counts.
        Returns a count vector of sampled cards.
        """
        # Build list of card indices with repetition based on count
        card_indices = []
        for i, count in enumerate(deck):
            card_indices.extend([i] * int(count))

        if len(card_indices) < n:
            # If deck has fewer cards than n, sample all available
            n = len(card_indices)

        # Randomly sample n card indices
        sampled_indices = np.random.choice(card_indices, size=n, replace=False)

        # Convert back to count vector
        held_out = np.zeros_like(deck)
        for idx in sampled_indices:
            held_out[idx] += 1

        return held_out


def get_min_winrate(n_games: int, p: float = 0.55, stdev: float = 1.96) -> float:
    """
    Returns minimum winrate that true winrate > p
    For the default stdev=1.96, this is 95% confident
    """
    return p + stdev * math.sqrt(n_games * p * (1 - p)) / n_games


def create_deckbuild_dataset(
    set_abbreviation: str,
    draft_mode: str = "Premier",
    overwrite: bool = False,
    omit_first_days: int = 2,
    train_fraction: float = 0.8,
    n_holdout: int = 1,
    data_folder_17lands: str = "../data/17lands/",
    data_folder_training_set: str = "../data/training_sets/",
    data_folder_cards: str = "../data/cards/",
) -> Tuple[str, str]:
    """
    Creates deck completion training and validation datasets from 17lands game data.

    Args:
        set_abbreviation: Three letter abbreviation of set.
        draft_mode: Use either "Premier", "Trad", "PickTwo", or "PickTwoTrad".
        overwrite: If False, won't overwrite an existing dataset.
        omit_first_days: Omit this many days from the beginning of the dataset.
        train_fraction: Fraction of dataset to use for training.
        n_holdout: Number of cards to hold out from deck during training.
        data_folder_17lands: Folder where raw 17lands files are stored.
        data_folder_training_set: Folder where processed datasets are stored.
        data_folder_cards: Folder where card info is stored.

    Returns:
        Tuple of (train_path, val_path)
    """
    # Check if training set exists
    train_filename = f"{set_abbreviation}_{draft_mode}_deckbuild_train.pth"
    val_filename = f"{set_abbreviation}_{draft_mode}_deckbuild_val.pth"
    train_path = data_folder_training_set + train_filename
    val_path = data_folder_training_set + val_filename

    if not overwrite and os.path.exists(train_path) and os.path.exists(val_path):
        print("Training and validation sets already exist. Skipping.")
        return train_path, val_path

    # Validate input file - use game data (not draft data)
    csv_path = f"{data_folder_17lands}game_data_public.{set_abbreviation}.{draft_mode}Draft.csv.gz"
    if os.path.exists(csv_path):
        print(f"Using input file {csv_path}")
    else:
        raise FileNotFoundError(f"Did not find file {csv_path}")

    # Initialization on a single chunk
    for game_chunk in pd.read_csv(csv_path, chunksize=10000, compression="gzip"):
        # Remove basics
        game_chunk = remove_basics_deckbuild(game_chunk)

        # Get date after omit_first_days
        first_date_str = game_chunk["draft_time"].min()
        first_date_obj = datetime.strptime(first_date_str, "%Y-%m-%d %H:%M:%S")
        min_date_obj = first_date_obj + timedelta(days=omit_first_days)
        min_date_str = min_date_obj.strftime("%Y-%m-%d %H:%M:%S")

        # Get card names from deck columns
        deck_cols = sorted([col for col in game_chunk.columns if col.startswith("deck_")])
        cardnames = [col[5:] for col in deck_cols]  # Remove "deck_" prefix

        print(f"Found {len(cardnames)} cards")
        print("Completed initialization.")
        break

    # Process full input csv in chunks
    deck_chunks, sideboard_chunks = [], []
    chunk_size = 100000
    t0 = time.time()

    for i, game_chunk in enumerate(
        pd.read_csv(csv_path, chunksize=chunk_size, compression="gzip")
    ):
        # Remove basics
        game_chunk = remove_basics_deckbuild(game_chunk)

        # Omit first days
        game_chunk = game_chunk[game_chunk["draft_time"] >= min_date_str]

        # Filter by winrate (95% confidence that winrate >= 0.55)
        min_winrate = game_chunk["user_n_games_bucket"].apply(get_min_winrate, p=0.55, stdev=1.5)
        game_chunk = game_chunk[game_chunk["user_game_win_rate_bucket"] >= min_winrate]

        # Extract deck columns
        deck_cols = sorted([col for col in game_chunk.columns if col.startswith("deck_")])
        deck_chunk = game_chunk[deck_cols].values.astype(np.uint8)

        # Extract sideboard columns
        sideboard_cols = sorted([col for col in game_chunk.columns if col.startswith("sideboard_")])
        sideboard_chunk = game_chunk[sideboard_cols].values.astype(np.uint8)

        deck_chunks.append(deck_chunk)
        sideboard_chunks.append(sideboard_chunk)

        if i % 10 == 0:
            examples_loaded = chunk_size * i
            print(f"Loaded {examples_loaded} games, t={round(time.time() - t0, 1)}s")

    print("Loaded all game data.")

    # Concatenate all chunks
    decks = np.vstack(deck_chunks)
    sideboards = np.vstack(sideboard_chunks)

    print(f"Total examples: {len(decks)}")

    # Get rarities for set
    card_csv_path = data_folder_cards + set_abbreviation + ".csv"
    if os.path.exists(card_csv_path):
        rarities = pd.read_csv(card_csv_path)["rarity"].tolist()
    else:
        print(f"Warning: Card file {card_csv_path} not found. Using empty rarities.")
        rarities = ["unknown"] * len(cardnames)

    # Create train and validation split
    decks_train, decks_val, sideboards_train, sideboards_val = train_test_split(
        decks, sideboards, test_size=1 - train_fraction, random_state=42
    )

    # Create datasets
    train_dataset = DeckbuildDataset(
        decks_train, sideboards_train, cardnames, rarities, n_holdout=n_holdout
    )
    val_dataset = DeckbuildDataset(
        decks_val, sideboards_val, cardnames, rarities, n_holdout=n_holdout
    )

    # Save datasets
    if not os.path.exists(data_folder_training_set):
        os.makedirs(data_folder_training_set)

    torch.save(train_dataset, train_path)
    print(f"Saved {len(train_dataset)} training examples to {train_path}")

    torch.save(val_dataset, val_path)
    print(f"Saved {len(val_dataset)} validation examples to {val_path}")

    return train_path, val_path
