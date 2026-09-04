import gc
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

BASIC_NAMES = ["Forest", "Island", "Mountain", "Plains", "Swamp"]

# 17lands rank buckets, weakest to strongest.
RANK_ORDER = ["bronze", "silver", "gold", "platinum", "diamond", "mythic"]


def winrate_dataset_path(
    set_abbreviation: str,
    draft_mode: str,
    split: str,
    rank_tag: str = "diamond",
    data_folder_training_set: str = "data/winrate_sets/",
) -> str:
    """Path of a winrate dataset split. ``rank_tag`` names the rank filter it was built with."""
    return (
        f"{data_folder_training_set}{set_abbreviation}_{draft_mode}"
        f"_{rank_tag}_winrate_{split}.pth"
    )


def get_cardnames(csv_path: str) -> List[str]:
    """
    Cardnames for a set, in the same (sorted pack_card_) order used by DraftNet.
    """
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    pack_cols = sorted([c for c in columns if c.startswith("pack_card_")])
    cardnames = [c[len("pack_card_"):] for c in pack_cols]
    return [c for c in cardnames if c not in BASIC_NAMES]


class CollectionWinrateDataset(Dataset):
    """
    Draft collections paired with the match record of the deck they became.

    One row per (draft, number-of-picks) pair. Collections are stored compactly as
    the pick sequence of each draft; the count vector is materialized in batch by
    ``build_collections`` so that every prefix length is available for free.
    """

    def __init__(
        self,
        pick_sequences: np.ndarray,  # (n_drafts, picks_per_draft) int16 card indices
        wins: np.ndarray,  # (n_drafts,) int
        losses: np.ndarray,  # (n_drafts,) int
        cardnames: List[str],
        n_picks_options: Optional[List[int]] = None,
        draft_ids: Optional[np.ndarray] = None,
    ):
        self.pick_sequences = pick_sequences.astype(np.int64)
        self.wins = wins.astype(np.float32)
        self.losses = losses.astype(np.float32)
        self.cardnames = cardnames
        self.picks_per_draft = pick_sequences.shape[1]
        self.draft_ids = draft_ids

        if n_picks_options is None:
            n_picks_options = list(range(1, self.picks_per_draft + 1))
        self.n_picks_options = list(n_picks_options)

        self.n_drafts = pick_sequences.shape[0]
        self.n_options = len(self.n_picks_options)

    def __len__(self):
        return self.n_drafts * self.n_options

    def __getitem__(self, index):
        draft_index, option_index = divmod(index, self.n_options)
        n_picks = self.n_picks_options[option_index]
        return (
            torch.from_numpy(self.pick_sequences[draft_index]),
            n_picks,
            self.wins[draft_index],
            self.losses[draft_index],
        )


def build_collections(
    pick_sequences: torch.Tensor, n_picks: torch.Tensor, num_cards: int
) -> torch.Tensor:
    """
    Turn a batch of pick sequences into collection count vectors.

    Args:
        pick_sequences: (batch, picks_per_draft) card indices.
        n_picks: (batch,) number of picks to include, 1..picks_per_draft.
        num_cards: size of the collection vector; index ``num_cards`` is the pad slot.

    Returns:
        (batch, num_cards) float tensor of card counts.
    """
    batch, max_picks = pick_sequences.shape
    positions = torch.arange(max_picks, device=pick_sequences.device).unsqueeze(0)
    keep = positions < n_picks.unsqueeze(1)
    # One extra slot absorbs picks that are not draftable cards (basic lands), which
    # keeps every draft the same length without adding them to the collection.
    collections = torch.zeros(
        (batch, num_cards + 1), dtype=torch.float32, device=pick_sequences.device
    )
    collections.scatter_add_(1, pick_sequences, keep.float())
    return collections[:, :num_cards]


def create_winrate_dataset(
    set_abbreviation: str,
    draft_mode: str = "Premier",
    min_rank: Optional[str] = "diamond",
    omit_first_days: int = 2,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    overwrite: bool = False,
    data_folder_17lands: str = "data/17lands/",
    data_folder_training_set: str = "data/winrate_sets/",
    seed: int = 42,
) -> Dict[str, str]:
    """
    Build train/val/test datasets of (draft collection -> match record) from raw 17lands data.

    The split is by draft, so no collection from a held-out draft appears in training.
    Validation is used for early stopping and architecture selection; test is only for
    reporting, since the effect sizes here are small enough that selecting on the
    reported split would flatter the winner.

    Args:
        set_abbreviation (str): Three letter abbreviation of the set.
        draft_mode (str): "Premier", "Trad", "PickTwo", or "PickTwoTrad".
        min_rank (str | None): Keep drafts at this 17lands rank or above ("diamond" ->
            diamond + mythic). None keeps every draft, including the ~8% whose rank is
            not recorded.
        omit_first_days (int): Omit this many days from the beginning of the dataset.
        train_fraction (float): Fraction of drafts used for training.
        val_fraction (float): Fraction of drafts used for early stopping / model selection.
        overwrite (bool): If False, reuse existing files.
    """
    rank_tag = "allranks" if min_rank is None else min_rank
    paths = {
        split: winrate_dataset_path(
            set_abbreviation, draft_mode, split, rank_tag, data_folder_training_set
        )
        for split in ["train", "val", "test"]
    }
    if not overwrite and all(os.path.exists(path) for path in paths.values()):
        print("Winrate train/validation/test sets already exist. Skipping.")
        return paths

    csv_path = (
        f"{data_folder_17lands}draft_data_public.{set_abbreviation}."
        f"{draft_mode}Draft.csv.gz"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Did not find file {csv_path}")
    print(f"Using input file {csv_path}")

    cardnames = get_cardnames(csv_path)
    card_to_index = {name: i for i, name in enumerate(cardnames)}
    print(f"{len(cardnames)} draftable cards in {set_abbreviation}.")

    if min_rank is None:
        allowed_ranks = None
        print("Keeping all ranks, including unranked drafts.")
    else:
        allowed_ranks = set(RANK_ORDER[RANK_ORDER.index(min_rank):])
        print(f"Keeping ranks: {sorted(allowed_ranks)}")

    # Only the draft metadata is needed - the collection is reconstructed from the
    # pick sequence, so the (very wide) pack_card_/pool_ columns can be skipped.
    usecols = [
        "draft_id",
        "draft_time",
        "rank",
        "event_match_wins",
        "event_match_losses",
        "pack_number",
        "pick_number",
        "pick",
    ]
    t0 = time.time()
    df = pd.read_csv(csv_path, usecols=usecols)
    print(f"Read {len(df)} picks in {round(time.time() - t0, 1)}s.")

    n_drafts_total = df["draft_id"].nunique()

    # Omit the first days of the format.
    first_date = datetime.strptime(df["draft_time"].min(), "%Y-%m-%d %H:%M:%S")
    min_date_str = (first_date + timedelta(days=omit_first_days)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    df = df[df["draft_time"] >= min_date_str]

    # Rank filter (no filtering on player winrate or games played).
    if allowed_ranks is not None:
        df = df[df["rank"].isin(allowed_ranks)]

    # Drop drafts where no matches were played - they carry no signal.
    df = df[(df["event_match_wins"] + df["event_match_losses"]) > 0]

    df = df.sort_values(["draft_id", "pack_number", "pick_number"], kind="stable")

    # Keep only complete drafts (the modal pick count for the format).
    picks_per_draft_counts = df.groupby("draft_id").size()
    picks_per_draft = int(picks_per_draft_counts.mode().iloc[0])
    complete = picks_per_draft_counts[picks_per_draft_counts == picks_per_draft].index
    df = df[df["draft_id"].isin(complete)]
    print(f"{picks_per_draft} picks per draft; kept {len(complete)} complete drafts.")

    # Basic lands wheel to the last pick of a pack; they are real picks but not part of
    # the collection, so they go to the pad slot rather than shortening the draft.
    pad_index = len(cardnames)
    pick_index = (
        df["pick"].map(card_to_index).fillna(pad_index).to_numpy(dtype=np.int16)
    )
    pick_sequences = pick_index.reshape(-1, picks_per_draft)

    per_draft = df.drop_duplicates("draft_id")
    wins = per_draft["event_match_wins"].to_numpy(dtype=np.int16)
    losses = per_draft["event_match_losses"].to_numpy(dtype=np.int16)
    draft_ids = per_draft["draft_id"].to_numpy()

    n_drafts = pick_sequences.shape[0]
    games = wins + losses
    print(
        f"Kept {n_drafts} of {n_drafts_total} drafts "
        f"({round(100 * n_drafts / n_drafts_total, 1)}%). "
        f"Mean winrate={round(float(wins.sum() / games.sum()), 4)}, "
        f"mean games={round(float(games.mean()), 2)}."
    )

    del df, per_draft, pick_index
    gc.collect()

    # Split by draft.
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_drafts)
    n_train = int(round(train_fraction * n_drafts))
    n_val = int(round(val_fraction * n_drafts))
    split_indices = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }

    os.makedirs(data_folder_training_set, exist_ok=True)
    for split, indices in split_indices.items():
        dataset = CollectionWinrateDataset(
            pick_sequences[indices], wins[indices], losses[indices],
            cardnames, draft_ids=draft_ids[indices],
        )
        torch.save(dataset, paths[split])
        print(f"Saved {dataset.n_drafts} {split} drafts to {paths[split]}")

    return paths
