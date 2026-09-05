"""
An explicit, learnable colour-structure term for the winrate models.

Two-colour decks beat three-plus-colour decks, and the base models only see this through
a 358-wide count vector with ~17k labelled drafts to learn it from. Colour structure is
a low-dimensional, permutation-invariant property of the collection - which colours does
not matter, only how concentrated the pool is - so handing it over directly is a much
cheaper way to learn it than hoping an MLP discovers it from raw counts.

The term is **additive on the logit** rather than concatenated into the trunk:

    logit = base_network(collection, n_picks) + colour_head(colour_features, n_picks)

Additive keeps the effect readable (the head's output *is* the colour adjustment, in
logit units) and keeps it from being entangled with, and washed out by, 400 hidden units
of card counts. The head's output layer is zero-initialized, so an untrained
``WinrateColourNet`` is exactly its base network and any gain is attributable.
"""
import os
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .winratemodel import WinrateNet
from .winratetransfer import WinrateTransferNet

MONO = ["W", "U", "B", "R", "G"]


def colour_matrix(cardnames: List[str], set_abbreviation: str,
                  cards_folder: str = "data/cards/") -> torch.Tensor:
    """
    (7, num_cards) indicator over [W, U, B, R, G, Multicolor, Colorless].

    ``color_identity`` in the card CSVs does not name the pair for a gold card, so
    multicolour cards get their own row rather than being split across two colours.
    """
    cards = pd.read_csv(os.path.join(cards_folder, f"{set_abbreviation}.csv"))
    cards = cards.set_index("name").reindex(cardnames)
    identity = cards["color_identity"].fillna("Colorless").to_numpy()
    rows = [(identity == c) for c in MONO + ["Multicolor", "Colorless"]]
    return torch.tensor([row.astype("float32") for row in rows])


class ColourHead(nn.Module):
    """
    Colour-structure features of a collection, and a small head that scores them.

    Features (16), all invariant to which colours are involved:
      * the five colour shares, sorted descending - the shape of the colour split
      * the five colour counts, sorted descending - the same, in absolute terms, since
        12/12/12 and 4/4/4 are different situations
      * share of cards in the top two colours
      * Herfindahl index and entropy of the colour distribution
      * a soft count of colours above a **learnable** threshold, with a learnable
        sharpness - this is the term that expresses "two colours, not three"
      * multicolour and colourless mass, which sit outside the colour split
    """

    N_FEATURES = 16

    def __init__(
        self,
        cardnames: List[str],
        set_abbreviation: str,
        picks_per_draft: int = 42,
        hidden_dim: int = 32,
        pick_embedding_dim: int = 8,
        cards_folder: str = "data/cards/",
    ):
        super(ColourHead, self).__init__()
        self.register_buffer(
            "colours", colour_matrix(cardnames, set_abbreviation, cards_folder)
        )
        # How many cards of a colour count as "being in" that colour, and how sharply
        # that boundary applies. Both learned: the right threshold for a 12-pick pool is
        # not the right threshold for a finished one.
        self.count_threshold = nn.Parameter(torch.tensor(4.0))
        self.count_sharpness = nn.Parameter(torch.tensor(1.0))

        self.pick_embedding = nn.Embedding(picks_per_draft + 1, pick_embedding_dim)
        self.feature_norm = nn.BatchNorm1d(self.N_FEATURES)
        self.hidden = nn.Linear(self.N_FEATURES + pick_embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        # Start as a no-op so the model begins life identical to its base network.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def features(self, collection: torch.Tensor) -> torch.Tensor:
        mass = collection @ self.colours.T  # (batch, 7)
        mono = mass[:, :5]
        total = mono.sum(dim=1, keepdim=True).clamp(min=1.0)
        share = mono / total

        sorted_share, _ = torch.sort(share, dim=1, descending=True)
        sorted_count, _ = torch.sort(mono, dim=1, descending=True)
        top2 = sorted_share[:, :2].sum(dim=1, keepdim=True)
        herfindahl = (share ** 2).sum(dim=1, keepdim=True)
        entropy = -(share * torch.log(share.clamp(min=1e-8))).sum(dim=1, keepdim=True)
        soft_count = torch.sigmoid(
            (mono - self.count_threshold) * F.softplus(self.count_sharpness)
        ).sum(dim=1, keepdim=True)

        return torch.cat(
            [sorted_share, sorted_count, top2, herfindahl, entropy, soft_count,
             mass[:, 5:6], mass[:, 6:7]],
            dim=1,
        )

    def forward(self, collection: torch.Tensor, n_picks: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [self.feature_norm(self.features(collection)), self.pick_embedding(n_picks)],
            dim=1,
        )
        return self.output(F.gelu(self.hidden(x))).squeeze(1)


class WinrateColourNet(WinrateNet):
    """``WinrateNet`` plus an additive learnable colour-structure term."""

    def __init__(self, cardnames: List[str], set_abbreviation: str,
                 draft_mode: str = "Premier", cards_folder: str = "data/cards/",
                 **kwargs):
        super(WinrateColourNet, self).__init__(cardnames, **kwargs)
        self.colour_head = ColourHead(
            cardnames, set_abbreviation,
            picks_per_draft=kwargs.get("picks_per_draft", 42),
            cards_folder=cards_folder,
        )

    def forward(self, collection, n_picks, extra=None):
        base = super(WinrateColourNet, self).forward(collection, n_picks, extra)
        return base + self.colour_head(collection, n_picks)


class WinrateTransferColourNet(WinrateTransferNet):
    """``WinrateTransferNet`` plus an additive learnable colour-structure term."""

    def __init__(self, cardnames: List[str], set_abbreviation: str,
                 draft_mode: str = "Premier", cards_folder: str = "data/cards/",
                 **kwargs):
        super(WinrateTransferColourNet, self).__init__(
            cardnames, set_abbreviation, draft_mode=draft_mode, **kwargs
        )
        self.colour_head = ColourHead(
            cardnames, set_abbreviation,
            picks_per_draft=kwargs.get("picks_per_draft", 42),
            cards_folder=cards_folder,
        )

    def forward(self, collection, n_picks, extra=None):
        base = super(WinrateTransferColourNet, self).forward(collection, n_picks, extra)
        return base + self.colour_head(collection, n_picks)
