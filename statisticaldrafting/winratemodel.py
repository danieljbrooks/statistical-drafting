import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WinrateNet(nn.Module):
    """
    Predicts the per-match win probability of the deck a draft collection becomes.

    Input is a card count vector for the collection so far, optionally conditioned on
    how many picks have been made (an early collection carries less information, so the
    model needs to know how far into the draft it is to stay calibrated).

    Output is a single logit; ``torch.sigmoid`` of it is the predicted match winrate.
    """

    def __init__(
        self,
        cardnames: List[str],
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        input_dropout: float = 0.0,
        picks_per_draft: int = 42,
        pick_embedding_dim: int = 16,
        use_pick_conditioning: bool = True,
        base_winrate: float = 0.55,
        residual: bool = False,
    ):
        super(WinrateNet, self).__init__()

        self.cardnames = cardnames
        self.num_cards = len(cardnames)
        self.picks_per_draft = picks_per_draft
        self.use_pick_conditioning = use_pick_conditioning
        self.residual = residual

        if hidden_dims is None:
            hidden_dims = [400, 400]
        self.hidden_dims = list(hidden_dims)

        input_dim = self.num_cards
        if use_pick_conditioning:
            self.pick_embedding = nn.Embedding(picks_per_draft + 1, pick_embedding_dim)
            input_dim += pick_embedding_dim

        # Dropping cards from the collection is a strong augmentation here: there are
        # only ~25k labelled drafts, so a wide network otherwise memorizes each pool.
        self.input_dropout_layer = nn.Dropout(input_dropout)

        dims = [input_dim] + self.hidden_dims
        self.hidden_layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(dim) for dim in dims[1:])
        self.dropout_layer = nn.Dropout(dropout)

        out_dim = self.hidden_dims[-1] if self.hidden_dims else input_dim
        self.output_layer = nn.Linear(out_dim, 1)

        # Start every prediction at the population winrate.
        nn.init.zeros_(self.output_layer.weight)
        nn.init.constant_(
            self.output_layer.bias, math.log(base_winrate / (1 - base_winrate))
        )

    def forward(self, collection, n_picks):
        """
        Args:
            collection: (batch, num_cards) float card counts.
            n_picks: (batch,) long, number of picks made so far.
        """
        x = self.input_dropout_layer(collection)
        if self.use_pick_conditioning:
            x = torch.cat([x, self.pick_embedding(n_picks)], dim=1)

        for layer, norm in zip(self.hidden_layers, self.norms):
            h = self.dropout_layer(F.gelu(norm(layer(x))))
            x = x + h if (self.residual and h.shape == x.shape) else h

        return self.output_layer(x).squeeze(1)


class WinrateLinear(nn.Module):
    """
    Baseline: winrate logit is a learned weight per card, summed over the collection.

    This is the "every card has a fixed value" model, with no synergy terms. Useful as
    a floor for how much the deeper networks are actually buying.
    """

    def __init__(
        self,
        cardnames: List[str],
        picks_per_draft: int = 42,
        use_pick_conditioning: bool = True,
        base_winrate: float = 0.55,
        input_dropout: float = 0.0,
    ):
        super(WinrateLinear, self).__init__()
        self.cardnames = cardnames
        self.num_cards = len(cardnames)
        self.picks_per_draft = picks_per_draft
        self.use_pick_conditioning = use_pick_conditioning

        self.input_dropout_layer = nn.Dropout(input_dropout)
        self.card_weights = nn.Linear(self.num_cards, 1)
        nn.init.zeros_(self.card_weights.weight)
        nn.init.constant_(
            self.card_weights.bias, math.log(base_winrate / (1 - base_winrate))
        )
        if use_pick_conditioning:
            # Per-pick-count scale and offset, so a partial collection can be shrunk
            # toward the mean by the right amount.
            self.pick_scale = nn.Embedding(picks_per_draft + 1, 1)
            self.pick_bias = nn.Embedding(picks_per_draft + 1, 1)
            nn.init.ones_(self.pick_scale.weight)
            nn.init.zeros_(self.pick_bias.weight)

    def forward(self, collection, n_picks):
        z = self.card_weights(self.input_dropout_layer(collection)).squeeze(1)
        if self.use_pick_conditioning:
            z = z * self.pick_scale(n_picks).squeeze(1) + self.pick_bias(n_picks).squeeze(1)
        return z


def expected_match_wins(p, max_wins: int = 7, max_losses: int = 3):
    """
    Expected match wins of an MTG Arena event that ends at ``max_wins`` wins or
    ``max_losses`` losses, given a per-match win probability ``p``.
    """
    p = torch.as_tensor(p, dtype=torch.float64)
    q = 1 - p
    expected = torch.zeros_like(p)
    # Runs that end in max_losses losses, with w wins first.
    for w in range(max_wins):
        combinations = math.comb(w + max_losses - 1, w)
        expected = expected + w * combinations * p**w * q**max_losses
    # Runs that reach max_wins, with l losses along the way.
    for l in range(max_losses):
        combinations = math.comb(max_wins + l - 1, l)
        expected = expected + max_wins * combinations * p**max_wins * q**l
    return expected


class WinrateEmbeddingNet(nn.Module):
    """
    Low-rank alternative: each card gets an embedding, the collection is the (count
    weighted) sum of its cards' embeddings, and a small MLP reads deck quality off that
    summary. The bottleneck caps capacity, which matters because the number of labelled
    drafts is small relative to the number of cards.
    """

    def __init__(
        self,
        cardnames: List[str],
        embedding_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        input_dropout: float = 0.0,
        picks_per_draft: int = 42,
        use_pick_conditioning: bool = True,
        base_winrate: float = 0.55,
    ):
        super(WinrateEmbeddingNet, self).__init__()
        self.cardnames = cardnames
        self.num_cards = len(cardnames)
        self.picks_per_draft = picks_per_draft
        self.use_pick_conditioning = use_pick_conditioning
        self.hidden_dims = list(hidden_dims or [64])

        self.input_dropout_layer = nn.Dropout(input_dropout)
        self.card_embedding = nn.Linear(self.num_cards, embedding_dim, bias=False)
        nn.init.normal_(self.card_embedding.weight, std=0.05)

        input_dim = embedding_dim
        if use_pick_conditioning:
            self.pick_embedding = nn.Embedding(picks_per_draft + 1, 8)
            input_dim += 8

        dims = [input_dim] + self.hidden_dims
        self.hidden_layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(dim) for dim in dims[1:])
        self.dropout_layer = nn.Dropout(dropout)
        self.input_norm = nn.BatchNorm1d(input_dim)

        self.output_layer = nn.Linear(self.hidden_dims[-1], 1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.constant_(
            self.output_layer.bias, math.log(base_winrate / (1 - base_winrate))
        )

    def forward(self, collection, n_picks):
        x = self.card_embedding(self.input_dropout_layer(collection))
        if self.use_pick_conditioning:
            x = torch.cat([x, self.pick_embedding(n_picks)], dim=1)
        x = self.input_norm(x)
        for layer, norm in zip(self.hidden_layers, self.norms):
            x = self.dropout_layer(F.gelu(norm(layer(x))))
        return self.output_layer(x).squeeze(1)
