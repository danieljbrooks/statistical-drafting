"""
Reuse a trained DraftNet as a feature extractor for deck winrate prediction.

DraftNet is fit to millions of picks by strong drafters; the winrate model has only
tens of thousands of labelled drafts. Anything DraftNet already knows about card
quality and synergy is signal the winrate model does not have to learn from its own
scarce labels.

Three readings of "what DraftNet knows", all available here:
  * ``power``   - scores with an empty pool: context-free card power.
  * ``context`` - scores with the current collection as the pool: what DraftNet wants
                  next, which also says how well the collection's own cards fit it.
  * ``hidden``  - the 400-d penultimate activation: DraftNet's learned representation
                  of the collection itself.
"""
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DraftNet


def load_draftnet(
    cardnames: List[str],
    set_abbreviation: str,
    draft_mode: str = "Premier",
    model_folder: str = "data/models/",
) -> DraftNet:
    """Load a trained DraftNet for a set, frozen and in eval mode."""
    model_path = os.path.join(model_folder, f"{set_abbreviation}_{draft_mode}.pt")
    network = DraftNet(cardnames=cardnames)
    network.load_state_dict(torch.load(model_path, weights_only=True))
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad = False
    return network


def draftnet_hidden(network: DraftNet, collection: torch.Tensor) -> torch.Tensor:
    """Penultimate DraftNet activation for a batch of collections."""
    x = collection
    for layer, norm in zip(network.hidden_layers, network.norms):
        # Dropout is identity in eval mode; kept in the same order as DraftNet.forward.
        x = norm(network.dropout_layer(F.gelu(layer(x))))
    return x


def draftnet_scores(network: DraftNet, collection: torch.Tensor) -> torch.Tensor:
    """Per-card pick scores for a batch of collections, with every card available."""
    return network.output_layer(draftnet_hidden(network, collection))


def top_k_count_sum(counts: torch.Tensor, values: torch.Tensor, k: int) -> torch.Tensor:
    """
    Sum of the k highest-valued cards in a collection, counting duplicates.

    A 42-card pool becomes a ~23-spell deck, so the best k cards are a much better proxy
    for what actually gets played than the whole pool. Ranking is a nonlinearity the
    winrate model cannot express over raw counts.
    """
    # At most k distinct cards can fill k slots, so the top-k cards by value are enough
    # - a full sort of every card is wasted work.
    owned = values.masked_fill(counts <= 0, float("-inf"))
    values_sorted, order = torch.topk(owned, k=min(k, values.shape[1]), dim=1)
    counts_sorted = torch.gather(counts, 1, order)
    values_sorted = torch.nan_to_num(values_sorted, neginf=0.0)
    # Copies of each card that fall inside the top k, allowing for duplicates.
    cumulative_before = torch.cumsum(counts_sorted, dim=1) - counts_sorted
    taken = torch.clamp(k - cumulative_before, min=0)
    taken = torch.minimum(taken, counts_sorted)
    return (taken * values_sorted).sum(dim=1)


class DraftNetFeatures(nn.Module):
    """
    Frozen DraftNet feature extractor.

    ``feature_modes`` selects any combination of "scalars", "hidden" and "context".
    """

    SCALAR_NAMES = [
        "power_sum",
        "power_mean",
        "power_max",
        "power_top23",
        "context_sum",
        "context_mean",
        "context_top23",
        "synergy_sum",
    ]

    def __init__(
        self,
        cardnames: List[str],
        set_abbreviation: str,
        draft_mode: str = "Premier",
        feature_modes: Optional[List[str]] = None,
        model_folder: str = "data/models/",
        deck_size: int = 23,
    ):
        super(DraftNetFeatures, self).__init__()
        self.cardnames = cardnames
        self.num_cards = len(cardnames)
        self.feature_modes = list(feature_modes or ["scalars"])
        self.deck_size = deck_size

        self.draftnet = load_draftnet(
            cardnames, set_abbreviation, draft_mode, model_folder
        )
        # Context-free card power: DraftNet's pack-1 pick-1 scores. Constant, so it is
        # computed once and kept as a buffer.
        with torch.no_grad():
            empty_pool = torch.zeros(2, self.num_cards)
            power = draftnet_scores(self.draftnet, empty_pool)[0]
        self.register_buffer("power", power)

        self.output_dim = 0
        if "scalars" in self.feature_modes:
            self.output_dim += len(self.SCALAR_NAMES)
        if "hidden" in self.feature_modes:
            self.output_dim += self.draftnet.hidden_layers[-1].out_features
        if "context" in self.feature_modes:
            self.output_dim += self.num_cards

    def train(self, mode: bool = True):
        """Keep the frozen DraftNet in eval mode so its BatchNorm stats never move."""
        super(DraftNetFeatures, self).train(mode)
        self.draftnet.eval()
        return self

    @torch.no_grad()
    def forward(self, collection: torch.Tensor) -> torch.Tensor:
        parts = []
        need_context = (
            "context" in self.feature_modes or "scalars" in self.feature_modes
        )
        need_hidden = "hidden" in self.feature_modes
        if need_context or need_hidden:
            hidden = draftnet_hidden(self.draftnet, collection)
        if need_context:
            context = self.draftnet.output_layer(hidden)

        if "scalars" in self.feature_modes:
            power = self.power.unsqueeze(0).expand_as(collection)
            n_cards = collection.sum(dim=1).clamp(min=1)
            power_sum = (collection * power).sum(dim=1)
            context_sum = (collection * context).sum(dim=1)
            masked_power = torch.where(
                collection > 0, power, torch.full_like(power, float("-inf"))
            )
            parts.append(
                torch.stack(
                    [
                        power_sum,
                        power_sum / n_cards,
                        masked_power.max(dim=1).values.clamp(min=-30.0),
                        top_k_count_sum(collection, power, self.deck_size),
                        context_sum,
                        context_sum / n_cards,
                        top_k_count_sum(collection, context, self.deck_size),
                        context_sum - power_sum,
                    ],
                    dim=1,
                )
            )
        if "hidden" in self.feature_modes:
            parts.append(hidden)
        if "context" in self.feature_modes:
            parts.append(context)
        return torch.cat(parts, dim=1)


class WinrateTransferNet(nn.Module):
    """
    Winrate model that reads frozen DraftNet features, optionally alongside the raw
    collection vector.
    """

    def __init__(
        self,
        cardnames: List[str],
        set_abbreviation: str,
        draft_mode: str = "Premier",
        feature_modes: Optional[List[str]] = None,
        use_collection: bool = True,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.5,
        input_dropout: float = 0.2,
        picks_per_draft: int = 42,
        pick_embedding_dim: int = 16,
        use_pick_conditioning: bool = True,
        base_winrate: float = 0.55,
        model_folder: str = "data/models/",
    ):
        super(WinrateTransferNet, self).__init__()
        import math

        self.cardnames = cardnames
        self.num_cards = len(cardnames)
        self.use_collection = use_collection
        self.picks_per_draft = picks_per_draft
        self.use_pick_conditioning = use_pick_conditioning

        self.features = DraftNetFeatures(
            cardnames, set_abbreviation, draft_mode, feature_modes, model_folder
        )
        # DraftNet logits and activations live on very different scales from card counts.
        self.feature_norm = nn.BatchNorm1d(self.features.output_dim)

        input_dim = self.features.output_dim
        if use_collection:
            input_dim += self.num_cards
        if use_pick_conditioning:
            self.pick_embedding = nn.Embedding(picks_per_draft + 1, pick_embedding_dim)
            input_dim += pick_embedding_dim

        self.input_dropout_layer = nn.Dropout(input_dropout)
        hidden_dims = [400, 400] if hidden_dims is None else list(hidden_dims)
        dims = [input_dim] + hidden_dims
        self.hidden_layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(dim) for dim in dims[1:])
        self.dropout_layer = nn.Dropout(dropout)

        self.output_layer = nn.Linear(hidden_dims[-1] if hidden_dims else input_dim, 1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.constant_(
            self.output_layer.bias, math.log(base_winrate / (1 - base_winrate))
        )

    def forward(self, collection, n_picks, extra=None):
        # ``extra`` is this model's own frozen features, precomputed once by
        # precompute_features; recomputing them every epoch is pure waste.
        features = self.features(collection) if extra is None else extra
        parts = [self.feature_norm(features)]
        if self.use_collection:
            parts.append(self.input_dropout_layer(collection))
        if self.use_pick_conditioning:
            parts.append(self.pick_embedding(n_picks))
        x = torch.cat(parts, dim=1)

        for layer, norm in zip(self.hidden_layers, self.norms):
            x = self.dropout_layer(F.gelu(norm(layer(x))))
        return self.output_layer(x).squeeze(1)


class WinrateFineTuneNet(nn.Module):
    """
    Winrate model whose trunk *is* DraftNet, initialized from the trained pick model and
    then fine-tuned (or frozen) with a fresh scalar head.

    This is the strongest form of the transfer idea: rather than reading DraftNet's
    outputs, the winrate model starts from DraftNet's learned representation of a
    collection and adapts it.
    """

    def __init__(
        self,
        cardnames: List[str],
        set_abbreviation: str,
        draft_mode: str = "Premier",
        freeze_trunk: bool = False,
        dropout: float = 0.5,
        input_dropout: float = 0.2,
        picks_per_draft: int = 42,
        pick_embedding_dim: int = 16,
        use_pick_conditioning: bool = True,
        base_winrate: float = 0.55,
        model_folder: str = "data/models/",
    ):
        super(WinrateFineTuneNet, self).__init__()
        import math

        self.cardnames = cardnames
        self.num_cards = len(cardnames)
        self.picks_per_draft = picks_per_draft
        self.use_pick_conditioning = use_pick_conditioning
        self.freeze_trunk = freeze_trunk

        draftnet = load_draftnet(cardnames, set_abbreviation, draft_mode, model_folder)
        self.hidden_layers = draftnet.hidden_layers
        self.norms = draftnet.norms
        for parameter in self.parameters():
            parameter.requires_grad = not freeze_trunk

        self.input_dropout_layer = nn.Dropout(input_dropout)
        self.dropout_layer = nn.Dropout(dropout)

        head_dim = self.hidden_layers[-1].out_features
        if use_pick_conditioning:
            self.pick_embedding = nn.Embedding(picks_per_draft + 1, pick_embedding_dim)
            head_dim += pick_embedding_dim
        self.output_layer = nn.Linear(head_dim, 1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.constant_(
            self.output_layer.bias, math.log(base_winrate / (1 - base_winrate))
        )

    def train(self, mode: bool = True):
        super(WinrateFineTuneNet, self).train(mode)
        if self.freeze_trunk:
            self.hidden_layers.eval()
            self.norms.eval()
        return self

    def forward(self, collection, n_picks, extra=None):
        x = self.input_dropout_layer(collection)
        with torch.set_grad_enabled(self.training and not self.freeze_trunk):
            for layer, norm in zip(self.hidden_layers, self.norms):
                x = norm(self.dropout_layer(F.gelu(layer(x))))
        if self.use_pick_conditioning:
            x = torch.cat([x, self.pick_embedding(n_picks)], dim=1)
        return self.output_layer(self.dropout_layer(x)).squeeze(1)
