import torch
import torch.nn as nn
import torch.nn.functional as F


class DeckbuildNet(nn.Module):
    """
    MLP network to predict deck completion.

    Given a partial deck and available cards, predicts which cards
    should be added to complete the deck.
    """

    def __init__(self, cardnames, hidden_dims=None, dropout_rate=0.6):
        """
        Args:
            cardnames: List of card names in the set.
            hidden_dims: List of hidden layer dimensions. Defaults to [num_cards, 400, 400].
            dropout_rate: Dropout rate for regularization.
        """
        super(DeckbuildNet, self).__init__()

        self.cardnames = cardnames
        num_cards = len(cardnames)

        if hidden_dims is None:
            hidden_dims = [num_cards, 400, 400]

        # Network layers
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(dim) for dim in hidden_dims[1:])
        self.output_layer = nn.Linear(hidden_dims[-1], num_cards)

    def forward(self, partial_deck, available_cards):
        """
        Forward pass for deck completion prediction.

        Args:
            partial_deck: (batch, num_cards) - count of each card in partial deck
            available_cards: (batch, num_cards) - binary mask of available cards

        Returns:
            Logits for each card position, masked to only score available cards.
        """
        x = partial_deck.float()

        # Hidden layers
        for layer, norm in zip(self.hidden_layers, self.norms):
            x = layer(x)
            x = F.gelu(x)
            x = self.dropout_layer(x)
            x = norm(x)

        # Output layer
        x = self.output_layer(x)

        # Apply available cards mask - only score available cards
        x = x * available_cards.float()

        return x
