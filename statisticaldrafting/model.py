import torch
import torch.nn as nn
import torch.nn.functional as F


class DraftNet(nn.Module):
    def __init__(self, cardnames):
        """
        Simple MLP network to predict draft picks.

        Args:
            cardnames (List[str]): Names of cards in the set.
            dropout (float): Dropout rate for regularization.
        """
        super(DraftNet, self).__init__()

        # Customize to given set.
        self.cardnames = cardnames
        hidden_dims = [len(self.cardnames), 400, 400]

        # Network layers. 
        self.dropout_layer = nn.Dropout(0.6)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(dim) for dim in hidden_dims[1:])
        self.output_layer = nn.Linear(hidden_dims[-1], len(cardnames))


    def forward(self, x, pack):
        # Hidden layers
        for layer, norm in zip(self.hidden_layers, self.norms):
            x = layer(x)
            x = F.gelu(x)
            x = self.dropout_layer(x)
            x = norm(x) # Apply BatchNorm1d (ensure correct shape: [batch_size, num_features])

        # Output layer
        x = self.output_layer(x)
        
        # Require cards to be in pack. To get pick order, use a ones vector for the pack. 
        x = x * pack
        return x
