import torch
import torch.nn as nn
import torch.nn.functional as F


class DeckbuildEncDecNet(nn.Module):
    """
    Encoder/decoder model for deckbuilding with attention-based encoding.

    Architecture:
    1. Card Embeddings: Each card represented as a 64-dim learnable vector
    2. Attention-based Deck Encoder: Uses attention to aggregate cards in partial deck
    3. Card Scorer: Direct dot product between deck embedding and card embeddings
    """

    def __init__(
        self,
        cardnames,
        embed_dim=64,
        num_heads=4,
        dropout_rate=0.1,
    ):
        """
        Args:
            cardnames: List of card names in the set.
            embed_dim: Dimension of card and deck embeddings (default 64).
            num_heads: Number of attention heads (default 4).
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()

        self.cardnames = cardnames
        self.num_cards = len(cardnames)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Card embeddings (learnable)
        self.card_embeddings = nn.Embedding(
            num_embeddings=self.num_cards,
            embedding_dim=embed_dim
        )

        # Learnable query vector for attention pooling over deck
        self.deck_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Multi-head attention for deck encoding
        self.deck_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        # Optional: small MLP to refine deck embedding after attention
        self.deck_refine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Temperature parameter for scoring (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

    def encode_deck(self, partial_deck):
        """
        Encode partial deck into deck embedding using attention.

        Args:
            partial_deck: (batch, num_cards) - count of each card

        Returns:
            deck_embedding: (batch, embed_dim)
        """
        batch_size = partial_deck.shape[0]

        # Get all card embeddings
        card_indices = torch.arange(self.num_cards, device=partial_deck.device)
        card_embeds = self.card_embeddings(card_indices)  # (num_cards, embed_dim)

        # Expand to batch: (batch, num_cards, embed_dim)
        card_embeds_batch = card_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        # Create attention mask based on card counts
        # Cards with count > 0 should be attended to, weighted by their count
        # Normalize counts to get attention weights
        card_counts = partial_deck.float()  # (batch, num_cards)

        # Create key padding mask: True for cards not in deck (count = 0)
        key_padding_mask = (card_counts == 0)  # (batch, num_cards)

        # Weight card embeddings by their counts (as a simple form of emphasis)
        # This allows cards that appear multiple times to have more influence
        count_weights = card_counts.unsqueeze(-1)  # (batch, num_cards, 1)
        weighted_card_embeds = card_embeds_batch * count_weights

        # Expand query to batch size
        query = self.deck_query.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)

        # Apply attention: query attends to cards in deck
        # Output: (batch, 1, embed_dim)
        deck_embedding, _ = self.deck_attention(
            query=query,
            key=weighted_card_embeds,
            value=weighted_card_embeds,
            key_padding_mask=key_padding_mask,
        )

        # Remove sequence dimension: (batch, embed_dim)
        deck_embedding = deck_embedding.squeeze(1)

        # Refine the embedding
        deck_embedding = self.deck_refine(deck_embedding)

        return deck_embedding

    def score_cards(self, deck_embedding):
        """
        Score all cards based on deck embedding via dot product similarity.

        Args:
            deck_embedding: (batch, embed_dim)

        Returns:
            scores: (batch, num_cards) - logits for each card
        """
        # Get all card embeddings
        card_indices = torch.arange(self.num_cards, device=deck_embedding.device)
        card_embeds = self.card_embeddings(card_indices)  # (num_cards, embed_dim)

        # Compute similarity: dot product between deck embedding and each card embedding
        # deck_embedding: (batch, embed_dim)
        # card_embeds: (num_cards, embed_dim)
        # → (batch, num_cards)
        scores = torch.matmul(deck_embedding, card_embeds.t())

        # Scale by learnable temperature
        scores = scores / self.temperature.clamp(min=0.1)

        return scores

    def forward(self, partial_deck, available_cards):
        """
        Forward pass for deck completion prediction.

        Uses attention to encode the partial deck, then scores each card by
        similarity to the deck embedding.

        Args:
            partial_deck: (batch, num_cards) - count of each card in partial deck
            available_cards: (batch, num_cards) - binary mask of available cards

        Returns:
            Logits for each card position, masked to only score available cards.
        """
        # Encode deck state via attention
        deck_embedding = self.encode_deck(partial_deck)

        # Score all cards via dot product
        scores = self.score_cards(deck_embedding)

        # Apply available cards mask
        # Use large negative value for unavailable cards (not -inf to avoid NaN)
        # This approach works whether scores are positive or negative
        scores = scores + (1 - available_cards.float()) * (-1e9)

        return scores
