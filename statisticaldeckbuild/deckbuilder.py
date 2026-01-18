import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .model import DeckbuildNet


class IterativeDeckBuilder:
    """
    Builds optimal decks from card pools using iterative refinement with DeckbuildNet.

    The algorithm works by:
    1. Starting with fractional card counts proportional to the pool
    2. Using the model to score all cards given the current partial deck
    3. Updating counts via softmax-weighted blending
    4. Constraining to target size and pool limits
    5. Repeating until convergence
    6. Rounding to integer counts using the largest-remainder method
    """

    def __init__(
        self,
        set_abbreviation: str,
        draft_mode: str = "Premier",
        model_folder: str = "../data/models/",
        cards_folder: str = "../data/cards/",
        device: torch.device = None,
    ):
        """
        Initialize the deck builder.

        Args:
            set_abbreviation: Three letter abbreviation of set (e.g., "FDN").
            draft_mode: Draft mode ("Premier", "Trad", etc.)
            model_folder: Folder containing trained model weights.
            cards_folder: Folder containing card data CSVs.
            device: Device to run model on (auto-detected if None).
        """
        self.set_abbreviation = set_abbreviation
        self.draft_mode = draft_mode

        # Load card data
        cards_path = os.path.join(cards_folder, f"{set_abbreviation}.csv")
        if not os.path.exists(cards_path):
            raise FileNotFoundError(f"Card data not found: {cards_path}")
        self.card_df = pd.read_csv(cards_path)
        self.cardnames = self.card_df["name"].tolist()
        self.card_to_idx = {name: i for i, name in enumerate(self.cardnames)}
        self.num_cards = len(self.cardnames)

        # Set up device
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        # Load model
        model_path = os.path.join(model_folder, f"{set_abbreviation}_{draft_mode}_deckbuild.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.network = DeckbuildNet(cardnames=self.cardnames)
        self.network.load_state_dict(torch.load(model_path, weights_only=True))
        self.network.to(self.device)
        self.network.eval()

    def pool_to_vectors(self, pool: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a pool of card names to count and availability vectors.

        Args:
            pool: List of card names (can contain duplicates).

        Returns:
            Tuple of (pool_counts, available_mask):
                - pool_counts: Array of shape (num_cards,) with count of each card
                - available_mask: Binary array of shape (num_cards,) indicating which cards are in pool
        """
        pool_counts = np.zeros(self.num_cards, dtype=np.float32)

        for card_name in pool:
            if card_name in self.card_to_idx:
                pool_counts[self.card_to_idx[card_name]] += 1
            elif card_name not in ["Plains", "Island", "Forest", "Swamp", "Mountain"]:
                print(f"Warning: '{card_name}' not found in card set, skipping.")

        available_mask = (pool_counts > 0).astype(np.float32)
        return pool_counts, available_mask

    def initialize_deck(self, pool_counts: np.ndarray, target_deck_size: int) -> np.ndarray:
        """
        Initialize fractional deck counts proportional to pool.

        Args:
            pool_counts: Array of card counts in the pool.
            target_deck_size: Target number of cards in deck (e.g., 23).

        Returns:
            Initial fractional deck counts summing to target_deck_size.
        """
        total_pool = pool_counts.sum()
        if total_pool == 0:
            return np.zeros_like(pool_counts)

        # Start with counts proportional to pool representation
        deck_counts = pool_counts * (target_deck_size / total_pool)
        return deck_counts

    def get_card_scores(self, deck_counts: np.ndarray, available_mask: np.ndarray) -> np.ndarray:
        """
        Get model scores for all cards given the current partial deck.

        Args:
            deck_counts: Current (possibly fractional) deck counts.
            available_mask: Binary mask of available cards in pool.

        Returns:
            Array of scores for each card position.
        """
        with torch.no_grad():
            partial_deck = torch.from_numpy(deck_counts).float().unsqueeze(0).to(self.device)
            available = torch.from_numpy(available_mask).float().unsqueeze(0).to(self.device)
            scores = self.network(partial_deck, available)
            return scores.cpu().numpy().squeeze()

    def softmax_update(
        self,
        deck_counts: np.ndarray,
        scores: np.ndarray,
        pool_counts: np.ndarray,
        available_mask: np.ndarray,
        target_deck_size: int,
        temperature: float,
        learning_rate: float,
    ) -> np.ndarray:
        """
        Update deck counts using softmax-weighted blending.

        Args:
            deck_counts: Current fractional deck counts.
            scores: Model scores for each card.
            pool_counts: Maximum count of each card (pool limits).
            available_mask: Binary mask of available cards.
            target_deck_size: Target number of cards in deck.
            temperature: Softmax temperature (higher = more exploration).
            learning_rate: Blending rate between current and target.

        Returns:
            Updated fractional deck counts.
        """
        # Mask out unavailable cards for softmax
        masked_scores = np.where(available_mask > 0, scores, -np.inf)

        # Compute softmax probabilities
        max_score = np.max(masked_scores[available_mask > 0]) if available_mask.sum() > 0 else 0
        exp_scores = np.exp((masked_scores - max_score) / temperature)
        exp_scores = np.where(available_mask > 0, exp_scores, 0)
        probs = exp_scores / (exp_scores.sum() + 1e-10)

        # Compute target deck from probabilities
        target = probs * target_deck_size

        # Blend current deck with target
        new_deck = (1 - learning_rate) * deck_counts + learning_rate * target

        # Clamp to pool limits
        new_deck = np.clip(new_deck, 0, pool_counts)

        # Normalize to maintain target deck size
        current_sum = new_deck.sum()
        if current_sum > 0:
            new_deck = new_deck * (target_deck_size / current_sum)
            # Re-clamp after normalization
            new_deck = np.clip(new_deck, 0, pool_counts)
            # May need slight re-normalization if clamping changed the sum
            remaining = target_deck_size - new_deck.sum()
            if remaining > 0:
                # Distribute remaining among cards not at their limit
                space = pool_counts - new_deck
                space_sum = space.sum()
                if space_sum > 0:
                    new_deck += space * (remaining / space_sum)

        return new_deck

    def round_to_integer_deck(
        self,
        fractional_deck: np.ndarray,
        pool_counts: np.ndarray,
        target_size: int,
    ) -> np.ndarray:
        """
        Round fractional deck counts to integers using the largest-remainder method.

        Args:
            fractional_deck: Fractional deck counts.
            pool_counts: Maximum count of each card (pool limits).
            target_size: Exact number of cards needed in final deck.

        Returns:
            Integer deck counts summing to exactly target_size.
        """
        # Start with floor values
        integer_deck = np.floor(fractional_deck).astype(np.int32)
        remainders = fractional_deck - integer_deck

        # Calculate deficit
        deficit = target_size - integer_deck.sum()

        if deficit > 0:
            # Get indices sorted by remainder (descending)
            # Only consider cards that haven't hit their pool limit
            can_add = integer_deck < pool_counts
            sorted_indices = np.argsort(-remainders)

            # Add 1 to cards with largest remainders until deficit = 0
            for idx in sorted_indices:
                if deficit <= 0:
                    break
                if can_add[idx] and remainders[idx] > 0:
                    integer_deck[idx] += 1
                    deficit -= 1

        elif deficit < 0:
            # Rare case: need to remove cards
            # Remove from cards with smallest remainders
            sorted_indices = np.argsort(remainders)
            for idx in sorted_indices:
                if deficit >= 0:
                    break
                if integer_deck[idx] > 0:
                    integer_deck[idx] -= 1
                    deficit += 1

        return integer_deck

    def build_deck(
        self,
        pool: List[str],
        target_deck_size: int = 23,
        max_iterations: int = 100,
        convergence_tolerance: float = 0.01,
        initial_temperature: float = 2.0,
        final_temperature: float = 0.5,
        learning_rate: float = 0.3,
        verbose: bool = False,
    ) -> Dict:
        """
        Build an optimal deck from a card pool using iterative refinement.

        Args:
            pool: List of card names (can contain duplicates for multiple copies).
            target_deck_size: Number of non-land cards in deck (default 23).
            max_iterations: Maximum number of refinement iterations.
            convergence_tolerance: Stop when max change is below this value.
            initial_temperature: Starting softmax temperature (more exploration).
            final_temperature: Ending softmax temperature (more greedy).
            learning_rate: Blending rate for updates.
            verbose: If True, print progress information.

        Returns:
            Dictionary containing:
                - 'deck': List of card names in deck
                - 'deck_counts': Dict mapping card name to count
                - 'sideboard': List of card names in sideboard
                - 'sideboard_counts': Dict mapping card name to count
                - 'scores': Dict mapping card name to final model score
                - 'converged': Whether algorithm converged
                - 'iterations': Number of iterations run
        """
        # Convert pool to vectors
        pool_counts, available_mask = self.pool_to_vectors(pool)

        if pool_counts.sum() == 0:
            raise ValueError("Pool contains no valid cards.")

        if pool_counts.sum() < target_deck_size:
            if verbose:
                print(f"Warning: Pool ({int(pool_counts.sum())} cards) smaller than target deck size ({target_deck_size}).")
            target_deck_size = int(pool_counts.sum())

        # Initialize deck
        deck_counts = self.initialize_deck(pool_counts, target_deck_size)

        # Iterative refinement
        converged = False
        iteration = 0

        for iteration in range(max_iterations):
            # Anneal temperature
            progress = iteration / max(max_iterations - 1, 1)
            temperature = initial_temperature + (final_temperature - initial_temperature) * progress

            # Get scores and update deck
            scores = self.get_card_scores(deck_counts, available_mask)
            new_deck = self.softmax_update(
                deck_counts, scores, pool_counts, available_mask,
                target_deck_size, temperature, learning_rate
            )

            # Check convergence
            max_change = np.max(np.abs(new_deck - deck_counts))
            deck_counts = new_deck

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: temp={temperature:.2f}, max_change={max_change:.4f}")

            if max_change < convergence_tolerance:
                converged = True
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break

        # Get final scores for all cards in pool
        final_scores = self.get_card_scores(deck_counts, available_mask)

        # Round to integer deck
        integer_deck = self.round_to_integer_deck(deck_counts, pool_counts, target_deck_size)

        # Build result
        deck_list = []
        deck_counts_dict = {}
        sideboard_list = []
        sideboard_counts_dict = {}
        scores_dict = {}

        for i, card_name in enumerate(self.cardnames):
            if pool_counts[i] > 0:
                scores_dict[card_name] = float(final_scores[i])
                deck_count = int(integer_deck[i])
                sideboard_count = int(pool_counts[i]) - deck_count

                if deck_count > 0:
                    deck_counts_dict[card_name] = deck_count
                    deck_list.extend([card_name] * deck_count)

                if sideboard_count > 0:
                    sideboard_counts_dict[card_name] = sideboard_count
                    sideboard_list.extend([card_name] * sideboard_count)

        return {
            "deck": deck_list,
            "deck_counts": deck_counts_dict,
            "sideboard": sideboard_list,
            "sideboard_counts": sideboard_counts_dict,
            "scores": scores_dict,
            "converged": converged,
            "iterations": iteration + 1,
        }

    def print_deck_and_sideboard(self, result: Dict, pool: List[str] = None) -> None:
        """
        Print a formatted view of the deck and sideboard.

        Args:
            result: Result dictionary from build_deck().
            pool: Original pool (optional, for verification).
        """
        scores = result["scores"]
        deck_counts = result["deck_counts"]
        sideboard_counts = result["sideboard_counts"]

        # Sort all cards by score (descending)
        all_cards_with_scores = sorted(scores.items(), key=lambda x: -x[1])

        # Print deck
        deck_size = sum(deck_counts.values())
        print(f"\n{'='*40}")
        print(f"DECK ({deck_size} cards)")
        print(f"{'='*40}")
        print(f"{'Score':>7}  {'Qty':>3}  Card Name")
        print(f"{'-'*40}")

        for card_name, score in all_cards_with_scores:
            if card_name in deck_counts:
                qty = deck_counts[card_name]
                print(f"{score:7.2f}  {qty:>3}  {card_name}")

        # Print sideboard
        sideboard_size = sum(sideboard_counts.values())
        print(f"\n{'='*40}")
        print(f"SIDEBOARD ({sideboard_size} cards)")
        print(f"{'='*40}")
        print(f"{'Score':>7}  {'Qty':>3}  Card Name")
        print(f"{'-'*40}")

        for card_name, score in all_cards_with_scores:
            if card_name in sideboard_counts:
                qty = sideboard_counts[card_name]
                print(f"{score:7.2f}  {qty:>3}  {card_name}")

        # Print summary
        print(f"\n{'='*40}")
        print(f"Converged: {result['converged']} ({result['iterations']} iterations)")

        # Verification
        if pool is not None:
            pool_counter = Counter(pool)
            # Only count cards that are in the set
            valid_pool_size = sum(
                count for card, count in pool_counter.items()
                if card in self.card_to_idx
            )
            if deck_size + sideboard_size != valid_pool_size:
                print(f"WARNING: Deck ({deck_size}) + Sideboard ({sideboard_size}) != Pool ({valid_pool_size})")
            else:
                print(f"Verified: Deck + Sideboard = Pool ({valid_pool_size} cards)")
