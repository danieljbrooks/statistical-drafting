import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .model import DeckbuildNet


class IterativeDeckBuilder:
    """
    Builds optimal decks from card pools using two-phase refinement with DeckbuildNet.

    Phase 1 - Mean-field:
        Iteratively refine fractional card counts using softmax-weighted updates.
        Converges to ~100% inclusion for good cards, ~0% for bad cards.

    Phase 2 - Card-by-card:
        Starting from rounded mean-field result, greedily swap the weakest deck
        card with the strongest sideboard card until stable.
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

    def get_card_scores(self, deck_counts: np.ndarray, available_mask: np.ndarray) -> np.ndarray:
        """
        Get model scores for all cards given the current partial deck.

        Args:
            deck_counts: Current (possibly fractional) deck counts.
            available_mask: Binary mask of available cards in pool.

        Returns:
            Array of scores (0-100) for each card position.
        """
        with torch.no_grad():
            partial_deck = torch.from_numpy(deck_counts).float().unsqueeze(0).to(self.device)
            available = torch.from_numpy(available_mask).float().unsqueeze(0).to(self.device)
            raw_scores = self.network(partial_deck, available)
            raw_scores = raw_scores.cpu().numpy().squeeze()

        # Convert raw scores to 0-100 ratings (sigmoid transformation)
        # Average card in pool is ~50.0
        mean = raw_scores[available_mask > 0].mean() if available_mask.sum() > 0 else 0
        std = raw_scores[available_mask > 0].std() if available_mask.sum() > 0 else 1
        ratings = 100 / (1 + np.exp(-1.2 * (raw_scores - mean) / std))

        return ratings

    # =========================================================================
    # Phase 1: Mean-field approach
    # =========================================================================

    def mean_field_update(
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
        Single mean-field update step using softmax-weighted blending.

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

    def run_mean_field(
        self,
        pool_counts: np.ndarray,
        available_mask: np.ndarray,
        target_deck_size: int,
        max_iterations: int = 100,
        convergence_tolerance: float = 0.01,
        temperature: float = 1.0,
        learning_rate: float = 0.3,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, bool, int]:
        """
        Run mean-field iteration to get fractional deck counts.

        Args:
            pool_counts: Count of each card in the pool.
            available_mask: Binary mask of available cards.
            target_deck_size: Target number of cards in deck.
            max_iterations: Maximum iterations before stopping.
            convergence_tolerance: Stop when max change is below this.
            temperature: Softmax temperature.
            learning_rate: Blending rate for updates.
            verbose: Print progress information.

        Returns:
            Tuple of (fractional_deck, converged, iterations)
        """
        # Initialize: proportional to pool
        total_pool = pool_counts.sum()
        deck_counts = pool_counts * (target_deck_size / total_pool) if total_pool > 0 else np.zeros_like(pool_counts)

        converged = False
        iteration = 0

        for iteration in range(max_iterations):
            scores = self.get_card_scores(deck_counts, available_mask)
            new_deck = self.mean_field_update(
                deck_counts, scores, pool_counts, available_mask,
                target_deck_size, temperature, learning_rate
            )

            max_change = np.max(np.abs(new_deck - deck_counts))
            deck_counts = new_deck

            if verbose and iteration % 10 == 0:
                print(f"  Mean-field iter {iteration}: max_change={max_change:.4f}")

            if max_change < convergence_tolerance:
                converged = True
                if verbose:
                    print(f"  Mean-field converged at iteration {iteration}")
                break

        return deck_counts, converged, iteration + 1

    # =========================================================================
    # Phase 2: Card-by-card refinement
    # =========================================================================

    def round_mean_field(
        self,
        fractional_deck: np.ndarray,
        pool_counts: np.ndarray,
        target_size: int,
    ) -> np.ndarray:
        """
        Round fractional deck to integers using largest-remainder method.

        Args:
            fractional_deck: Fractional deck counts.
            pool_counts: Maximum count of each card (pool limits).
            target_size: Exact number of cards needed.

        Returns:
            Integer deck counts summing to target_size.
        """
        integer_deck = np.floor(fractional_deck).astype(np.int32)
        remainders = fractional_deck - integer_deck

        deficit = target_size - integer_deck.sum()

        if deficit > 0:
            # Sort by remainder descending, add to highest remainders
            can_add = integer_deck < pool_counts
            sorted_indices = np.argsort(-remainders)
            for idx in sorted_indices:
                if deficit <= 0:
                    break
                if can_add[idx] and remainders[idx] > 0:
                    integer_deck[idx] += 1
                    deficit -= 1

        return integer_deck

    def run_card_by_card(
        self,
        deck_counts: np.ndarray,
        pool_counts: np.ndarray,
        available_mask: np.ndarray,
        max_swaps: int = 50,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, List[Tuple[str, str, float, float]], int]:
        """
        Refine deck by greedily swapping weakest deck card with strongest sideboard card.

        Args:
            deck_counts: Current integer deck counts.
            pool_counts: Count of each card in the pool.
            available_mask: Binary mask of available cards.
            max_swaps: Maximum number of swaps before stopping.
            verbose: Print each swap.

        Returns:
            Tuple of (final_deck_counts, swap_history, num_swaps)
            swap_history is list of (card_out, card_in, score_out, score_in)
        """
        deck = deck_counts.astype(np.int32).copy()
        sideboard = (pool_counts - deck).astype(np.int32)
        swap_history = []

        for swap_num in range(max_swaps):
            # Get current scores
            scores = self.get_card_scores(deck.astype(np.float32), available_mask)

            # Find weakest card in deck (lowest score among cards with count > 0)
            deck_mask = deck > 0
            if not deck_mask.any():
                break
            deck_scores = np.where(deck_mask, scores, np.inf)
            worst_deck_idx = np.argmin(deck_scores)
            worst_deck_score = scores[worst_deck_idx]

            # Find strongest card in sideboard (highest score among cards with count > 0)
            sb_mask = sideboard > 0
            if not sb_mask.any():
                break
            sb_scores = np.where(sb_mask, scores, -np.inf)
            best_sb_idx = np.argmax(sb_scores)
            best_sb_score = scores[best_sb_idx]

            # If sideboard's best is better than deck's worst, swap
            if best_sb_score > worst_deck_score:
                # Perform swap
                deck[worst_deck_idx] -= 1
                sideboard[worst_deck_idx] += 1
                deck[best_sb_idx] += 1
                sideboard[best_sb_idx] -= 1

                card_out = self.cardnames[worst_deck_idx]
                card_in = self.cardnames[best_sb_idx]
                swap_history.append((card_out, card_in, worst_deck_score, best_sb_score))

                if verbose:
                    print(f"  Swap {swap_num + 1}: OUT {card_out} ({worst_deck_score:.2f}) <- IN {card_in} ({best_sb_score:.2f})")
            else:
                # No beneficial swap possible, we're done
                if verbose:
                    print(f"  Card-by-card converged after {swap_num} swaps")
                break

        return deck, swap_history, len(swap_history)

    # =========================================================================
    # Main build method
    # =========================================================================

    def build_deck(
        self,
        pool: List[str],
        target_deck_size: int = 23,
        max_iterations: int = 100,
        convergence_tolerance: float = 0.01,
        temperature: float = 1.0,
        learning_rate: float = 0.3,
        max_swaps: int = 50,
        verbose: bool = False,
    ) -> Dict:
        """
        Build an optimal deck from a card pool using mean-field + card-by-card refinement.

        Args:
            pool: List of card names (can contain duplicates for multiple copies).
            target_deck_size: Number of non-land cards in deck (default 23).
            max_iterations: Maximum mean-field iterations.
            convergence_tolerance: Mean-field convergence threshold.
            temperature: Softmax temperature for mean-field.
            learning_rate: Blending rate for mean-field updates.
            max_swaps: Maximum card-by-card swaps.
            verbose: Print progress information.

        Returns:
            Dictionary containing deck, sideboard, scores, and metadata.
        """
        # Convert pool to vectors
        pool_counts, available_mask = self.pool_to_vectors(pool)

        if pool_counts.sum() == 0:
            raise ValueError("Pool contains no valid cards.")

        if pool_counts.sum() < target_deck_size:
            if verbose:
                print(f"Warning: Pool ({int(pool_counts.sum())} cards) smaller than target ({target_deck_size}).")
            target_deck_size = int(pool_counts.sum())

        # Phase 1: Mean-field
        if verbose:
            print("Phase 1: Mean-field iteration")
        fractional_deck, mf_converged, mf_iterations = self.run_mean_field(
            pool_counts, available_mask, target_deck_size,
            max_iterations, convergence_tolerance, temperature, learning_rate, verbose
        )

        # Round to integer deck
        integer_deck = self.round_mean_field(fractional_deck, pool_counts, target_deck_size)

        # Phase 2: Card-by-card refinement
        if verbose:
            print("\nPhase 2: Card-by-card refinement")
        final_deck, swap_history, num_swaps = self.run_card_by_card(
            integer_deck, pool_counts, available_mask, max_swaps, verbose
        )

        # Get final scores
        final_scores = self.get_card_scores(final_deck.astype(np.float32), available_mask)

        # Build result dictionaries
        deck_list = []
        deck_counts_dict = {}
        sideboard_list = []
        sideboard_counts_dict = {}
        scores_dict = {}
        fractional_dict = {}

        for i, card_name in enumerate(self.cardnames):
            if pool_counts[i] > 0:
                scores_dict[card_name] = float(final_scores[i])
                fractional_dict[card_name] = float(fractional_deck[i])
                deck_count = int(final_deck[i])
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
            "fractional_counts": fractional_dict,
            "mean_field_converged": mf_converged,
            "mean_field_iterations": mf_iterations,
            "card_by_card_swaps": num_swaps,
            "swap_history": swap_history,
        }

    # =========================================================================
    # Visualization
    # =========================================================================

    def print_mean_field_deck(self, result: Dict) -> None:
        """
        Print the mean-field fractional deck showing inclusion percentages.

        Args:
            result: Result dictionary from build_deck().
        """
        fractional = result["fractional_counts"]
        scores = result["scores"]

        # Sort by fractional count descending
        sorted_cards = sorted(fractional.items(), key=lambda x: -x[1])

        print(f"\n{'='*55}")
        print("MEAN-FIELD DECK (fractional inclusion)")
        print(f"{'='*55}")
        print(f"{'Score':>7}  {'Count':>6}  {'%':>5}  Card Name")
        print(f"{'-'*55}")

        for card_name, frac_count in sorted_cards:
            score = scores[card_name]
            pct = (frac_count / max(frac_count, 0.01)) * 100 if frac_count > 0.01 else frac_count * 100
            # Show as percentage of 1 copy
            pct_display = frac_count * 100 / max(1, frac_count) if frac_count >= 1 else frac_count * 100
            print(f"{score:7.2f}  {frac_count:6.2f}  {frac_count*100/23:5.1f}%  {card_name}")

        print(f"\nTotal: {sum(fractional.values()):.1f} cards")
        print(f"Converged: {result['mean_field_converged']} ({result['mean_field_iterations']} iterations)")

    def print_deck_and_sideboard(self, result: Dict, pool: List[str] = None) -> None:
        """
        Print a formatted view of the final deck and sideboard.

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
        print(f"FINAL DECK ({deck_size} cards)")
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
        print(f"Mean-field: {result['mean_field_iterations']} iterations")
        print(f"Card-by-card: {result['card_by_card_swaps']} swaps")

        # Verification
        if pool is not None:
            pool_counter = Counter(pool)
            valid_pool_size = sum(
                count for card, count in pool_counter.items()
                if card in self.card_to_idx
            )
            if deck_size + sideboard_size != valid_pool_size:
                print(f"WARNING: Deck ({deck_size}) + Sideboard ({sideboard_size}) != Pool ({valid_pool_size})")
            else:
                print(f"Verified: Deck + Sideboard = Pool ({valid_pool_size} cards)")
