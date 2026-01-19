import json
import time
from typing import Dict, List, Tuple

import numpy as np

from .trainingset import DeckbuildDataset


def pool_from_dataset_example(dataset: DeckbuildDataset, index: int) -> List[str]:
    """
    Extract pool (deck + sideboard) as list of card names.

    Args:
        dataset: DeckbuildDataset instance
        index: Index of the example

    Returns:
        List of card names in the pool (with duplicates for multiple copies)
    """
    pool = []
    total_counts = dataset.decks[index] + dataset.sideboards[index]
    for card_name, count in zip(dataset.cardnames, total_counts):
        if count > 0:
            pool.extend([card_name] * int(count))
    return pool


def human_deck_to_counts(dataset: DeckbuildDataset, index: int) -> np.ndarray:
    """
    Get human deck as count vector (num_cards,).

    Args:
        dataset: DeckbuildDataset instance
        index: Index of the example

    Returns:
        Count vector for the human deck
    """
    return dataset.decks[index]


def predicted_deck_to_counts(result: Dict, cardnames: List[str]) -> np.ndarray:
    """
    Convert builder result to count vector.

    Args:
        result: Dictionary returned by IterativeDeckBuilder.build_deck()
        cardnames: List of card names in the dataset order

    Returns:
        Count vector matching cardnames order
    """
    counts = np.zeros(len(cardnames))
    for card_name, count in result["deck_counts"].items():
        idx = cardnames.index(card_name)
        counts[idx] = count
    return counts


def compute_card_accuracy(predicted_deck: np.ndarray, human_deck: np.ndarray) -> Tuple[int, int]:
    """
    Compute card accuracy between predicted and human decks.

    Returns (num_matches, total_cards) where:
    - num_matches: Sum of min(predicted_count, human_count) for each card
    - total_cards: Total cards in human deck

    Args:
        predicted_deck: Count vector for predicted deck
        human_deck: Count vector for human deck

    Returns:
        Tuple of (num_matches, total_cards)
    """
    matches = np.minimum(predicted_deck, human_deck).sum()
    total = human_deck.sum()
    return int(matches), int(total)


def compute_difference_count(predicted_deck: np.ndarray, human_deck: np.ndarray) -> int:
    """
    Compute number of cards different (symmetric difference).

    Args:
        predicted_deck: Count vector for predicted deck
        human_deck: Count vector for human deck

    Returns:
        Number of cards different
    """
    return int(np.abs(predicted_deck - human_deck).sum())


def evaluate_deckbuilder(
    val_dataset: DeckbuildDataset,
    builder,  # IterativeDeckBuilder
    max_examples: int = None,
    progress_interval: int = 10,
    verbose: bool = True,
    save_results: bool = True,
    output_path: str = None,
) -> Dict:
    """
    Main evaluation function - runs deckbuilder on validation examples.

    Args:
        val_dataset: Validation DeckbuildDataset
        builder: IterativeDeckBuilder instance
        max_examples: Maximum number of examples to evaluate (None = all)
        progress_interval: Print progress every N examples
        verbose: Print progress information
        save_results: Whether to save results to JSON
        output_path: Path to save results (auto-generated if None)

    Returns:
        Dictionary with evaluation results
    """
    # Initialize tracking
    total_matches = 0
    total_cards = 0
    difference_counts = []
    per_example_results = []

    num_examples = min(len(val_dataset), max_examples or len(val_dataset))

    if verbose:
        print(f"Evaluating on {num_examples} examples...")
        print(f"Target deck size: 23 cards")
        print()

    start_time = time.time()

    for i in range(num_examples):
        example_start = time.time()

        # 1. Extract pool from validation example
        pool = pool_from_dataset_example(val_dataset, i)

        # 2. Run IterativeDeckBuilder
        result = builder.build_deck(pool, target_deck_size=23, verbose=False)

        # 3. Convert prediction to count vector
        predicted_deck = predicted_deck_to_counts(result, val_dataset.cardnames)

        # 4. Get human deck
        human_deck = human_deck_to_counts(val_dataset, i)

        # 5. Compute metrics
        matches, total = compute_card_accuracy(predicted_deck, human_deck)
        num_different = compute_difference_count(predicted_deck, human_deck)

        # 6. Accumulate
        total_matches += matches
        total_cards += total
        difference_counts.append(num_different)

        # 7. Store per-example results
        per_example_results.append({
            "index": i,
            "num_matches": matches,
            "num_different": num_different,
            "pool_size": len(pool),
            "time_seconds": time.time() - example_start,
        })

        # Progress reporting
        if verbose and (i + 1) % progress_interval == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            examples_per_sec = (i + 1) / elapsed
            eta_seconds = avg_time * (num_examples - i - 1)

            current_accuracy = 100 * total_matches / total_cards if total_cards > 0 else 0
            print(f"  [{i+1:4d}/{num_examples}] "
                  f"Accuracy: {current_accuracy:.2f}% | "
                  f"Speed: {examples_per_sec:.2f} ex/s | "
                  f"ETA: {eta_seconds:.1f}s")

    total_time = time.time() - start_time

    # Build difference distribution
    difference_distribution = {
        str(i): difference_counts.count(i) for i in range(24)  # 0 to 23 cards different
    }

    # Build results dictionary
    results = {
        "card_accuracy": {
            "total_matches": total_matches,
            "total_cards": total_cards,
            "accuracy_percentage": 100 * total_matches / total_cards if total_cards > 0 else 0,
        },
        "difference_distribution": difference_distribution,
        "summary_stats": {
            "mean_cards_different": float(np.mean(difference_counts)),
            "median_cards_different": float(np.median(difference_counts)),
            "std_cards_different": float(np.std(difference_counts)),
            "min_cards_different": int(np.min(difference_counts)),
            "max_cards_different": int(np.max(difference_counts)),
        },
        "per_example_results": per_example_results,
        "metadata": {
            "num_examples_evaluated": num_examples,
            "target_deck_size": 23,
            "total_time_seconds": total_time,
            "avg_time_per_example": total_time / num_examples if num_examples > 0 else 0,
        },
    }

    if verbose:
        print()
        print(f"Evaluation completed in {total_time:.1f}s")
        print()

    # Save results if requested
    if save_results:
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"evaluation_results_{timestamp}.json"

        save_results_func(results, output_path)
        if verbose:
            print(f"Results saved to {output_path}")

    return results


def save_results_func(results: Dict, output_path: str) -> None:
    """
    Save results to JSON file.

    Args:
        results: Results dictionary from evaluate_deckbuilder()
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def print_summary(results: Dict) -> None:
    """
    Print formatted summary of evaluation results.

    Args:
        results: Results dictionary from evaluate_deckbuilder()
    """
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print()

    # Card accuracy
    acc = results["card_accuracy"]
    print(f"Card Accuracy:")
    print(f"  Total matches:  {acc['total_matches']:,} / {acc['total_cards']:,}")
    print(f"  Accuracy:       {acc['accuracy_percentage']:.2f}%")
    print()

    # Summary statistics
    stats = results["summary_stats"]
    print(f"Difference Statistics:")
    print(f"  Mean cards different:   {stats['mean_cards_different']:.2f}")
    print(f"  Median cards different: {stats['median_cards_different']:.1f}")
    print(f"  Std cards different:    {stats['std_cards_different']:.2f}")
    print(f"  Range:                  [{stats['min_cards_different']}, {stats['max_cards_different']}]")
    print()

    # Metadata
    meta = results["metadata"]
    print(f"Evaluation Metadata:")
    print(f"  Examples evaluated:     {meta['num_examples_evaluated']:,}")
    print(f"  Total time:             {meta['total_time_seconds']:.1f}s")
    print(f"  Avg time per example:   {meta['avg_time_per_example']:.3f}s")
    print()

    # Difference distribution (show all non-zero entries)
    dist = results["difference_distribution"]
    print(f"Difference Distribution:")
    sorted_dist = sorted(dist.items(), key=lambda x: int(x[0]))
    for num_diff_str, count in sorted_dist:
        if count > 0:
            pct = 100 * count / meta['num_examples_evaluated']
            print(f"  {num_diff_str:>2} cards different: {count:4d} examples ({pct:5.2f}%)")

    print("=" * 60)


def plot_difference_distribution(results: Dict, save_path: str = None) -> None:
    """
    Create histogram of difference distribution using matplotlib.

    Args:
        results: Results dictionary from evaluate_deckbuilder()
        save_path: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return

    # Extract distribution
    dist = results["difference_distribution"]
    num_diffs = [int(k) for k in dist.keys()]
    counts = [dist[str(k)] for k in num_diffs]

    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(num_diffs, counts, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Cards Different', fontsize=12)
    plt.ylabel('Number of Examples', fontsize=12)
    plt.title('Difference Distribution: Predicted vs Human Decks', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # Add statistics as text
    stats = results["summary_stats"]
    stats_text = (
        f"Mean: {stats['mean_cards_different']:.2f}\n"
        f"Median: {stats['median_cards_different']:.1f}\n"
        f"Std: {stats['std_cards_different']:.2f}"
    )
    plt.text(0.98, 0.97, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
