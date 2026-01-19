#!/usr/bin/env python3
"""
Show a single random pool evaluation with all cards sorted by model rating.
"""

import argparse
import random
import torch
import statisticaldeckbuild as sdb


def show_pool_evaluation(
    val_dataset,
    builder,
    pool_index,
):
    """Show detailed evaluation for a single pool."""

    # Extract pool
    pool = sdb.evaluate.pool_from_dataset_example(val_dataset, pool_index)

    # Build deck
    result = builder.build_deck(pool, target_deck_size=23, verbose=False)

    # Get human and predicted decks
    human_deck = val_dataset.decks[pool_index]
    predicted_deck = sdb.evaluate.predicted_deck_to_counts(result, val_dataset.cardnames)

    # Calculate accuracy
    matches, total = sdb.compute_card_accuracy(predicted_deck, human_deck)
    accuracy = 100 * matches / total if total > 0 else 0
    num_different = sdb.compute_difference_count(predicted_deck, human_deck)

    # Build pool information
    pool_counts = {}
    for card in pool:
        pool_counts[card] = pool_counts.get(card, 0) + 1

    # Build card information list
    card_info = []
    for i, card_name in enumerate(val_dataset.cardnames):
        if card_name in pool_counts:
            score = result['scores'].get(card_name, 0)
            pool_count = pool_counts[card_name]
            human_count = int(human_deck[i])
            pred_count = int(predicted_deck[i])

            # Determine locations
            if human_count > 0:
                human_loc = "MD"
            else:
                human_loc = "SB"

            if pred_count > 0:
                model_loc = "MD"
            else:
                model_loc = "SB"

            # Match if exact count matches
            match = "✓" if human_count == pred_count else "✗"

            card_info.append({
                'name': card_name,
                'score': score,
                'pool_count': pool_count,
                'human_count': human_count,
                'pred_count': pred_count,
                'human_loc': human_loc,
                'model_loc': model_loc,
                'match': match,
            })

    # Sort by score descending
    card_info.sort(key=lambda x: x['score'], reverse=True)

    # Print header
    print()
    print("=" * 90)
    print(f"POOL #{pool_index}")
    print("=" * 90)
    print(f"Total cards in pool: {len(pool)}")
    print(f"Card accuracy: {accuracy:.1f}% ({matches}/{total} cards match)")
    print(f"Cards different: {num_different}")
    print()

    # Print table header
    print(f"{'Rating':>7}  {'Card Name':<35} {'Pool':>4}  {'Human':>6}  {'Model':>6}  {'Match':>5}")
    print("-" * 88)

    # Print cards
    for card in card_info:
        print(f"{card['score']:7.2f}  {card['name']:<35} {card['pool_count']:>4}  "
              f"{card['human_count']:>6}  {card['pred_count']:>6}  {card['match']:>5}")

    print()
    print(f"Legend: Numbers show copies in main deck (0 = all in sideboard), ✓ = Exact match, ✗ = Disagreement")
    print("=" * 90)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Show a single random pool evaluation with all cards"
    )
    parser.add_argument("--set", type=str, required=True, help="Set abbreviation")
    parser.add_argument("--mode", type=str, default="Premier", help="Draft mode")
    parser.add_argument("--pool-index", type=int, default=None, help="Specific pool index (default: random)")
    parser.add_argument(
        "--data-folder",
        type=str,
        default="../data/training_sets/",
        help="Folder containing validation datasets"
    )
    parser.add_argument(
        "--model-folder",
        type=str,
        default="../data/models/",
        help="Folder containing trained models"
    )
    parser.add_argument(
        "--cards-folder",
        type=str,
        default="../data/cards/",
        help="Folder containing card data"
    )

    args = parser.parse_args()

    # Load validation dataset
    val_dataset_path = f"{args.data_folder}{args.set}_{args.mode}_deckbuild_val.pth"
    val_dataset = torch.load(val_dataset_path, weights_only=False)

    # Initialize builder
    builder = sdb.IterativeDeckBuilder(
        set_abbreviation=args.set,
        draft_mode=args.mode,
        model_folder=args.model_folder,
        cards_folder=args.cards_folder,
    )

    # Select pool index
    if args.pool_index is None:
        pool_index = random.randint(0, len(val_dataset) - 1)
        print(f"Randomly selected pool index: {pool_index}")
    else:
        pool_index = args.pool_index
        print(f"Using pool index: {pool_index}")

    # Show pool evaluation
    show_pool_evaluation(val_dataset, builder, pool_index)

    print()
    print("Run again to see another random pool, or use --pool-index N to see a specific pool.")


if __name__ == "__main__":
    main()
