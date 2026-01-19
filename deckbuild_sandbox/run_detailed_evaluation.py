#!/usr/bin/env python3
"""
Detailed evaluation showing card-by-card breakdowns for each game.
"""

import argparse
import torch
import statisticaldeckbuild as sdb


def print_game_details(
    game_num,
    pool,
    result,
    human_deck,
    predicted_deck,
    cardnames,
    scores,
):
    """Print detailed breakdown for a single game."""
    print("\n" + "=" * 80)
    print(f"GAME {game_num}")
    print("=" * 80)

    # Get pool cards with their counts
    pool_counts = {}
    for card in pool:
        pool_counts[card] = pool_counts.get(card, 0) + 1

    # Build card information list
    card_info = []
    for i, card_name in enumerate(cardnames):
        if card_name in pool_counts:
            score = scores.get(card_name, 0)
            human_count = int(human_deck[i])
            pred_count = int(predicted_deck[i])

            # Determine locations
            human_loc = "MD" if human_count > 0 else "SB"
            pred_loc = "MD" if pred_count > 0 else "SB"
            match = "✓" if human_loc == pred_loc else "✗"

            card_info.append({
                'name': card_name,
                'score': score,
                'human_count': human_count,
                'pred_count': pred_count,
                'human_loc': human_loc,
                'pred_loc': pred_loc,
                'match': match,
            })

    # Sort by score descending
    card_info.sort(key=lambda x: x['score'], reverse=True)

    # Calculate matches
    total_cards = sum(c['human_count'] for c in card_info)
    matches = sum(min(c['human_count'], c['pred_count']) for c in card_info)
    accuracy = 100 * matches / total_cards if total_cards > 0 else 0

    # Print header
    print(f"\nPool: {len(pool)} cards | Accuracy: {accuracy:.1f}% ({matches}/{total_cards} cards)")
    print()
    print(f"{'Score':>7}  {'Card Name':<35} {'Human':>6} {'Model':>6} {'Match':>6}")
    print("-" * 80)

    # Print cards
    for card in card_info:
        print(f"{card['score']:7.2f}  {card['name']:<35} "
              f"{card['human_loc']:>6} {card['pred_loc']:>6} {card['match']:>6}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Detailed evaluation showing per-game card breakdowns"
    )
    parser.add_argument("--set", type=str, required=True, help="Set abbreviation")
    parser.add_argument("--mode", type=str, default="Premier", help="Draft mode")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to evaluate")
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
    print(f"Loading validation dataset: {val_dataset_path}")
    val_dataset = torch.load(val_dataset_path, weights_only=False)
    print(f"  ✓ Loaded {len(val_dataset)} validation examples\n")

    # Initialize builder
    print("Initializing IterativeDeckBuilder...")
    builder = sdb.IterativeDeckBuilder(
        set_abbreviation=args.set,
        draft_mode=args.mode,
        model_folder=args.model_folder,
        cards_folder=args.cards_folder,
    )
    print(f"  ✓ Device: {builder.device}\n")

    # Evaluate games
    print(f"Evaluating {args.num_games} games...\n")

    total_matches = 0
    total_cards = 0

    for game_idx in range(args.num_games):
        # Extract pool
        pool = sdb.evaluate.pool_from_dataset_example(val_dataset, game_idx)

        # Build deck
        result = builder.build_deck(pool, target_deck_size=23, verbose=False)

        # Get human and predicted decks
        human_deck = val_dataset.decks[game_idx]
        predicted_deck = sdb.evaluate.predicted_deck_to_counts(result, val_dataset.cardnames)

        # Print details
        print_game_details(
            game_num=game_idx + 1,
            pool=pool,
            result=result,
            human_deck=human_deck,
            predicted_deck=predicted_deck,
            cardnames=val_dataset.cardnames,
            scores=result['scores'],
        )

        # Accumulate stats
        matches, total = sdb.compute_card_accuracy(predicted_deck, human_deck)
        total_matches += matches
        total_cards += total

    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Games evaluated: {args.num_games}")
    print(f"Card accuracy: {100 * total_matches / total_cards:.2f}% ({total_matches}/{total_cards})")
    print("=" * 80)


if __name__ == "__main__":
    main()
