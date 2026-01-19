#!/usr/bin/env python3
"""
Compare 2-layer vs 3-layer models on 500 validation examples with progress.
"""

import torch
import statisticaldeckbuild as sdb
import time

def main():
    print("=" * 80)
    print("COMPARING 2-LAYER VS 3-LAYER MODELS (FULL VALIDATION SET)")
    print("=" * 80)
    print()

    # Configuration
    SET_ABBREVIATION = "FDN"
    DRAFT_MODE = "Premier"
    NUM_EXAMPLES = None  # None = evaluate all examples

    # Load validation dataset
    print("Loading validation dataset...")
    val_path = f"../data/training_sets/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_val.pth"
    val_dataset = torch.load(val_path, weights_only=False)
    num_to_eval = NUM_EXAMPLES or len(val_dataset)
    print(f"  ✓ Total validation examples: {len(val_dataset):,}")
    print(f"  ✓ Evaluating on: {num_to_eval:,} examples (FULL SET)")
    print()

    # Test 2-layer model
    print("=" * 80)
    print("TESTING 2-LAYER MODEL (BASELINE)")
    print("=" * 80)
    print()

    builder_2layer = sdb.IterativeDeckBuilder(
        set_abbreviation=SET_ABBREVIATION,
        draft_mode=DRAFT_MODE,
        model_folder="../data/models/",
        cards_folder="../data/cards/",
    )

    results_2layer = sdb.evaluate_deckbuilder(
        val_dataset=val_dataset,
        builder=builder_2layer,
        max_examples=NUM_EXAMPLES,
        progress_interval=500,  # Print every 500 examples
        verbose=True,
        save_results=True,
        output_path=f"evaluation_results/eval_{SET_ABBREVIATION}_{DRAFT_MODE}_2layer_full.json",
    )

    print()
    print("2-LAYER RESULTS:")
    print("-" * 80)
    sdb.print_summary(results_2layer)

    # Test 3-layer model
    print()
    print("=" * 80)
    print("TESTING 3-LAYER MODEL")
    print("=" * 80)
    print()

    # Load the 3-layer model
    network_3layer = sdb.DeckbuildNet(
        cardnames=val_dataset.cardnames,
        hidden_dims=[len(val_dataset.cardnames), 400, 400, 400],
        dropout_rate=0.6
    )
    network_3layer.load_state_dict(
        torch.load(f"../data/models/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_3layer.pt",
                   weights_only=True)
    )
    network_3layer.eval()

    builder_3layer = sdb.IterativeDeckBuilder(
        set_abbreviation=SET_ABBREVIATION,
        draft_mode=DRAFT_MODE,
        model_folder="../data/models/",
        cards_folder="../data/cards/",
    )
    builder_3layer.network = network_3layer

    results_3layer = sdb.evaluate_deckbuilder(
        val_dataset=val_dataset,
        builder=builder_3layer,
        max_examples=NUM_EXAMPLES,
        progress_interval=500,  # Print every 500 examples
        verbose=True,
        save_results=True,
        output_path=f"evaluation_results/eval_{SET_ABBREVIATION}_{DRAFT_MODE}_3layer_full.json",
    )

    print()
    print("3-LAYER RESULTS:")
    print("-" * 80)
    sdb.print_summary(results_3layer)

    # Comparison
    print()
    print("=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    print()

    acc_2 = results_2layer['card_accuracy']['accuracy_percentage']
    acc_3 = results_3layer['card_accuracy']['accuracy_percentage']
    diff_2 = results_2layer['summary_stats']['mean_cards_different']
    diff_3 = results_3layer['summary_stats']['mean_cards_different']

    acc_delta = acc_3 - acc_2
    diff_delta = diff_2 - diff_3

    print(f"{'Metric':<30} {'2-Layer':>12} {'3-Layer':>12} {'Difference':>12}")
    print("-" * 80)
    print(f"{'Card Accuracy':<30} {acc_2:>11.2f}% {acc_3:>11.2f}% {acc_delta:>11.2f}%")
    print(f"{'Mean Cards Different':<30} {diff_2:>12.2f} {diff_3:>12.2f} {diff_delta:>12.2f}")
    print(f"{'Median Cards Different':<30} {results_2layer['summary_stats']['median_cards_different']:>12.1f} {results_3layer['summary_stats']['median_cards_different']:>12.1f}")
    print(f"{'Std Cards Different':<30} {results_2layer['summary_stats']['std_cards_different']:>12.2f} {results_3layer['summary_stats']['std_cards_different']:>12.2f}")
    print()

    # Statistical significance check (rough)
    total_cards_2 = results_2layer['card_accuracy']['total_cards']
    total_cards_3 = results_3layer['card_accuracy']['total_cards']

    print(f"Total cards evaluated: {total_cards_2:,} (2-layer), {total_cards_3:,} (3-layer)")
    print()

    if abs(acc_delta) < 0.5:
        print("⚖️  Models perform SIMILARLY (difference < 0.5%)")
    elif acc_delta > 0:
        print(f"✓ 3-layer model is BETTER by {acc_delta:.2f}% card accuracy")
    else:
        print(f"✗ 3-layer model is WORSE by {abs(acc_delta):.2f}% card accuracy")

    print()
    print("=" * 80)
    print()

    # Save comparison
    comparison = {
        "2-layer": results_2layer,
        "3-layer": results_3layer,
        "summary": {
            "accuracy_difference": acc_delta,
            "mean_diff_difference": diff_delta,
            "num_examples": num_to_eval,
        }
    }

    import json
    with open(f"evaluation_results/comparison_2layer_vs_3layer_full.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("Comparison saved to: evaluation_results/comparison_2layer_vs_3layer_full.json")

if __name__ == "__main__":
    main()
