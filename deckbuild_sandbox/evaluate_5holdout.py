#!/usr/bin/env python3
"""
Evaluate the 5-holdout model on actual deckbuilding and compare with baseline.
"""

import torch
import statisticaldeckbuild as sdb

def main():
    print("=" * 80)
    print("EVALUATING 5-HOLDOUT MODEL ON DECKBUILDING TASK")
    print("=" * 80)
    print()

    # Configuration
    SET_ABBREVIATION = "FDN"
    DRAFT_MODE = "Premier"

    # Load validation dataset (using the original 1-holdout dataset for fair comparison)
    print("Loading validation dataset (1-holdout version for fair comparison)...")
    val_path = f"../data/training_sets/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_val.pth"
    val_dataset = torch.load(val_path, weights_only=False)
    print(f"  ✓ Total validation examples: {len(val_dataset):,}")
    print()

    # Test 5-holdout model
    print("=" * 80)
    print("TESTING 5-HOLDOUT MODEL")
    print("=" * 80)
    print()

    # Load the 5-holdout model
    network_5holdout = sdb.DeckbuildNet(
        cardnames=val_dataset.cardnames,
        hidden_dims=[len(val_dataset.cardnames), 400, 400],
        dropout_rate=0.6
    )
    network_5holdout.load_state_dict(
        torch.load(f"../data/models/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_5holdout.pt",
                   weights_only=True)
    )
    network_5holdout.eval()

    builder_5holdout = sdb.IterativeDeckBuilder(
        set_abbreviation=SET_ABBREVIATION,
        draft_mode=DRAFT_MODE,
        model_folder="../data/models/",
        cards_folder="../data/cards/",
    )
    builder_5holdout.network = network_5holdout

    results_5holdout = sdb.evaluate_deckbuilder(
        val_dataset=val_dataset,
        builder=builder_5holdout,
        max_examples=None,  # Full validation set
        progress_interval=500,
        verbose=True,
        save_results=True,
        output_path=f"evaluation_results/eval_{SET_ABBREVIATION}_{DRAFT_MODE}_5holdout_full.json",
    )

    print()
    print("5-HOLDOUT MODEL RESULTS:")
    print("-" * 80)
    sdb.print_summary(results_5holdout)

    # Load baseline results for comparison
    print()
    print("=" * 80)
    print("LOADING BASELINE (1-HOLDOUT) RESULTS FOR COMPARISON")
    print("=" * 80)
    print()

    import json
    with open("evaluation_results/eval_FDN_Premier_2layer_full.json", "r") as f:
        results_baseline = json.load(f)

    print("BASELINE (1-HOLDOUT) RESULTS:")
    print("-" * 80)
    print(f"Card Accuracy: {results_baseline['card_accuracy']['accuracy_percentage']:.2f}%")
    print(f"Mean Cards Different: {results_baseline['summary_stats']['mean_cards_different']:.2f}")
    print()

    # Comparison
    print()
    print("=" * 80)
    print("DETAILED COMPARISON: 1-HOLDOUT vs 5-HOLDOUT")
    print("=" * 80)
    print()

    acc_1 = results_baseline['card_accuracy']['accuracy_percentage']
    acc_5 = results_5holdout['card_accuracy']['accuracy_percentage']
    diff_1 = results_baseline['summary_stats']['mean_cards_different']
    diff_5 = results_5holdout['summary_stats']['mean_cards_different']

    acc_delta = acc_5 - acc_1
    diff_delta = diff_1 - diff_5

    print(f"{'Metric':<30} {'1-Holdout':>12} {'5-Holdout':>12} {'Difference':>12}")
    print("-" * 80)
    print(f"{'Training Task':<30} {'Predict 1':>12} {'Predict 5':>12}")
    print(f"{'Training Val Accuracy':<30} {'71.42%':>12} {'99.37%':>12}")
    print()
    print(f"{'Deckbuild Card Accuracy':<30} {acc_1:>11.2f}% {acc_5:>11.2f}% {acc_delta:>11.2f}%")
    print(f"{'Mean Cards Different':<30} {diff_1:>12.2f} {diff_5:>12.2f} {diff_delta:>12.2f}")
    print(f"{'Median Cards Different':<30} {results_baseline['summary_stats']['median_cards_different']:>12.1f} {results_5holdout['summary_stats']['median_cards_different']:>12.1f}")
    print(f"{'Std Cards Different':<30} {results_baseline['summary_stats']['std_cards_different']:>12.2f} {results_5holdout['summary_stats']['std_cards_different']:>12.2f}")
    print()

    # Statistical significance
    total_cards = results_5holdout['card_accuracy']['total_cards']
    print(f"Total cards evaluated: {total_cards:,}")
    print()

    if abs(acc_delta) < 0.3:
        print("⚖️  Models perform SIMILARLY (difference < 0.3%)")
    elif acc_delta > 0:
        print(f"✓ 5-HOLDOUT model is BETTER by {acc_delta:.2f}% card accuracy")
        print(f"  This suggests training on 5 cards improves deckbuilding performance!")
    else:
        print(f"✗ 5-HOLDOUT model is WORSE by {abs(acc_delta):.2f}% card accuracy")
        print(f"  Training on more cards did not improve deckbuilding performance.")

    print()
    print("=" * 80)
    print()

    # Save comparison
    comparison = {
        "1-holdout": results_baseline,
        "5-holdout": results_5holdout,
        "summary": {
            "accuracy_difference": acc_delta,
            "mean_diff_difference": diff_delta,
            "num_examples": len(val_dataset),
        }
    }

    with open(f"evaluation_results/comparison_1holdout_vs_5holdout_full.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("Comparison saved to: evaluation_results/comparison_1holdout_vs_5holdout_full.json")

if __name__ == "__main__":
    main()
