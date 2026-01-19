#!/usr/bin/env python3
"""
Compare encoder/decoder model vs MLP baseline on deckbuilding task.
"""

import torch
import statisticaldeckbuild as sdb
import json

def main():
    print("=" * 80)
    print("COMPARING ENCODER/DECODER VS MLP BASELINE")
    print("=" * 80)
    print()

    # Configuration
    SET_ABBREVIATION = "FDN"
    DRAFT_MODE = "Premier"
    NUM_EXAMPLES = 500  # Quick evaluation

    # Load validation dataset
    print("Loading validation dataset...")
    val_path = f"data/training_sets/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_val.pth"
    val_dataset = torch.load(val_path, weights_only=False)
    print(f"  ✓ Total validation examples: {len(val_dataset):,}")
    print(f"  ✓ Evaluating on: {NUM_EXAMPLES:,} examples")
    print()

    # Test MLP baseline (2-layer model)
    print("=" * 80)
    print("TESTING MLP BASELINE (2-LAYER)")
    print("=" * 80)
    print()

    builder_mlp = sdb.IterativeDeckBuilder(
        set_abbreviation=SET_ABBREVIATION,
        draft_mode=DRAFT_MODE,
        model_folder="data/models/",
        cards_folder="data/cards/",
    )

    results_mlp = sdb.evaluate_deckbuilder(
        val_dataset=val_dataset,
        builder=builder_mlp,
        max_examples=NUM_EXAMPLES,
        progress_interval=50,
        verbose=True,
        save_results=True,
        output_path=f"evaluation_results/eval_{SET_ABBREVIATION}_{DRAFT_MODE}_mlp_{NUM_EXAMPLES}.json",
    )

    print()
    print("MLP BASELINE RESULTS:")
    print("-" * 80)
    sdb.print_summary(results_mlp)

    # Test encoder/decoder model
    print()
    print("=" * 80)
    print("TESTING ENCODER/DECODER MODEL")
    print("=" * 80)
    print()

    # Load the attention-based encoder/decoder model
    network_encdec = sdb.DeckbuildEncDecNet(
        cardnames=val_dataset.cardnames,
        embed_dim=64,
        num_heads=4,
        dropout_rate=0.1,
    )

    model_path = f"data/models/{SET_ABBREVIATION}_{DRAFT_MODE}_deckbuild_encdec.pt"
    print(f"Loading model from: {model_path}")
    network_encdec.load_state_dict(
        torch.load(model_path, weights_only=True)
    )
    network_encdec.eval()

    # Count parameters
    total_params_encdec = sum(p.numel() for p in network_encdec.parameters())
    total_params_mlp = sum(p.numel() for p in builder_mlp.network.parameters())
    print(f"  ✓ Encoder/decoder parameters: {total_params_encdec:,}")
    print(f"  ✓ MLP baseline parameters: {total_params_mlp:,}")
    print(f"  ✓ Parameter reduction: {100 * (1 - total_params_encdec / total_params_mlp):.1f}%")
    print()

    builder_encdec = sdb.IterativeDeckBuilder(
        set_abbreviation=SET_ABBREVIATION,
        draft_mode=DRAFT_MODE,
        model_folder="data/models/",
        cards_folder="data/cards/",
    )
    builder_encdec.network = network_encdec

    results_encdec = sdb.evaluate_deckbuilder(
        val_dataset=val_dataset,
        builder=builder_encdec,
        max_examples=NUM_EXAMPLES,
        progress_interval=50,
        verbose=True,
        save_results=True,
        output_path=f"evaluation_results/eval_{SET_ABBREVIATION}_{DRAFT_MODE}_encdec_{NUM_EXAMPLES}.json",
    )

    print()
    print("ENCODER/DECODER RESULTS:")
    print("-" * 80)
    sdb.print_summary(results_encdec)

    # Detailed comparison
    print()
    print("=" * 80)
    print("DETAILED COMPARISON: MLP vs ENCODER/DECODER")
    print("=" * 80)
    print()

    acc_mlp = results_mlp['card_accuracy']['accuracy_percentage']
    acc_encdec = results_encdec['card_accuracy']['accuracy_percentage']
    diff_mlp = results_mlp['summary_stats']['mean_cards_different']
    diff_encdec = results_encdec['summary_stats']['mean_cards_different']

    acc_delta = acc_encdec - acc_mlp
    diff_delta = diff_mlp - diff_encdec

    print(f"{'Metric':<30} {'MLP':>12} {'EncDec':>12} {'Difference':>12}")
    print("-" * 80)
    print(f"{'Parameters':<30} {total_params_mlp:>12,} {total_params_encdec:>12,} {f'-{100 * (1 - total_params_encdec / total_params_mlp):.1f}%':>12}")
    print(f"{'Card Accuracy':<30} {acc_mlp:>11.2f}% {acc_encdec:>11.2f}% {acc_delta:>11.2f}%")
    print(f"{'Mean Cards Different':<30} {diff_mlp:>12.2f} {diff_encdec:>12.2f} {diff_delta:>12.2f}")
    print(f"{'Median Cards Different':<30} {results_mlp['summary_stats']['median_cards_different']:>12.1f} {results_encdec['summary_stats']['median_cards_different']:>12.1f}")
    print(f"{'Std Cards Different':<30} {results_mlp['summary_stats']['std_cards_different']:>12.2f} {results_encdec['summary_stats']['std_cards_different']:>12.2f}")

    # Speed comparison
    speed_mlp = results_mlp['performance']['examples_per_second']
    speed_encdec = results_encdec['performance']['examples_per_second']
    print(f"{'Speed (examples/sec)':<30} {speed_mlp:>12.2f} {speed_encdec:>12.2f} {speed_encdec - speed_mlp:>11.2f}")
    print()

    # Interpretation
    print("INTERPRETATION:")
    print("-" * 80)

    if abs(acc_delta) < 0.5:
        print("⚖️  Models perform SIMILARLY (difference < 0.5%)")
    elif acc_delta > 0:
        print(f"✓ Encoder/decoder is BETTER by {acc_delta:.2f}% card accuracy")
        print(f"  with {100 * (1 - total_params_encdec / total_params_mlp):.1f}% fewer parameters!")
    else:
        print(f"✗ Encoder/decoder is WORSE by {abs(acc_delta):.2f}% card accuracy")
        print(f"  but uses {100 * (1 - total_params_encdec / total_params_mlp):.1f}% fewer parameters")

    print()

    # Save comparison
    comparison = {
        "mlp": results_mlp,
        "encdec": results_encdec,
        "summary": {
            "accuracy_difference": acc_delta,
            "mean_diff_difference": diff_delta,
            "num_examples": NUM_EXAMPLES,
            "param_reduction_pct": 100 * (1 - total_params_encdec / total_params_mlp),
        }
    }

    output_file = f"evaluation_results/comparison_mlp_vs_encdec_{NUM_EXAMPLES}.json"
    with open(output_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"Comparison saved to: {output_file}")
    print()

if __name__ == "__main__":
    main()
