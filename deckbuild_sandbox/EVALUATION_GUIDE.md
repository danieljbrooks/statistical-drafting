# Deckbuilding Model Evaluation Guide

This guide explains how to evaluate your deckbuilding models using the evaluation harness.

## Quick Start

### Basic Usage

```bash
# Quick test on 10 examples
python run_evaluation.py --set FDN --mode Premier --max-examples 10

# Medium evaluation on 100 examples with plots
python run_evaluation.py --set FDN --mode Premier --max-examples 100 --plot

# Full evaluation on all validation data
python run_evaluation.py --set FDN --mode Premier --full
```

## Command-Line Options

### Required Arguments

- `--set`: Set abbreviation (e.g., `FDN`, `BLB`, `DSK`)

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `Premier` | Draft mode (Premier, Trad, etc.) |
| `--max-examples` | All | Maximum number of examples to evaluate |
| `--full` | False | Evaluate all examples (shortcut for --max-examples None) |
| `--progress-interval` | 10 | Print progress every N examples |
| `--data-folder` | `data/training_sets/` | Folder containing validation datasets |
| `--model-folder` | `data/models/` | Folder containing trained models |
| `--cards-folder` | `data/cards/` | Folder containing card data |
| `--output-folder` | `evaluation_results/` | Folder to save results |
| `--no-save` | False | Don't save results to file |
| `--plot` | False | Generate and save distribution plot |
| `--quiet` | False | Suppress progress output |

## Metrics Explained

### Card Accuracy
The percentage of individual cards that match between predicted and human decks.
- Calculated as: `sum(min(predicted_count, human_count)) / total_cards`
- **Higher is better** (100% = perfect match)
- Random baseline would be near 0%

### Difference Distribution
Histogram showing how many examples have N cards different from the human deck.
- Calculated as: `sum(|predicted_count - human_count|)` for each example
- **Lower is better** (0 = perfect match)
- Distribution should peak at low values (3-6 cards) if model is working well

### Summary Statistics
- **Mean cards different**: Average number of cards different across all examples
- **Median cards different**: Middle value of the difference distribution
- **Std cards different**: Standard deviation of differences
- **Range**: Min and max cards different

## Examples

### 1. Quick Sanity Check (10 examples, no saving)
```bash
python run_evaluation.py --set FDN --mode Premier --max-examples 10 --no-save
```

### 2. Standard Evaluation (100 examples with plot)
```bash
python run_evaluation.py --set FDN --mode Premier --max-examples 100 --plot
```

### 3. Full Evaluation (all validation data)
```bash
python run_evaluation.py --set FDN --mode Premier --full --plot
```

This will take longer but gives the most accurate results.

### 4. Custom Paths
```bash
python run_evaluation.py --set FDN --mode Premier --max-examples 100 \
    --data-folder ./my_data/training_sets \
    --model-folder ./my_data/models \
    --output-folder ./my_results
```

### 5. Quiet Mode (minimal output)
```bash
python run_evaluation.py --set FDN --mode Premier --full --quiet
```

## Understanding the Output

### Console Output

```
======================================================================
DECKBUILDING MODEL EVALUATION
======================================================================
Set:                FDN
Draft Mode:         Premier
Max Examples:       100
Validation Data:    data/training_sets/FDN_Premier_deckbuild_val.pth
======================================================================

Loading validation dataset...
  ✓ Loaded 7483 validation examples
  ✓ Number of cards: 281

Initializing IterativeDeckBuilder...
  ✓ Loaded model successfully
  ✓ Device: cpu

Running evaluation...
Evaluating on 100 examples...
Target deck size: 23 cards

  [  10/100] Accuracy: 80.43% | Speed: 150.2 ex/s | ETA: 0.6s
  [  20/100] Accuracy: 81.52% | Speed: 160.5 ex/s | ETA: 0.5s
  ...

============================================================
EVALUATION SUMMARY
============================================================

Card Accuracy:
  Total matches:  1,863 / 2,300
  Accuracy:       81.00%

Difference Statistics:
  Mean cards different:   6.74
  Median cards different: 7.0
  Std cards different:    2.31
  Range:                  [0, 15]

Difference Distribution:
   0 cards different:    2 examples ( 2.00%)
   1 cards different:    5 examples ( 5.00%)
   2 cards different:    8 examples ( 8.00%)
   ...
```

### Saved Files

Results are saved to `evaluation_results/` (or your custom output folder):

1. **JSON file**: `eval_FDN_Premier_20260118_143022.json`
   - Contains all metrics and per-example results
   - Can be loaded for further analysis

2. **Plot file** (if `--plot` is used): `eval_FDN_Premier_20260118_143022_distribution.png`
   - Bar chart of difference distribution
   - Includes summary statistics

## Interpreting Results

### Good Model Performance
- Card accuracy: **> 75%**
- Mean cards different: **< 8**
- Difference distribution peaks at **3-6 cards**

### Poor Model Performance
- Card accuracy: **< 60%**
- Mean cards different: **> 10**
- Difference distribution is flat or peaks at high values

### Baseline (Random)
- Card accuracy: **~30-40%** (randomly selecting cards from pool)
- Mean cards different: **> 12**

## Tips

1. **Start small**: Test with `--max-examples 10` first to verify everything works
2. **Use plots**: The `--plot` flag helps visualize model performance
3. **Full evaluation**: Run `--full` for final model assessment
4. **Save results**: Don't use `--no-save` unless testing - results are valuable for comparison
5. **Track progress**: For long runs, the progress indicator shows ETA

## Troubleshooting

### "Validation dataset not found"
- Check that your `--set` and `--mode` match the dataset filename
- Verify the `--data-folder` path is correct
- Expected filename format: `{SET}_{MODE}_deckbuild_val.pth`

### "Model not found"
- Check that the model exists in `--model-folder`
- Expected filename format: `{SET}_{MODE}_deckbuild.pt`

### Slow evaluation
- Normal speed: 100-200 examples/second on CPU
- If much slower, check if model is on wrong device
- For large evaluations, consider running overnight

## Programmatic Usage

You can also use the evaluation functions directly in Python:

```python
import torch
import statisticaldeckbuild as sdb

# Load validation dataset
val_dataset = torch.load("data/training_sets/FDN_Premier_deckbuild_val.pth",
                         weights_only=False)

# Initialize builder
builder = sdb.IterativeDeckBuilder(
    set_abbreviation="FDN",
    draft_mode="Premier",
    model_folder="data/models/",
    cards_folder="data/cards/",
)

# Run evaluation
results = sdb.evaluate_deckbuilder(
    val_dataset=val_dataset,
    builder=builder,
    max_examples=100,
    verbose=True,
)

# Display results
sdb.print_summary(results)
sdb.plot_difference_distribution(results, save_path="distribution.png")
```

## Next Steps

After evaluating your model:

1. **Analyze results**: Look at the difference distribution to understand where the model struggles
2. **Compare models**: Evaluate different model versions to track improvements
3. **Inspect failures**: Load the JSON results to find examples with high differences
4. **Iterate**: Use insights to improve model architecture or training
