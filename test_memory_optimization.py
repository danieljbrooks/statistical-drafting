#!/usr/bin/env python3
"""
Test script to verify memory-optimized dataset creation logic.
"""
import os
import sys

# Add the package to the path
sys.path.insert(0, os.path.dirname(__file__))

import statisticaldrafting as sd

def test_dataset_creation():
    """Test that dataset creation works with the new memory-optimized approach."""

    # Use TLA Premier as test case (should have data from recent workflow runs)
    set_code = "TLA"
    draft_mode = "Premier"

    print("=" * 50)
    print(f"Testing dataset creation for {set_code} {draft_mode}")
    print("=" * 50)

    # Change to notebooks directory (where the code expects to be run from)
    os.chdir(os.path.join(os.path.dirname(__file__), "notebooks"))

    # Clean up any existing dataset to force fresh creation
    train_path = f"../data/training_sets/{set_code}_{draft_mode}_train.pth"
    val_path = f"../data/training_sets/{set_code}_{draft_mode}_val.pth"

    if os.path.exists(train_path):
        os.remove(train_path)
        print(f"Removed existing {train_path}")
    if os.path.exists(val_path):
        os.remove(val_path)
        print(f"Removed existing {val_path}")

    # Create dataset with new logic
    try:
        train_path_result, val_path_result = sd.create_dataset(
            set_abbreviation=set_code,
            draft_mode=draft_mode,
            overwrite=True,
            omit_first_days=2,
        )

        print("\n" + "=" * 50)
        print("✓ Dataset creation completed successfully!")
        print("=" * 50)

        # Load and verify the datasets
        import torch
        train_dataset = torch.load(train_path_result, weights_only=False)
        val_dataset = torch.load(val_path_result, weights_only=False)

        train_size = len(train_dataset)
        val_size = len(val_dataset)
        total_size = train_size + val_size

        print(f"\nDataset sizes:")
        print(f"  Training set: {train_size:,} examples")
        print(f"  Validation set: {val_size:,} examples")
        print(f"  Total: {total_size:,} examples")

        # Verify split ratio is approximately 80/20
        train_ratio = train_size / total_size
        val_ratio = val_size / total_size

        print(f"\nSplit ratios:")
        print(f"  Training: {train_ratio:.1%}")
        print(f"  Validation: {val_ratio:.1%}")

        # Check that ratios are reasonable (within 1% of 80/20)
        assert 0.79 <= train_ratio <= 0.81, f"Training ratio {train_ratio:.1%} is not close to 80%"
        assert 0.19 <= val_ratio <= 0.21, f"Validation ratio {val_ratio:.1%} is not close to 20%"

        print("\n✓ Split ratios are correct!")

        # Verify data shapes are consistent
        pool_sample, pack_sample, pick_sample = train_dataset[0]
        print(f"\nSample data shapes:")
        print(f"  Pool: {pool_sample.shape}")
        print(f"  Pack: {pack_sample.shape}")
        print(f"  Pick: {pick_sample.shape}")

        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"\n✗ Error during dataset creation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_creation()
    sys.exit(0 if success else 1)
