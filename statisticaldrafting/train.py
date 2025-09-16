import time
import warnings
import json
import os
from datetime import datetime
from typing import Dict

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import statisticaldrafting as sd


def evaluate_model(val_dataloader, network):
    """
    Evaluate model pick accuracy on validation dataset.
    """
    # Count number correct picks.
    num_correct, num_incorrect = 0, 0
    for pool, pack, human_pick_vector in val_dataloader:  # Assumes batch size of 1.
        # TODO: vectorize for performance.
        human_pick_index = torch.argmax(human_pick_vector.int(), 1)
        network.eval()
        with torch.no_grad():
            bot_pick_vector = network(pool.float(), pack.float())
            bot_picks_index = torch.argmax(bot_pick_vector, 1)
        if torch.equal(human_pick_index, bot_picks_index):
            num_correct += 1
        else:
            num_incorrect += 1

    # Return and print result.
    percent_correct = 100 * num_correct / (num_correct + num_incorrect)
    print(f"Validation set pick accuracy = {round(percent_correct, 2)}%")
    return percent_correct


def train_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    network: torch.nn.Module,
    learning_rate: float = 0.03,
    experiment_name: str = "test",
    model_folder: str = "../data/models/",
):
    """
    Train and evaluate model.
    """
    # Optimizer parameters.
    # loss_fn = torch.nn.CrossEntropyLoss() # Previous implementation. 
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(network.parameters(), lr=learning_rate) # , weight_decay=1e-5)

    # Initial evaluation.
    print(f"Starting to train model. learning_rate={learning_rate}")
    best_percent_correct, best_epoch = evaluate_model(val_dataloader, network), 0

    # Train model.
    t0 = time.time()
    time_last_message = t0
    epoch = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94) # added
    while (epoch - best_epoch) <= 40:
        network.train()
        epoch_training_loss = list()
        print(f"\nStarting epoch {epoch}  lr={round(scheduler.get_last_lr()[0], 5)}")
        for i, (pool, pack, pick_vector) in enumerate(train_dataloader):
            optimizer.zero_grad()
            predicted_pick = network(pool.float(), pack.float())
            
            # Previous implementation. 
            # loss = loss_fn(predicted_pick, pick_vector.float())
            # loss.backward()

            # print(f"predicted_pick.shape, {predicted_pick.shape}")
            # print(f"pick_vector.float().shape, {pick_vector.float().shape}")

            # New recommended pattern. 
            loss_per_example = loss_fn(predicted_pick, pick_vector.float())
            
            # Find raredraft. 
            rarities = train_dataloader.dataset.rarities
            prediction_rarities = [rarities[i] for i in torch.argmax(predicted_pick, dim=1).tolist()]
            pick_rarities = [rarities[i] for i in torch.argmax(pick_vector.int(), dim=1).tolist()]
            is_raredraft = [(pick in ["common", "uncommon"]) and (pred not in ["common", "uncommon"]) for pick, pred in zip(pick_rarities, prediction_rarities)]
            raredraft_weight = torch.Tensor([3 if rd else 1 for rd in is_raredraft]) # Raredraft penalty here. 
            weighted_loss = loss_per_example * raredraft_weight
            final_loss = weighted_loss.mean()
            final_loss.backward()

            optimizer.step()
            epoch_training_loss.append(final_loss.item())

            # Provide updates every 10 seconds.
            if time_last_message - time.time() > 10:
                examples_processed = (i + 1) * pool.shape[0]
                print(
                    f"Training complete on {examples_processed} examples, time={round(time.time() - t0, 1)}"
                )

        print(f"Training loss: {round(np.mean(epoch_training_loss), 4)}")

        # Evaluate every 2 epochs
        if epoch % 2 == 0 and epoch > 0:
            # Evaluation.
            network = network.eval()
            percent_correct = evaluate_model(val_dataloader, network)

            # Save best model.
            if percent_correct > best_percent_correct:
                best_percent_correct = percent_correct
                best_epoch = epoch
                weights_path = (
                    model_folder + experiment_name + ".pt"
                )  # TODO: change this
                print(f"Saving model weights to {weights_path}")
                torch.save(network.state_dict(), weights_path)
        epoch += 1
        scheduler.step() # Update learning rate. 
    print(f"Training complete for {weights_path}. Best performance={round(best_percent_correct, 2)}% Time={round(time.time()-t0)} seconds\n")
    
    # Return training information dictionary
    training_info = {
        "experiment_name": experiment_name,
        "training_picks": len(train_dataloader.dataset),
        "validation_picks": len(val_dataloader.dataset),
        "validation_accuracy": best_percent_correct,
        "num_epochs": best_epoch,
        "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return network, training_info


def _log_training_info(training_info: dict) -> None:
    """
    Append training information to model_refresh/training_logs.json
    """
    # Determine the path to training_logs.json
    # This function is called from notebooks/ directory, so we need to go up and into model_refresh/
    logs_path = "../model_refresh/training_logs.json"
    
    try:
        # Load existing logs or create empty list
        if os.path.exists(logs_path):
            with open(logs_path, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Append new training info
        logs.append(training_info)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(logs_path), exist_ok=True)
        
        # Write updated logs
        with open(logs_path, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"📝 Training log saved to {logs_path}")
        
    except Exception as e:
        print(f"⚠️  Failed to save training log: {e}")


def default_training_pipeline(
    set_abbreviation: str,
    draft_mode: str,
    overwrite_dataset: str = True
) -> Dict:
    """
    End to end training pipeline using default values.

    Args:
            set_abbreviation (str): Three letter abbreviation of set to create training set of.
            draft_mode (str): Use either "Premier" or "Trad" draft data.
            overwrite_dataset (bool): If False, won't overwrite an existing dataset for the set and draft mode.
    """
    # Create dataset.
    train_path, val_path = sd.create_dataset(
        set_abbreviation=set_abbreviation,
        draft_mode=draft_mode,
        overwrite=overwrite_dataset,
    )

    dataset_folder = "../data/training_sets/"

    train_dataset = torch.load(train_path, weights_only=False)
    train_dataloader = DataLoader(train_dataset, batch_size=10000, shuffle=True)

    val_dataset = torch.load(val_path, weights_only=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Train network.
    network = sd.DraftNet(cardnames=train_dataset.cardnames)

    network, training_info = sd.train_model(
        train_dataloader,
        val_dataloader,
        network,
        experiment_name=f"{set_abbreviation}_{draft_mode}",
    )

    # Export final model to ONNX.
    model_path = f"../data/models/{set_abbreviation}_{draft_mode}.pt"
    onnx_path = f"../data/onnx/{set_abbreviation}_{draft_mode}.onnx"
    
    try:
        print(f"Exporting model to ONNX format: {onnx_path}")
        sd.create_onnx_model(
            model_path=model_path,
            cardnames=train_dataset.cardnames,
            onnx_path=onnx_path
        )
    except Exception as e:
        print(f"Failed to export ONNX model: {e}")
    
    # Log training information to training_logs.json
    _log_training_info(training_info)
    
    return training_info 