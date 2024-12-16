import time
import warnings

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
    print(f"Validation set pick accuracy = {round(percent_correct, 1)}%")
    return percent_correct


def train_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    network: torch.nn.Module,
    learning_rate: float = 0.01,
    experiment_name: str = "test",
    model_folder: str = "../data/models/",
):
    """
    Train and evaluate model.
    """
    # Optimizer parameters.
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # Initial evaluation.
    print(f"Starting to train model. learning_rate={learning_rate}")
    best_percent_correct, best_epoch = evaluate_model(val_dataloader, network), 0

    # Train model.
    t0 = time.time()
    time_last_message = t0
    epoch = 0
    while (epoch - best_epoch) <= 20:
        network.train()
        epoch_training_loss = list()
        print(f"\nStarting epoch {epoch}")
        for i, (pool, pack, pick_vector) in enumerate(train_dataloader):
            optimizer.zero_grad()
            predicted_pick = network(pool.float(), pack.float())
            loss = loss_fn(predicted_pick, pick_vector.float())
            loss.backward()
            optimizer.step()
            epoch_training_loss.append(loss.item())

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
    print(f"Training complete. Time={round(time.time()-t0)} seconds")
    return network


def default_training_pipeline(
    set_abbreviation: str,
    draft_mode: str,
    overwrite_dataset: str = True,
) -> None:
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

    train_dataset = torch.load(train_path)
    train_dataloader = DataLoader(train_dataset, batch_size=10000, shuffle=True)

    val_dataset = torch.load(val_path)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Train network.
    network = sd.DraftNet(cardnames=train_dataset.cardnames)

    sd.train_model(
        train_dataloader,
        val_dataloader,
        network,
        experiment_name=f"{set_abbreviation}_{draft_mode}",
    )
