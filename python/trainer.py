import pathlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import time
import typing
import collections
import utils
from utils import compute_loss
from tqdm import tqdm


def loss_funk(X_pred, X):
    return ((X-X_pred)**2).mean()
    return ((1-(X_pred)/(torch.abs(X)+0.01))**2).mean()


class Trainer:

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 early_stop_count: int,
                 epochs: int,
                 gradient_steps: int,
                 model: torch.nn.Module,
                 dataloaders: typing.List[torch.utils.data.DataLoader],
                 name: str):
        """
            Initialize our trainer class.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.epochs = epochs
        self.max_gradient_steps = gradient_steps

        #self.loss_criterion = nn.MSELoss()
        self.loss_criterion = loss_funk
        # Initialize the model
        self.model = model
        # Transfer model to GPU VRAM, if possible.
        self.model = utils.to_cuda(self.model)
        print(self.model)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), learning_rate)

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = dataloaders

        # Validate our model everytime we pass through 50% of the dataset
        self.num_steps_per_val = 200  # len(self.dataloader_train) // 16
        self.global_step = 0
        self.start_time = time.time()

        # Tracking variables
        self.VALIDATION_LOSS = collections.OrderedDict()
        self.TRAIN_LOSS = collections.OrderedDict()

        self.name = name
        self.checkpoint_dir = pathlib.Path(f"checkpoints/{name}")
        self.statistic_dir = pathlib.Path(f"statistic/{name}")

    def validation_epoch(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()
        validation_loss = compute_loss(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_LOSS[self.global_step] = validation_loss
        used_time = time.time() - self.start_time
        print(
            f"Epoch: {self.epoch:>2}",
            f"Batches per seconds: {self.global_step / used_time:.2f}",
            f"Global step: {self.global_step:>6}",
            f"Validation Loss: {validation_loss:.4f},",
            sep="\t")
        # Compute for testing set
        # test_loss = compute_loss(
        #    self.dataloader_test, self.model, self.loss_criterion
        # )
        #self.TEST_LOSS[self.global_step] = test_loss

        self.model.train()

    def should_early_stop(self):
        """
            Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = list(self.VALIDATION_LOSS.values()
                             )[-self.early_stop_count:]
        first_loss = relevant_loss[0]
        if first_loss == min(relevant_loss):
            print("Early stop criteria met")
            return True
        return False

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        def should_validate_model():
            return self.global_step % self.num_steps_per_val == 0

        for epoch in range(self.epochs):
            self.epoch = epoch
            # Perform a full pass through all the training samples
            for X_batch, _norm_params in tqdm(self.dataloader_train):

                # Perform the forward pass
                X_batch = utils.to_cuda(X_batch)
                predictions = self.model(X_batch)
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, X_batch)
                self.TRAIN_LOSS[self.global_step] = loss.detach().cpu().item()

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()

                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                self.global_step += 1
                # Compute loss/accuracy for all three datasets.
                if should_validate_model():
                    self.validation_epoch()
                    self.save_model()
                    self.save_statistic()
                    if self.should_early_stop():
                        print("Early stopping.")
                        return
                if self.global_step >= self.max_gradient_steps:
                    self.validation_epoch()
                    self.save_model()
                    self.save_statistic()
                    print("Max steps taken")
                    return

    def save_model(self):
        def is_best_model():
            """
                Returns True if current model has the lowest validation loss
            """
            validation_losses = list(self.VALIDATION_LOSS.values())
            return validation_losses[-1] == min(validation_losses)
        state_dict = self.model.state_dict()
        cpt_filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")
        utils.save_checkpoint(state_dict, cpt_filepath, is_best_model())

    def load_best_model(self):
        state_dict = utils.load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}")
            return
        self.model.load_state_dict(state_dict)

    def save_statistic(self):
        statistics = {"val_loss": self.VALIDATION_LOSS,
                      "train_loss": self.TRAIN_LOSS}
        utils.save_training_statistics(
            statistics, self.statistic_dir, self.name)

    def load_statistic(self, name):
        statistics = utils.load_training_statistic(self.statistic_dir, name)
        if "val_loss" in statistics.keys():
            self.VALIDATION_LOSS = statistics['val_loss']
        if "train_loss" in statistics.keys():
            self.TRAIN_LOSS = statistics['train_loss']


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.title("MSE Loss")
    utils.plot_loss(trainer.TRAIN_LOSS, label="Training loss")
    utils.plot_loss(trainer.VALIDATION_LOSS, label="Validation loss")
    plt.legend()

    plt.ylim([0, 0.01])
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show(block=False)
