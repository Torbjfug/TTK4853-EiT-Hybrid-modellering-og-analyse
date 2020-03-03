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


class Trainer:

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 early_stop_count: int,
                 epochs: int,
                 model: torch.nn.Module,
                 dataloaders: typing.List[torch.utils.data.DataLoader]):
        """
            Initialize our trainer class.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.epochs = epochs

        # Since we are doing multi-class classification, we use CrossEntropyLoss
        self.loss_criterion = nn.MSELoss()
        # Initialize the model
        self.model = model
        # Transfer model to GPU VRAM, if possible.
        self.model = utils.to_cuda(self.model)
        print(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        # self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                  self.learning_rate, momentum=0.9)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), learning_rate)

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = dataloaders

        # Validate our model everytime we pass through 50% of the dataset
        self.num_steps_per_val = len(self.dataloader_train) // 2
        self.global_step = 0
        self.start_time = time.time()

        # Tracking variables
        self.VALIDATION_LOSS = collections.OrderedDict()
        self.TEST_LOSS = collections.OrderedDict()
        self.TRAIN_LOSS = collections.OrderedDict()
        self.VALIDATION_ACC = collections.OrderedDict()
        self.TEST_ACC = collections.OrderedDict()

        self.checkpoint_dir = pathlib.Path(
            "checkpoints")

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
            for X_batch in tqdm(self.dataloader_train):

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
                    if self.should_early_stop():
                        print("Early stopping.")
                        return

    def save_model(self):
        def is_best_model():
            """
                Returns True if current model has the lowest validation loss
            """
            validation_losses = list(self.VALIDATION_LOSS.values())
            return validation_losses[-1] == min(validation_losses)
        state_dict = self.model.state_dict()
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")
        utils.save_checkpoint(state_dict, filepath, is_best_model())

    def load_best_model(self):
        state_dict = utils.load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}")
            return
        self.model.load_state_dict(state_dict)


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.title("MSE Loss")
    utils.plot_loss(trainer.TRAIN_LOSS, label="Training loss")
    utils.plot_loss(trainer.VALIDATION_LOSS, label="Validation loss")
    plt.legend()
    plt.show()
