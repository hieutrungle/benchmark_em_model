import torch

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn.functional as F
import logger
import os
import timer


class TorchTrainer:
    def __init__(
        self,
        model,
        training_loader,
        validation_loader,
        optimizer,
        device,
        args,
        loss_fn=F.l1_loss,
    ):
        self.model = model
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.args = args
        # Initializing in a separate cell so we can easily add more epochs to the same run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter("logs/trainer_{}".format(self.timestamp))
        self.writer = writer

        torch.set_float32_matmul_precision("high")

    @timer.Timer(logger_fn=logger.log)
    def train_one_epoch(self, epoch_index):
        running_loss = 0.0
        last_loss = 0.0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels).float()
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()

        last_loss = running_loss / (i + 1)  # loss per batch
        logger.log("  batch {} loss: {}".format(i + 1, last_loss))
        tb_x = epoch_index * len(self.training_loader) + i + 1
        self.writer.add_scalar("Loss/train", last_loss, tb_x)

        return last_loss

    def train(self, epochs):
        best_vloss = float("inf")
        for epoch in range(epochs):
            logger.log("EPOCH {}:".format(epoch + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    vinputs, vlabels = vdata
                    vinputs, vlabels = vinputs.to(self.device), vlabels.to(self.device)

                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            logger.log("LOSS train {} valid {}".format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            self.writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch + 1,
            )
            self.writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                # model_path = os.path.join(
                #     self.args.model_path,
                #     "model_{}_{}".format(self.timestamp, epoch) + ".pt",
                # )
                model_path = os.path.join(
                    self.args.model_path,
                    "model" + ".pt",
                )
                torch.save(self.model.state_dict(), model_path)

    def train_ipu(self, epochs):
        for epoch in range(epochs):
            logger.log("EPOCH {}:".format(epoch + 1))
            with timer.Timer(logger_fn=logger.log):
                for i, data in enumerate(self.training_loader):
                    inputs, labels = data
                    output, loss = self.model(inputs, labels)
                logger.log("LOSS train {}".format(loss))

        self.model.detachFromDevice()

        model_path = os.path.join(
            self.args.model_path,
            "model" + ".pt",
        )
        torch.save(self.model.state_dict(), model_path)
