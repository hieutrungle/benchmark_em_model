import torch

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn.functional as F
import logger
import os
import timer
import time
import torch.distributed as dist


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

    def average_gradients(self, model):
        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

    def reduce_dict(self, input_dict, average=True):
        world_size = float(dist.get_world_size())
        names, values = [], []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict

    @timer.Timer(logger_fn=logger.log)
    def train_one_epoch(self, epoch_index):
        running_loss = 0.0
        scaler = torch.cuda.amp.GradScaler()
        device = torch.device(f"cuda:{dist.get_rank()}")

        for i, data in enumerate(self.training_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels).float()
            scaler.scale(loss).backward()
            # self.average_gradients(self.model)
            scaler.step(self.optimizer)
            scaler.update()

            running_loss += loss.item()

        running_loss = running_loss / (i + 1)  # loss per batch

        return running_loss

    def train(self, epochs):
        best_vloss = float("inf")
        for epoch in range(epochs):
            logger.log("EPOCH {}:".format(epoch + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch)

            running_vloss = 0.0
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

            self.writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch + 1,
            )
            self.writer.flush()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = os.path.join(
                    self.args.model_path,
                    "model" + ".pt",
                )
                torch.save(self.model.state_dict(), model_path)

    def train_ipu(self, epochs):
        for epoch in range(epochs):
            logger.log("EPOCH {}:".format(epoch + 1))
            with timer.Timer(logger_fn=logger.log):
                loss = float("inf")
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
