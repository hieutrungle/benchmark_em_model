import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime
import os
import sys
import gc
import shutil
import scheduler
import utils
import logger
import torchmetrics
from pytorch_lightning.strategies.ddp import DDPStrategy


# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# AVAILABLE_GPUS = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
# NUM_GPUS = len(AVAILABLE_GPUS)

# DEVICE = torch.device(str(os.environ.get("DEVICE", "cpu")))
# NUM_GPUS = int(os.environ.get("NUM_GPUS", 0))
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

class ModuleTrainer(pl.LightningModule):
    def __init__(self, model, lr, warmup, max_iters, momentum, weight_decay, **kwargs):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters(ignore=["model"])
        # Create model
        self.model = model

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        return self.model(x)

    def _get_loss(self, batch):
        """
        Given a batch of data, this function returns the reconstruction loss (MSE in our case)
        """
        x = batch[0]
        y = batch[1]
        y_pred = self.forward(x)
        # loss = F.mse_loss(y_pred, y).float()
        # loss = torch.sum(torch.pow(y_pred-y,2),dtype=float)
        loss = F.l1_loss(y_pred, y).float()
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.hparams.weight_decay,
            amsgrad=False,
        )

        lr_scheduler = scheduler.CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        train_loss = self._get_loss(batch)
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        self.log("train_loss", train_loss, on_epoch=True, sync_dist=True)
        self.log("hp/train_loss", train_loss, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self._get_loss(batch)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("hp/val_loss", val_loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        test_loss = self._get_loss(batch)
        self.log("test_loss", test_loss, on_epoch=True, sync_dist=True)

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {"hp/train_loss": 0, "hp/val_loss": 0},
        )


def train(model, train_loader, val_loader, args):
    """train the model"""

    model_args = dict(
        lr=args.lr,
        warmup=args.epochs * len(train_loader) * args.warm_up_portion,
        max_iters=args.epochs * len(train_loader),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Save training parameters if we need to resume training in the future
    weight_filename = "sst-{epoch:03d}-{val_loss:.5f}"
    # if "resume_epoch" in resume_checkpoint:
    #     start_epoch = resume_checkpoint["resume_epoch"]
    #     weight_filename = f"resume_start_{start_epoch}_" + weight_filename
    #     version = "resume"
    # else:
    version = "pretrain"
    summaries_dir, checkpoints_dir = utils.mkdir_storage(args.model_path)
    _callbacks = get_callbacks(checkpoints_dir, weight_filename, verbose=args.verbose)

    logger.log(f"\nStart Training...")
    start_total_time = time.perf_counter()
    tfboard_logger = pl.loggers.tensorboard.TensorBoardLogger(
        summaries_dir, name="", version=version, log_graph=True, default_hp_metric=False
    )

    trainer = pl.Trainer(
        fast_dev_run=False,
        default_root_dir=os.path.join(checkpoints_dir),
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_interval,
        logger=tfboard_logger,
        callbacks=_callbacks,
        reload_dataloaders_every_n_epochs=2,
        # limit_val_batches=0.05,
        # limit_train_batches=0.05,
        gradient_clip_algorithm="norm",
        enable_progress_bar=args.verbose,
        strategy=DDPStrategy(find_unused_parameters=False),
    )
    lightning_model = ModuleTrainer(
        model=model,
        **model_args,
    )
    trainer.fit(lightning_model, train_loader, val_loader)
    total_training_time = time.perf_counter() - start_total_time
    logger.log(f"Training time: {total_training_time:0.2f} seconds")

    for i, (path, _) in enumerate(trainer.checkpoint_callback.best_k_models.items()):
        m = ModuleTrainer.load_from_checkpoint(path, model=model)
        torch.save(m.model.state_dict(), path.rpartition(".")[0] + ".pt")
    model = model.to(DEVICE)

    # Test best model on validation and test set
    logger.log(f"Loading best model from {_callbacks[1].best_model_path}")
    lightning_model = ModuleTrainer.load_from_checkpoint(
        _callbacks[1].best_model_path, model=model
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        enable_progress_bar=args.verbose,
    )
    val_result = trainer.test(lightning_model, val_loader, verbose=args.verbose)
    
    logger.log(f"\n{val_result}\n")

    gc.collect()
    logger.log(f"\nTraining completed!\n")

    return lightning_model.model


def get_callbacks(
    checkpoints_dir, weight_filename="{epoch:03d}-{train_loss:.2f}", verbose=False
):
    callbacks = [
        EarlyStopping("val_loss", patience=25, mode="min"),
        ModelCheckpoint(
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            dirpath=checkpoints_dir,
            filename=weight_filename,
            save_weights_only=True,
        ),
        LearningRateMonitor("step"),
    ]
    if verbose:
        callbacks.append(TQDMProgressBar(refresh_rate=500))
    return callbacks
