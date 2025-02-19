import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from .pytorch_model import FastTextModel


class FastTextModule(pl.LightningModule):
    """
    Pytorch Lightning Module for FastTextModel.
    """

    def __init__(
        self,
        model: FastTextModel,
        loss,
        optimizer,
        optimizer_params,
        scheduler,
        scheduler_params,
        scheduler_interval="epoch",
        **kwargs,
    ):
        """
        Initialize FastTextModule.

        Args:
            model: Model.
            loss: Loss
            optimizer: Optimizer
            optimizer_params: Optimizer parameters.
            scheduler: Scheduler.
            scheduler_params: Scheduler parameters.
            scheduler_interval: Scheduler interval.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss"])

        self.model = model
        self.loss = loss
        self.accuracy_fn = Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.scheduler_interval = scheduler_interval

    def forward(self, inputs) -> torch.Tensor:
        """
        Perform forward-pass.

        Args:
            batch (List[torch.LongTensor]): Batch to perform forward-pass on.

        Returns (torch.Tensor): Prediction.
        """
        return self.model(inputs[0], inputs[1])

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch (List[torch.LongTensor]): Training batch.
            batch_idx (int): Batch index.

        Returns (torch.Tensor): Loss tensor.
        """
        
        inputs, targets = batch[:-1], batch[-1]
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        accuracy = self.accuracy_fn(outputs, targets)
        self.log("train_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=True)
        
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx: int):
        """
        Validation step.

        Args:
            batch (List[torch.LongTensor]): Validation batch.
            batch_idx (int): Batch index.

        Returns (torch.Tensor): Loss tensor.
        """
        inputs, targets = batch[:-1], batch[-1]
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

        accuracy = self.accuracy_fn(outputs, targets)
        self.log("val_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx: int):
        """
        Test step.

        Args:
            batch (List[torch.LongTensor]): Test batch.
            batch_idx (int): Batch index.

        Returns (torch.Tensor): Loss tensor.
        """
        inputs, targets = batch[:-1], batch[-1]
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)

        accuracy = self.accuracy_fn(outputs, targets)

        return loss, accuracy

    def configure_optimizers(self):
        """
        Configure optimizer for Pytorch lighting.

        Returns: Optimizer and scheduler for pytorch lighting.
        """
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        scheduler = self.scheduler(optimizer, **self.scheduler_params)
        scheduler = {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "interval": self.scheduler_interval,
        }

        return [optimizer], [scheduler]
