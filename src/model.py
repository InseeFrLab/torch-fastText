"""
FastText model implemented with Pytorch.
Integrates additional categorical features.
"""
from typing import List
import torch
from torchmetrics import Accuracy
from torch import nn
import pytorch_lightning as pl


class FastTextModel(nn.Module):
    """
    FastText Pytorch Model.
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        num_classes: int,
        categorical_vocabulary_sizes: List[int],
        padding_idx: int = 0,
        sparse: bool = True,
    ):
        """
        Constructor for the FastTextModel class.

        Args:
            embedding_dim (int): Dimension of the text embedding space.
            buckets (int): Number of rows in the embedding matrix.
            num_classes (int): Number of classes.
            categorical_vocabulary_sizes (List[int]): List of the number of
                modalities for additional categorical features.
            padding_idx (int, optional): Padding index for the text
                descriptions. Defaults to 0.
            sparse (bool): Indicates if Embedding layer is sparse.
        """
        super(FastTextModel, self).__init__()
        self.num_classes = num_classes
        self.padding_idx = padding_idx

        self.embeddings = nn.Embedding(
            embedding_dim=embedding_dim,
            num_embeddings=vocab_size,
            padding_idx=padding_idx,
            sparse=sparse,
        )
        self.categorical_embeddings = {}
        for var_idx, vocab_size in enumerate(categorical_vocabulary_sizes):
            emb = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size)
            self.categorical_embeddings[var_idx] = emb
            setattr(self, "emb_{}".format(var_idx), emb)

        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs: List[torch.LongTensor]) -> torch.Tensor:
        """
        Forward method.

        Args:
            inputs (List[torch.LongTensor]): Model inputs.

        Returns:
            torch.Tensor: Model output.
        """
        # Embed tokens
        x_1 = inputs[0]
        x_1 = self.embeddings(x_1)

        x_cat = []
        for i, (variable, embedding_layer) in enumerate(
            self.categorical_embeddings.items()
        ):
            x_cat.append(embedding_layer(inputs[i + 1]))

        # Mean of tokens
        non_zero_tokens = x_1.sum(-1) != 0
        non_zero_tokens = non_zero_tokens.sum(-1)
        x_1 = x_1.sum(dim=-2)
        x_1 /= non_zero_tokens.unsqueeze(-1)
        x_1 = torch.nan_to_num(x_1)
        
        if x_cat != []:
            x_in = x_1 + torch.stack(x_cat, dim=0).sum(dim=0)
        else:
            x_in = x_1

        # Linear layer
        z = self.fc(x_in)
        return z


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
        scheduler_interval,
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

        self.model = model
        self.loss = loss
        self.accuracy_fn = Accuracy(
            task="multiclass",
            num_classes=self.model.num_classes
        )
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.scheduler_interval = scheduler_interval

    def forward(self, inputs: List[torch.LongTensor]) -> torch.Tensor:
        """
        Perform forward-pass.

        Args:
            batch (List[torch.LongTensor]): Batch to perform forward-pass on.

        Returns (torch.Tensor): Prediction.
        """
        return self.model(inputs)

    def training_step(
        self, batch: List[torch.LongTensor], batch_idx: int
    ) -> torch.Tensor:
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
        self.log("train_loss", loss, on_epoch=True)

        return loss

    def validation_step(self, batch: List[torch.LongTensor], batch_idx: int):
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
        self.log("validation_loss", loss, on_epoch=True)

        accuracy = self.accuracy_fn(outputs, targets)
        self.log('validation_accuracy', accuracy, on_epoch=True)
        return loss

    def test_step(self, batch: List[torch.LongTensor], batch_idx: int):
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
        self.log("test_loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """
        Configure optimizer for Pytorch lighting.

        Returns: Optimizer and scheduler for pytorch lighting.
        """
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        scheduler = self.scheduler(optimizer, **self.scheduler_params)
        scheduler = {
            "scheduler": scheduler,
            "monitor": "validation_loss",
            "interval": self.scheduler_interval,
        }

        return [optimizer], [scheduler]
