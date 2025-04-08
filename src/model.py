"""Primarily for the LightningModule that contains the model.

Other network modules, etc. can go here or not, depending on how confusing it is.
"""

from typing import Any

import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics.regression import R2Score


class DotProductRegression(pl.LightningModule):
    """Super simplified two-tower model for regression problem."""

    def __init__(
        self,
        morphers: dict[str, dict[str, Any]],
        embedder_size: int,
        optimizer_params: dict[str, Any],
    ):
        """Initialize the model."""
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.optimizer_params = optimizer_params

        self.input_embedders = nn.ModuleDict(
            {
                feature: morpher.make_embedding(embedder_size)
                for feature, morpher in morphers["input"].items()
            }
        )
        self.input_projector = nn.Sequential(
            nn.LayerNorm(embedder_size),
            nn.ReLU(),
            nn.Linear(embedder_size, embedder_size),
        )

        self.query_embedders = nn.ModuleDict(
            {
                feature: morpher.make_embedding(embedder_size)
                for feature, morpher in morphers["query"].items()
            }
        )
        self.query_projector = nn.Sequential(
            nn.LayerNorm(embedder_size),
            nn.ReLU(),
            nn.Linear(embedder_size, embedder_size),
        )

        # Metric
        self.r2 = R2Score()

    def configure_optimizers(self):
        """Lightning hook for optimizer setup.

        Body here is just an example, although it probably works okay for a lot of things.
        We can't pass arguments to this directly, so they need to go to the init.
        """

        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        return optimizer

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the model.

        - x is compatible with the output of data.TrainingDataset.__getitem__, literally the output
          from a dataloader.

        Returns whatever the output of the model is.
        """

        x_input = sum(
            embedder(x[feature]) for feature, embedder in self.input_embedders.items()
        )
        x_input = self.input_projector(x_input)

        x_query = sum(
            embedder(x[feature]) for feature, embedder in self.query_embedders.items()
        )
        x_query = self.query_projector(x_query)
        # We only care about direction for this one.
        x_query = nn.functional.normalize(x_query, p=2.0, dim=-1)

        # Dot product
        x = (x_input * x_query).sum(dim=-1)

        return x

    def step(self, stage: str, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generic step for training or validation, assuming they're similar.

        - stage: one of "train" or "valid"
        - x: dictionary of torch tensors, input to model, targets, etc.

        This MUST log "valid_loss" during the validation step in order to have the model checkpointing work as written.

        Returns loss as one-element tensor.
        """

        y_true = x["release_speed"]
        y_pred = self(x)

        loss = nn.functional.mse_loss(y_pred, y_true)
        self.log(f"{stage}_loss", loss)
        self.log(f"{stage}_R2", self.r2(y_pred, y_true))
        return loss

    def training_step(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Lightning hook for training."""
        return self.step("train", x)

    def validation_step(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Lightning hook for validation."""
        return self.step("valid", x)
