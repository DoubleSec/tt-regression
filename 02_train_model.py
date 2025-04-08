"""Train a model."""

import yaml
import torch
from lightning import pytorch as pl

from src.data import TrainingDataset
from src.model import DotProductRegression

if __name__ == "__main__":

    # Read the config ---------------------------------------
    # (Probably should use hydra eventually.)

    with open("cfg/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    # Hashtag reproducibility
    torch.manual_seed(config["random_seed"])

    # Generic dataset preparation ---------------------------

    ds = TrainingDataset(
        path=config["prepared_data_path"],
        input_columns=config["input_columns"],
        **config["dataset_params"],
    )
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [0.75, 0.15, 0.1])

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        shuffle=True,
        **config["dataloader_params"],
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        **config["dataloader_params"],
    )
    # For the moment we're not doing anything with the test dataset.

    # Set up the trainer ------------------------------------

    trainer = pl.Trainer(
        **config["trainer_params"],
        logger=pl.loggers.TensorBoardLogger("."),
        callbacks=pl.callbacks.ModelCheckpoint(
            dirpath="./model",
            save_top_k=1,
            monitor="valid_loss",
            filename="{epoch}-{valid_loss:.4f}",
        ),
    )

    with trainer.init_module():

        net = DotProductRegression(
            **config["model_params"],
            morphers=ds.morphers,
            optimizer_params=config["optimizer_params"],
        )
        # Maybe
        net.compile()

    torch.set_float32_matmul_precision("medium")
    trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=valid_dl)
