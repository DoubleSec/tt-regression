"""Contains tools for initial data preprocessing and then data loading during training."""

from pybaseball import statcast
import polars as pl
import torch
from torch.utils.data import Dataset
import morphers
from logzero import logger

MORPHER_DISPATCH = {
    "categorical": morphers.Integerizer,
    "numeric": morphers.Normalizer,
}


def initial_prep(
    path: str,
    start_date: str,
    end_date: str,
) -> str:
    """Do one-time data preparation.

    Here we'll just download statcast data and save it.
    """

    init_data = statcast(start_dt=start_date, end_dt=end_date)
    init_data = pl.DataFrame(init_data)
    logger.info(f"Writing {len(init_data)} rows to {path}")
    init_data.write_parquet(path)
    return path


class TrainingDataset(Dataset):
    """Torch Dataset class for training."""

    def __init__(
        self,
        path: str,
        keys: list[str],
        input_columns: list[str],
        prepared_morphers: list[morphers.base.base.Morpher] | None = None,
    ):
        """Load a dataset and do any prep required.

        If at all possible, this should be deterministic, given a specific file.
        Note that at the moment we're ignoring input columns.
        """

        super().__init__()
        self.keys = keys
        self.input_columns = input_columns
        # Pitch name is the "query" and release speed is the "value".
        self.data = (
            pl.read_parquet(path)
            .select(
                *keys,
                *[x[0] for x in self.input_columns],
                "pitch_name",
                "release_speed",
            )
            .drop_nulls()
        )
        if prepared_morphers is not None:
            self.morphers = prepared_morphers
        else:
            self.morphers = {
                "input": {
                    feature: MORPHER_DISPATCH[dtype].from_data(
                        self.data[feature], **kwargs
                    )
                    for feature, dtype, kwargs in self.input_columns
                },
                "query": {
                    "pitch_name": morphers.Integerizer.from_data(
                        self.data["pitch_name"]
                    ),
                },
                "target": {
                    "release_speed": morphers.Normalizer.from_data(
                        self.data["release_speed"]
                    ),
                },
            }

        # Apply all the morphers. We dropped nulls earlier,
        # so we don't need fill_missing
        self.data = self.data.with_columns(
            **{
                feature: morpher(pl.col(feature))
                for feature, morpher in self.morphers["input"].items()
            },
            **{
                feature: morpher(pl.col(feature))
                for feature, morpher in self.morphers["query"].items()
            },
            **{
                target: morpher(pl.col(target))
                for target, morpher in self.morphers["target"].items()
            },
        )

    def __len__(self) -> int:
        """Obvious"""
        return len(self.data)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        """Return a single instance of the training dataset, as a dictionary."""

        row = self.data.row(idx, named=True)

        return (
            {key: row[key] for key in self.keys}
            | {
                feature: torch.tensor(row[feature], dtype=morpher.required_dtype)
                for feature, morpher in self.morphers["input"].items()
            }
            | {
                feature: torch.tensor(row[feature], dtype=morpher.required_dtype)
                for feature, morpher in self.morphers["query"].items()
            }
            | {
                target: torch.tensor(row[target], dtype=morpher.required_dtype)
                for target, morpher in self.morphers["target"].items()
            }
        )
