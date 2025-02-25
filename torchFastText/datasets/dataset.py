"""
Dataset class for a FastTextModel without the fastText dependency.
"""

import os
import logging
from typing import List

import torch

from .tokenizer import NGramTokenizer

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


class FastTextModelDataset(torch.utils.data.Dataset):
    """
    FastTextModelDataset class.
    """

    def __init__(
        self,
        categorical_variables: List[List[int]],
        texts: List[str],
        outputs: List[int],
        tokenizer: NGramTokenizer,
        **kwargs,
    ):
        """
        Constructor for the TorchDataset class.

        Args:
            categorical_variables (List[List[int]]): The elements of this list
                are the values of each categorical variable across the dataset.
            text (List[str]): List of text descriptions.
            y (List[int]): List of outcomes.
            tokenizer (Tokenizer): Tokenizer.
        """
        self.categorical_variables = categorical_variables
        self.texts = texts
        self.outputs = outputs
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """
        Returns length of the data.

        Returns:
            int: Number of observations.
        """
        return len(self.outputs)

    def __str__(self) -> str:
        """
        Returns description of the Dataset.

        Returns:
            str: Description.
        """
        return f"<FastTextModelDataset(N={len(self)})>"

    def __getitem__(self, index: int) -> List:
        """
        Returns observation for a given index.

        Args:
            index (int): Index.

        Returns:
            List[int, str]: Observation with given index.
        """
        categorical_variables = (
            self.categorical_variables[index] if self.categorical_variables is not None else None
        )
        text = self.texts[index]
        y = self.outputs[index]
        return text, categorical_variables, y

    def collate_fn(self, batch):
        """
        Efficient batch processing without explicit loops.

        Args:
            batch: Data batch.

        Returns:
            Tuple[torch.LongTensor]: Observation with given index.
        """
        # Unzip the batch in one go using zip(*batch)
        text, *categorical_vars, y = zip(*batch)

        # Convert text to indices in parallel using map
        indices_batch = list(map(lambda x: self.tokenizer.indices_matrix(x)[0], text))

        # Get padding index once
        padding_index = self.tokenizer.get_buckets() + self.tokenizer.get_nwords()

        # Pad sequences efficiently
        padded_batch = torch.nn.utils.rnn.pad_sequence(
            indices_batch,
            batch_first=True,
            padding_value=padding_index,
        )

        # Handle categorical variables efficiently
        if self.categorical_variables is not None:
            categorical_tensors = torch.stack(
                [
                    torch.tensor(cat_var, dtype=torch.float32)
                    for cat_var in categorical_vars[
                        0
                    ]  # Access first element since zip returns tuple
                ]
            )
        else:
            categorical_tensors = torch.empty(
                padded_batch.shape[0], 1, dtype=torch.float32, device=padded_batch.device
            )

        # Convert labels to tensor in one go
        y = torch.tensor(y, dtype=torch.long)

        return (padded_batch, categorical_tensors, y)

    def create_dataloader(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = os.cpu_count() - 1,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        """
        Creates a Dataloader.

        Args:
            batch_size (int): Batch size.
            shuffle (bool, optional): Shuffle option. Defaults to False.
            drop_last (bool, optional): Drop last option. Defaults to False.

        Returns:
            torch.utils.data.DataLoader: Dataloader.
        """

        logger.info(f"Creating DataLoader with {num_workers} workers.")

        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True,
        )
