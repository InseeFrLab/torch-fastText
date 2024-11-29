"""
Dataset class for a FastTextModel without the fastText dependency.
"""

from typing import List

import numpy as np
import torch
from datasets.tokenizer import NGramTokenizer


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
        categorical_variables = self.categorical_variables[index]
        text = self.texts[index]
        y = self.outputs[index]
        return text, categorical_variables, y

    def collate_fn(self, batch):
        """
        Processing on a batch.

        Args:
            batch: Data batch.

        Returns:
            Tuple[torch.LongTensor]: Observation with given index.
        """
        # Get inputs
        text = [item[0] for item in batch]
        categorical_variables = [item[1] for item in batch]
        y = [item[-1] for item in batch]

        indices_batch = [self.tokenizer.indices_matrix(sentence)[0] for sentence in text]

        padding_index = self.tokenizer.get_buckets() + self.tokenizer.get_nwords()
        padded_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(indices) for indices in indices_batch],
            batch_first=True,
            padding_value=padding_index,
        )

        categorical_tensors = torch.Tensor(
            np.vstack(categorical_variables)
        )  # (batch_size, num_categorical_features)

        y = torch.LongTensor(y)

        return (padded_batch, categorical_tensors, y)

    def create_dataloader(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
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
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
            num_workers=num_workers,
        )