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
        tokenizer: NGramTokenizer,
        outputs: List[int] = None,
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

        if categorical_variables is not None and len(categorical_variables) != len(texts):
            raise ValueError("Categorical variables and texts must have the same length.")
        
        if outputs is not None and len(outputs) != len(texts):
            raise ValueError("Outputs and texts must have the same length.")
            
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
        return len(self.texts)

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

        if self.outputs is not None:
            y = self.outputs[index]
            return text, categorical_variables, y
        else:
            return text, categorical_variables

    def collate_fn(self, batch):
        """
        Efficient batch processing without explicit loops.

        Args:
            batch: Data batch.

        Returns:
            Tuple[torch.LongTensor]: Observation with given index.
        """

        # Unzip the batch in one go using zip(*batch)
        if self.outputs is not None:
            text, *categorical_vars, y = zip(*batch)
        else:
            text, *categorical_vars = zip(*batch)

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

        if self.outputs is not None:
            # Convert labels to tensor in one go
            y = torch.tensor(y, dtype=torch.long)
            return (padded_batch, categorical_tensors, y)
        else:
            return (padded_batch, categorical_tensors)

    def create_dataloader(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = os.cpu_count() - 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        """
        Creates a Dataloader from the FastTextModelDataset.
        Use collate_fn() to tokenize and pad the sequences.

        Args:
            batch_size (int): Batch size.
            shuffle (bool, optional): Shuffle option. Defaults to False.
            drop_last (bool, optional): Drop last option. Defaults to False.
            num_workers (int, optional): Number of workers. Defaults to os.cpu_count() - 1.
            pin_memory (bool, optional): Set True if working on GPU, False if CPU. Defaults to True.
            persistent_workers (bool, optional): Set True for training, False for inference. Defaults to True.
            **kwargs: Additional arguments for PyTorch DataLoader.

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
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            **kwargs,
        )
