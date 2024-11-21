import numpy as np
import torch

from dataset import FastTextModelDataset
from pytorch_model import FastTextModel
from tokenizer import NGramTokenizer


class torchFastText:
    def __init__(
        self,
        num_buckets,
        embedding_dim,
        num_classes,
        min_count,
        min_n,
        max_n,
        len_word_ngrams,
        sparse,
    ):
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.min_count = min_count
        self.min_n = min_n
        self.max_n = max_n
        self.len_word_ngrams = len_word_ngrams
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.sparse = sparse

    def build_tokenizer(self, training_text):
        self.tokenizer = NGramTokenizer(
            self.min_count,
            self.min_n,
            self.max_n,
            self.num_buckets,
            self.len_word_ngrams,
            training_text,
        )

    def build(self, training_text, categorical_variables):
        self.__build_tokenizer(training_text)
        self.pytorch_model = FastTextModel(
            tokenizer=self.tokenizer,
            embedding_dim=self.embedding_dim,
            vocab_size=self.num_buckets + self.tokenizer.get_nwords() + 1,
            num_classes=self.num_classes,
            categorical_vocabulary_sizes=np.max(categorical_variables, axis=0) + 1,
            padding_idx=self.num_buckets + self.tokenizer.get_nwords(),
            sparse=self.sparse,
        )

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs,
        batch_size,
        num_workers,
        train_proportion,
        lr,
        cpu_run=False,
        verbose=False,
    ):
        assert isinstance(
            X_train, np.ndarray
        ), "X_train must be a numpy array of shape (N,d), with the first column being the text and the rest being the categorical variables."

        try:
            training_text = X_train[:, 0].astype(str)
        except ValueError:
            print("The first column of X_train must be castable in string format.")

        try:
            categorical_variables = X_train[:, 1:].astype(int)
        except ValueError:
            print(
                f"Columns {1} to {X_train.shape[1]-1} of X_train must be castable in integer format."
            )

        if self.tokenizer is None or self.pytorch_model is None:
            if verbose:
                print("Building the model...")
            self.build(training_text, categorical_variables)

        train_dataset = FastTextModelDataset(
            categorical_variables=categorical_variables,
            texts=X_train,
            outputs=y_train,
            tokenizer=self.tokenizer,
        )
        val_dataset = FastTextModelDataset(
            categorical_variables=categorical_variables,
            texts=X_val,
            outputs=y_val,
            tokenizer=self.tokenizer,
        )

        if cpu_run:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # train_dataloader = train_dataset.create_dataloader(
        #     batch_size=batch_size, num_workers=num_workers
        # )
        # val_dataloader = val_dataset.create_dataloader(
        #     batch_size=batch_size, num_workers=num_workers
        # )
        # self.model = self.pytorch_model.to(device)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        # self.criterion = torch.nn.CrossEntropyLoss()

        return train_dataset, val_dataset
