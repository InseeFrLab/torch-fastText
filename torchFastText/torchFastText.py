import logging
import time
import json
from typing import Optional, Union, Type, List
from dataclasses import dataclass, field, asdict

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.optim import SGD, Adam

from .datasets.dataset import FastTextModelDataset
from .datasets.tokenizer import NGramTokenizer
from .model.pytorch_model import FastTextModel
from .model.lightning_module import FastTextModule
from .utilities.checkers import check_X, check_Y, NumpyJSONEncoder

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


@dataclass
class torchFastText:
    """
    The main class for the torchFastText model.

    Args for init:
        Architecture-related:
            Text-embedding matrix:
                num_tokens (int): Number of rows in the embedding matrix (without counting the word-rows)
                embedding_dim (int): Dimension of the embedding matrix
                sparse (bool): Whether the embedding matrix is sparse
            Categorical variables, if any:
                categorical_vocabulary_sizes (List[int]): List of the number of unique values for each categorical variable.
                    - Do not provide if no categorical variables
                    - Do not provide if you want this to be inferred from X_train in the build method
                categorical_embedding_dims (Union[List[int], int]): List of the embedding dimensions for each categorical variable.
                    - Do not provide if no categorical variables
                    - Do not provide if there are and the embeddings should be summed to the sentence embedding
                    - Provide an int if all embeddings should be of the same dimension, averaged and concatened to the sentence embedding
                    - Provide a list of int if each embedding should be of a different dimension, concatenated without aggregation to the sentence embedding
                num_categorical_features (int): Number of categorical variables.
                    - Not required, should match the length of the above mentioned lists
                    - Especially useful when you do not provide vocabulary sizes and an int as categorical_embedding_dims

            Tokenizer-related:
                min_count (int): Minimum number of times a word has to be in the training data to be given an embedding.
                min_n (int): Minimum length of character n-grams.
                max_n (int): Maximum length of character n-grams.
                len_word_ngrams (int): Maximum length of word n-grams.

        Other attributes, not exposed during initialization:
            tokenizer (NGramTokenizer): Tokenizer object, see build_from_tokenizer


    """

    # Required parameters

    # Embedding matrix
    embedding_dim: int
    sparse: bool

    # Tokenizer-related
    num_tokens: int
    min_count: int
    min_n: int
    max_n: int
    len_word_ngrams: int

    # Optional parameters with default values
    num_classes: Optional[int] = None
    num_rows: Optional[int] = (
        None  # Default = num_tokens + tokenizer.get_nwords() + 1, but can be customized
    )

    # Embedding matrices of categorical variables
    categorical_vocabulary_sizes: Optional[List[int]] = None
    categorical_embedding_dims: Optional[Union[List[int], int]] = None
    num_categorical_features: Optional[int] = None

    # Internal fields (not exposed during initialization)
    tokenizer: Optional[NGramTokenizer] = field(init=True, default=None)
    pytorch_model: Optional[FastTextModel] = field(init=False, default=None)
    lightning_module: Optional[FastTextModule] = field(init=True, default=None)
    trained: bool = field(init=False, default=False)

    direct_bagging: Optional[bool] = True  # Use nn.EmbeddingBag instead of nn.Embedding

    def _build_pytorch_model(self):
        if self.num_rows is None:
            if self.tokenizer is None:
                raise ValueError(
                    "Please provide a tokenizer (for instance using model.build_tokenizer()) or num_rows."
                )

            else:
                self.num_rows = self.tokenizer.padding_index + 1

        else:
            if self.tokenizer is not None:
                if self.num_rows != self.tokenizer.padding_index + 1:
                    logger.warning(
                        f"""Divergent values for num_rows: {self.num_rows} and {self.tokenizer.padding_index + 1} (tokenizer's padding index).
                        It is set to the max. The padding index will be updated (Always set to num_rows - 1)."""
                    )
                self.num_rows = max(self.num_rows, self.tokenizer.padding_index + 1)

        self.padding_idx = self.num_rows - 1

        self.pytorch_model = FastTextModel(
            tokenizer=self.tokenizer,
            embedding_dim=self.embedding_dim,
            num_rows=self.num_rows,
            num_classes=self.num_classes,
            categorical_vocabulary_sizes=self.categorical_vocabulary_sizes,
            categorical_embedding_dims=self.categorical_embedding_dims,
            padding_idx=self.padding_idx,
            sparse=self.sparse,
            direct_bagging=self.direct_bagging,
        )

    def _check_and_init_lightning(
        self,
        optimizer=None,
        optimizer_params=None,
        lr=None,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params=None,
        patience_scheduler=3,
        loss=torch.nn.CrossEntropyLoss(),
    ):
        if optimizer is None:
            if lr is None:
                raise ValueError("Please provide a learning rate")
            self.optimizer = SGD if self.sparse else Adam
            self.optimizer_params = {"lr": lr}
        else:
            self.optimizer = optimizer
            if optimizer_params is None:
                if lr is not None:
                    self.optimizer_params = {"lr": lr}
                else:
                    logger.warning(
                        "No optimizer parameters provided, nor learning rate. Using default parameters"
                    )
                    self.optimizer_params = {}

        self.scheduler = scheduler

        if scheduler_params is None:
            logger.warning(
                "No scheduler parameters provided. Using default parameters (suited for ReduceLROnPlateau)."
            )
            self.scheduler_params = {
                "mode": "min",
                "patience": patience_scheduler,
            }
        else:
            self.scheduler_params = scheduler_params

        self.loss = loss

        self.lightning_module = FastTextModule(
            model=self.pytorch_model,
            loss=self.loss,
            optimizer=self.optimizer,
            optimizer_params=self.optimizer_params,
            scheduler=self.scheduler,
            scheduler_params=self.scheduler_params,
            scheduler_interval="epoch",
        )

    @classmethod
    def build_from_tokenizer(
        cls: Type["torchFastText"],
        tokenizer: NGramTokenizer,
        embedding_dim: int,
        num_classes: Optional[int],
        categorical_vocabulary_sizes: Optional[List[int]],
        sparse: bool = False,
        categorical_embedding_dims: Optional[Union[List[int], int]] = None,
        num_categorical_features: Optional[int] = None,
        lightning=True,
        optimizer=None,
        optimizer_params: Optional[dict] = None,
        lr=None,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params: Optional[dict] = None,
        patience_scheduler=3,
        loss=torch.nn.CrossEntropyLoss(),
    ) -> "torchFastText":
        """
        Alternative constructor that initializes torchFastText from a tokenizer.
        Directly builds the PyTorch model and Lightning module (if lightning == True).

        Args:
            tokenizer: A NGramTokenizer object that provides min_n, max_n, and other variables.
            Refer to the NGramTokenizer, FastTextModule and above constructor for mthe other variables.

        Returns:
            torchFastText: An instance of torchFastText initialized using the tokenizer.
        """
        # Ensure the tokenizer has the required attributes
        if not all(
            hasattr(tokenizer, attr)
            for attr in ["min_count", "min_n", "max_n", "num_tokens", "word_ngrams"]
        ):
            raise ValueError(f" Attr missing in tokenizer: {tokenizer}")

        # Extract attributes from the tokenizer
        min_count = tokenizer.min_count
        min_n = tokenizer.min_n
        max_n = tokenizer.max_n
        num_tokens = tokenizer.num_tokens
        len_word_ngrams = tokenizer.word_ngrams

        wrapper = cls(
            num_tokens=num_tokens,
            embedding_dim=embedding_dim,
            min_count=min_count,
            min_n=min_n,
            max_n=max_n,
            len_word_ngrams=len_word_ngrams,
            sparse=sparse,
            num_classes=num_classes,
            categorical_vocabulary_sizes=categorical_vocabulary_sizes,
            categorical_embedding_dims=categorical_embedding_dims,
            num_categorical_features=num_categorical_features,
            tokenizer=tokenizer,
        )

        wrapper._build_pytorch_model()

        if lightning:
            wrapper._check_and_init_lightning(
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr=lr,
                scheduler=scheduler,
                scheduler_params=scheduler_params,
                patience_scheduler=patience_scheduler,
                loss=loss,
            )
        return wrapper

    def build_tokenizer(self, training_text):
        self.tokenizer = NGramTokenizer(
            self.min_count,
            self.min_n,
            self.max_n,
            self.num_tokens,
            self.len_word_ngrams,
            training_text,
        )

    def build(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray = None,
        lightning=True,
        optimizer=None,
        optimizer_params=None,
        lr=None,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params=None,
        patience_scheduler=3,
        loss=torch.nn.CrossEntropyLoss(),
    ):
        """
        Public method that acts as a wrapper to build model, inferring from training data if necessary.

        Args:
            X_train (np.ndarray): Array of shape (N,d) with the first column being the text and the rest being the categorical variables.
            y_train (np.ndarray): Array of shape (N,) with the labels. Optional if num_classes is provided at initialization (overwrites it otherwise).
            lightning (bool): Whether to build the Lightning module. Default is True.
            optimizer: Optimizer to use. If None, "lr" must be provided, SGD or Adam will be used whether self.sparse is True or False.
            optimizer_params: Dictionary containing optimizer parameters. If None, "lr" will be used as the learning rate.
            lr (float): Learning rate. Required if optimizer is None.
            scheduler: Scheduler to use. Default is ReduceLROnPlateau.
            scheduler_params: Dictionary containing scheduler parameters. Default is {"mode": "min", "patience": patience_scheduler}, well-suited for ReduceLROnPlateau.
            patience_scheduler (int): Patience for the scheduler. Default is 3.
            loss: Loss function to use. Default is CrossEntropyLoss.
        """
        training_text, categorical_variables, no_cat_var = check_X(X_train)

        if y_train is not None:
            if self.num_classes is not None:
                if self.num_classes != len(np.unique(y_train)):
                    logger.warning(
                        f"Old num_classes value is {self.num_classes}. New num_classes value is {len(np.unique(y_train))}."
                    )

            y_train = check_Y(y_train)
            self.num_classes = len(
                np.unique(y_train)
            )  # Be sure that y_train contains all the classes !
        else:
            if self.num_classes is None:
                raise ValueError(
                    "Either num_classes must be provided at init or y_train must be provided here."
                )

        if not no_cat_var:
            if self.num_categorical_features is not None:
                if self.num_categorical_features != categorical_variables.shape[1]:
                    logger.warning(
                        f"num_categorical_features: old value is {self.num_categorical_features}. New value is {categorical_variables.shape[1]}."
                    )

            self.num_categorical_features = categorical_variables.shape[1]

            categorical_vocabulary_sizes = np.max(categorical_variables, axis=0) + 1

            if self.categorical_vocabulary_sizes is not None:
                if self.categorical_vocabulary_sizes != list(categorical_vocabulary_sizes):
                    logger.warning(
                        "categorical_vocabulary_sizes was provided at initialization. It will be overwritten by the unique values in the training data."
                    )
            self.categorical_vocabulary_sizes = list(categorical_vocabulary_sizes)
        else:
            if self.categorical_vocabulary_sizes is not None:
                logger.warning(
                    "categorical_vocabulary_sizes was provided at initialization but no categorical variables are provided in X_train. Updating to None."
                )
                self.categorical_vocabulary_sizes = None
            if self.num_categorical_features is not None:
                logger.warning(
                    "num_categorical_features was provided at initialization but no categorical variables are provided in X_train. Updating to None."
                )
                self.num_categorical_features = None

        self.build_tokenizer(training_text)
        self._build_pytorch_model()

        if lightning:
            self._check_and_init_lightning(
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr=lr,
                scheduler=scheduler,
                scheduler_params=scheduler_params,
                patience_scheduler=patience_scheduler,
                loss=loss,
            )

    def build_data_loaders(self, X_train, y_train, X_val, y_val, batch_size, num_workers):
        """
        A public method to build the dataloaders, with few arguments and running checks.

        Args:
            X_train (np.ndarray): Array of shape (N,d) with the first column being the text and the rest being the categorical variables.
            y_train (np.ndarray): Array of shape (N,) with the labels.
            X_val (np.ndarray): Array of shape (N,d) with the first column being the text and the rest being the categorical variables.
            y_val (np.ndarray): Array of shape (N,) with the labels.
            batch_size (int): Batch size.
            num_workers (int): Number of workers for the dataloaders.

        Returns:
            Tuple[torch.utils.data.DataLoader]: Training and validation dataloaders.

        """

        training_text, train_categorical_variables, train_no_cat_var = check_X(X_train)
        val_text, val_categorical_variables, val_no_cat_var = check_X(X_val)
        y_train = check_Y(y_train)
        y_val = check_Y(y_val)

        # Datasets and dataloaders
        train_dataset = FastTextModelDataset(
            categorical_variables=train_categorical_variables,
            texts=training_text,
            outputs=y_train,
            tokenizer=self.tokenizer,
        )
        val_dataset = FastTextModelDataset(
            categorical_variables=val_categorical_variables,
            texts=val_text,
            outputs=y_val,
            tokenizer=self.tokenizer,
        )

        train_dataloader = train_dataset.create_dataloader(
            batch_size=batch_size, num_workers=num_workers
        )
        val_dataloader = val_dataset.create_dataloader(
            batch_size=batch_size, num_workers=num_workers
        )

        return train_dataloader, val_dataloader

    def __build_data_loaders(
        self,
        train_categorical_variables,
        training_text,
        y_train,
        val_categorical_variables,
        val_text,
        y_val,
        batch_size,
        num_workers,
    ):
        """
        A private method to build the dataloaders, without running checks.
        Used in train method (where checks are run beforehand).

        Args:
            train_categorical_variables (np.ndarray): Array of shape (N_train,d-1) with the categorical variables.
            training_text (np.ndarray): Array of shape (N_train,) with the text in string format
            y_train (np.ndarray): Array of shape (N_train,) with the labels.
            val_categorical_variables (np.ndarray): Array of shape (N_val,d-1) with the categorical variables.
            val_text (np.ndarray): Array of shape (N_val,) with the text in string format
            y_val (np.ndarray): Array of shape (N_val,) with the labels.
            batch_size (int): Batch size.
            num_workers (int): Number of workers for the dataloaders.

        Returns:
            Tuple[torch.utils.data.DataLoader]: Training and validation dataloaders.
        """

        # Datasets and dataloaders
        train_dataset = FastTextModelDataset(
            categorical_variables=train_categorical_variables,
            texts=training_text,
            outputs=y_train,
            tokenizer=self.tokenizer,
        )
        val_dataset = FastTextModelDataset(
            categorical_variables=val_categorical_variables,
            texts=val_text,
            outputs=y_val,
            tokenizer=self.tokenizer,
        )

        train_dataloader = train_dataset.create_dataloader(
            batch_size=batch_size, num_workers=num_workers
        )
        val_dataloader = val_dataset.create_dataloader(
            batch_size=batch_size, num_workers=num_workers
        )

        return train_dataloader, val_dataloader

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_epochs: int,
        batch_size: int,
        cpu_run: bool = False,
        num_workers: int = 12,
        optimizer=None,
        optimizer_params=None,
        lr: float = None,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        patience_scheduler: int = 3,
        loss=torch.nn.CrossEntropyLoss(),
        patience_train=3,
        verbose: bool = False,
        trainer_params: Optional[dict] = None,
    ):
        """
        Trains the FastText model using the provided training and validation data.
        This method checks the provided inputs for consistency and correctness, builds the model if necessary,
        creates data loaders for the training and validation datasets, and initiates the training process using a PyTorch Lightning trainer.
        After training, it loads the best model checkpoint, moves the model to CPU, and sets the model to evaluation mode.
        Parameters:
            X_train (np.ndarray): Array containing the training data features.
            y_train (np.ndarray): Array containing the training data labels.
            X_val (np.ndarray): Array containing the validation data features.
            y_val (np.ndarray): Array containing the validation data labels.
            num_epochs (int): The maximum number of epochs for the training process.
            batch_size (int): The size of the mini-batches used during training.
            cpu_run (bool, optional): Whether to force the training to run on the CPU. Defaults to False.
            num_workers (int, optional): Number of worker threads for data loading. Defaults to 12.
            optimizer (optional): A PyTorch optimizer to use for training. Must be from torch.optim if provided.
            optimizer_params (optional): Additional parameters for configuring the optimizer.
            lr (float, optional): The learning rate for the optimizer.
            scheduler (optional): A PyTorch learning rate scheduler. Defaults to torch.optim.lr_scheduler.ReduceLROnPlateau.
            patience_scheduler (int, optional): Number of epochs with no improvement after which learning rate will be reduced. Defaults to 3.
            loss (torch.nn.Module, optional): The loss function to optimize. Must be an instance of a PyTorch loss module. Defaults to torch.nn.CrossEntropyLoss().
            patience_train (int, optional): Number of epochs with no improvement for early stopping. Defaults to 3.
            verbose (bool, optional): Flag to enable verbose logging. Defaults to False.
            trainer_params (Optional[dict], optional): Additional parameters to be passed to the PyTorch Lightning Trainer.
        Returns:
            None
        Side Effects:
            - Builds the tokenizer and PyTorch model if not already built.
            - Creates training and validation data loaders.
            - Initiates training using a PyTorch Lightning Trainer.
            - Saves the best model checkpoint path in `self.best_model_path`.
            - Loads the best model and sets it to evaluation mode.
            - Sets `self.trained` to True upon successful training.
        Raises:
            AssertionError: If provided loss is not a PyTorch loss module, if optimizer is not from torch.optim,
                            or if scheduler is not a PyTorch learning rate scheduler.
            AssertionError: If there is a mismatch in the number of observations or the number of columns between the training and validation datasets.
        """
        ##### Formatting exception handling #####

        assert isinstance(loss, torch.nn.Module), "loss must be a PyTorch loss function."
        assert optimizer is None or optimizer.__module__.startswith("torch.optim"), (
            "optimizer must be a PyTorch optimizer."
        )
        assert scheduler.__module__ == "torch.optim.lr_scheduler", (
            "scheduler must be a PyTorch scheduler."
        )

        # checking right format for inputs
        if verbose:
            logger.info("Checking inputs...")

        training_text, train_categorical_variables, train_no_cat_var = check_X(X_train)
        val_text, val_categorical_variables, val_no_cat_var = check_X(X_val)
        y_train = check_Y(y_train)
        y_val = check_Y(y_val)

        # some checks
        assert train_no_cat_var == val_no_cat_var, (
            "X_train and X_val must have the same number of categorical variables."
        )
        # shape
        assert X_train.shape[0] == y_train.shape[0], (
            "X_train and y_train must have the same first dimension (number of observations)."
        )
        assert X_train.ndim > 1 and X_train.shape[1] == X_val.shape[1] or X_val.ndim == 1, (
            "X_train and X_val must have the same number of columns."
        )

        self.no_cat_var = train_no_cat_var

        if verbose:
            logger.info("Inputs successfully checked. Starting the training process..")

        ######## Starting the training process ########

        # Device
        if cpu_run:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if verbose:
            logger.info(f"Running on: {self.device}")

        # Build tokenizer PyTorch model (using training text and categorical variables)
        if self.tokenizer is None or self.pytorch_model is None:
            if verbose:
                start = time.time()
                logger.info("Building the model...")
            self.build(
                X_train,
                y_train,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr=lr,
                scheduler=scheduler,
                patience_scheduler=patience_scheduler,
                loss=loss.to(self.device),
            )
            if verbose:
                end = time.time()
                logger.info("Model successfully built in {:.2f} seconds.".format(end - start))

        self.pytorch_model = self.pytorch_model.to(self.device)

        # Dataloaders
        train_dataloader, val_dataloader = self.__build_data_loaders(
            train_categorical_variables=train_categorical_variables,
            training_text=training_text,
            y_train=y_train,
            val_categorical_variables=val_categorical_variables,
            val_text=val_text,
            y_val=y_val,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        if verbose:
            logger.info("Lightning module successfully created.")

        # Trainer callbacks
        checkpoints = [
            {
                "monitor": "val_loss",
                "save_top_k": 1,
                "save_last": False,
                "mode": "min",
            }
        ]
        callbacks = [ModelCheckpoint(**checkpoint) for checkpoint in checkpoints]
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=patience_train,
                mode="min",
            )
        )
        callbacks.append(LearningRateMonitor(logging_interval="step"))

        # Strategy
        strategy = "auto"

        train_params = {
            "callbacks": callbacks,
            "max_epochs": num_epochs,
            "num_sanity_val_steps": 2,
            "strategy": strategy,
            "log_every_n_steps": 1,
            "enable_progress_bar": True,
        }

        if trainer_params is not None:
            train_params = train_params | trainer_params

        # Trainer
        self.trainer = pl.Trainer(**train_params)

        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision("medium")

        if verbose:
            logger.info("Launching training...")
            start = time.time()
        self.trainer.fit(self.lightning_module, train_dataloader, val_dataloader)
        if verbose:
            end = time.time()
            logger.info("Training done in {:.2f} seconds.".format(end - start))

        # Load best model
        self.best_model_path = self.trainer.checkpoint_callback.best_model_path
        self.lightning_module = FastTextModule.load_from_checkpoint(
            self.best_model_path,
            model=self.pytorch_model,
            loss=self.loss,
            optimizer=self.optimizer,
            optimizer_params=self.optimizer_params,
            scheduler=self.scheduler,
            scheduler_params=self.scheduler_params,
            scheduler_interval="epoch",
        )
        self.pytorch_model = self.lightning_module.model.to("cpu")
        self.trained = True
        self.pytorch_model.eval()

    def load_from_checkpoint(self, path):
        self.lightning_module = FastTextModule.load_from_checkpoint(
            path,
            model=self.pytorch_model,
            loss=self.loss,
            optimizer=self.optimizer,
            optimizer_params=self.optimizer_params,
            scheduler=self.scheduler,
            scheduler_params=self.scheduler_params,
            scheduler_interval="epoch",
        )
        self.pytorch_model = self.lightning_module.model
        self.tokenizer = self.pytorch_model.tokenizer

        self.sparse = self.pytorch_model.sparse
        self.num_tokens = self.tokenizer.num_tokens
        self.embedding_dim = self.pytorch_model.embedding_dim
        self.num_classes = self.pytorch_model.num_classes
        self.min_n = self.tokenizer.min_n
        self.max_n = self.tokenizer.max_n
        self.len_word_ngrams = self.tokenizer.word_ngrams
        self.no_cat_var = self.pytorch_model.no_cat_var

    def validate(self, X, Y, batch_size=256, num_workers=12):
        """
        Validates the model on the given data.

        Args:
            X (np.ndarray): Array of shape (N,d) with the first column being the text and the rest being the categorical variables.
            Y (np.ndarray): Array of shape (N,) with the labels.

        Returns:
            float: The validation loss.
        """

        if not self.trained:
            raise Exception("Model must be trained first.")

        # checking right format for inputs
        text, categorical_variables, no_cat_var = check_X(X)
        y = check_Y(Y)

        if categorical_variables is not None:
            if categorical_variables.shape[1] != self.num_categorical_features:
                raise Exception(
                    f"X must have the same number of categorical variables as the training data ({self.num_categorical_features})."
                )
        else:
            assert self.pytorch_model.no_cat_var == True

        self.pytorch_model.to(X.device)

        dataset = FastTextModelDataset(
            categorical_variables=categorical_variables,
            texts=text,
            outputs=y,
            tokenizer=self.tokenizer,
        )
        dataloader = dataset.create_dataloader(batch_size=batch_size, num_workers=num_workers)

        return self.trainer.test(self.pytorch_model, test_dataloaders=dataloader, verbose=False)

    def predict(self, X, top_k=1, preprocess=False, verbose=False):
        """
        Predicts the "top_k" classes of the input text.

        Args:
            X (np.ndarray): Array of shape (N,d) with the first column being the text and the rest being the categorical variables.
            top_k (int): Number of classes to predict (by order of confidence).
            preprocess (bool): Whether to preprocess the text before predicting.

        Returns:
            np.ndarray: Array of shape (N,top_k)
        """

        if verbose:
            logger.info(
                "Preprocessing is set to True. Input text will be preprocessed, requiring NLTK and Unidecode librairies."
                if preprocess
                else "Preprocessing is set to False. Input text will not be preprocessed and fed as is to the model."
            )

        if not self.trained:
            raise Exception("Model must be trained first.")

        # checking right format for inputs
        text, categorical_variables, no_cat_var = check_X(X)
        if categorical_variables is not None:
            if categorical_variables.shape[1] != self.num_categorical_features:
                raise Exception(
                    f"X must have the same number of categorical variables as the training data ({self.num_categorical_features})."
                )
        else:
            assert self.pytorch_model.no_cat_var == True

        return self.pytorch_model.predict(
            text, categorical_variables, top_k=top_k, preprocess=preprocess
        )

    def predict_and_explain(self, X, top_k=1):
        if not self.trained:
            raise Exception("Model must be trained first.")

        # checking right format for inputs
        text, categorical_variables, no_cat_var = check_X(X)
        if categorical_variables is not None:
            if categorical_variables.shape[1] != self.num_categorical_features:
                raise Exception(
                    f"X must have the same number of categorical variables as the training data ({self.num_categorical_features})."
                )
        else:
            assert self.pytorch_model.no_cat_var == True

        return self.pytorch_model.predict_and_explain(text, categorical_variables, top_k=top_k)

    def to_json(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            data = asdict(self)

            # Exclude non-serializable fields
            data.pop("tokenizer", None)
            data.pop("pytorch_model", None)
            data.pop("lightning_module", None)

            data.pop("trained", None)  # Useless to save

            json.dump(data, f, cls=NumpyJSONEncoder, indent=4)

    @classmethod
    def from_json(cls: Type["torchFastText"], filepath: str) -> "torchFastText":
        """
        Load a dataclass instance from a JSON file.
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)

    def quantize():
        # TODO
        pass

    def dequantize():
        # TODO
        pass
