"""
FastText model implemented with Pytorch.
Integrates additional categorical features.
"""
from typing import List
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchmetrics import Accuracy
from torch import nn
import pytorch_lightning as pl
from scipy.special import softmax
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from scipy.special import softmax

from config.preprocess import clean_text_feature
from explainability.utils import tokenized_text_in_tokens, \
                                 map_processed_to_original, compute_preprocessed_word_score, \
                                 compute_word_score, preprocess_token, explain_continuous



class FastTextModel(nn.Module):
    """
    FastText Pytorch Model.
    """

    def __init__(
        self,
        tokenizer,
        nace_encoder,
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
        self.tokenizer = tokenizer
        self.nace_encoder = nace_encoder
        self.embedding_dim = embedding_dim
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

    def forward(self, encoded_text, additional_inputs) -> torch.Tensor:
        """
        Forward method.

        Args:
            encoded_text (torch.Tensor[Long]), shape (batch_size, seq_len): Tokenized + padded text (in integer (indices))
            additional_inputs (torch.Tensor[Long]): Additional categorical features.

        Returns:
            torch.Tensor: Model output: score for each class.
        """


        x_1 = encoded_text

        if x_1.dtype != torch.long:
            x_1 = x_1.long()

        # Embed tokens
        x_1 = self.embeddings(x_1) # (batch_size, seq_len, embedding_dim)

        # Embed categorical variables
        x_cat = []
        for i, (variable, embedding_layer) in enumerate(
            self.categorical_embeddings.items()
        ):
            x_cat.append(embedding_layer(additional_inputs[i].long()).squeeze())

        # Aggregating (via averaging) the embeddings of each sequence 
        non_zero_tokens = x_1.sum(-1) != 0
        non_zero_tokens = non_zero_tokens.sum(-1)
        x_1 = x_1.sum(dim=-2)
        x_1 /= non_zero_tokens.unsqueeze(-1)
        x_1 = torch.nan_to_num(x_1) # (batch_size, embedding_dim)
        
        # sum over all the categorical variables, output shape is (batch_size, embedding_dim)
        x_in = x_1 + torch.stack(x_cat, dim=0).sum(dim=0) 
        

        # Linear layer
        z = self.fc(x_in) #(batch_size, num_classes)
        return z
    
    def predict(self, text: List[str], params:dict[str, any] = None, top_k = 1, explain = False):
        """
        Args:
            text (List[str]): A list of text observations.
            params (Optional[Dict[str, Any]]): Additional parameters to
                pass to the model for inference.
            top_k (int): for each sentence, return the top_k most likely predictions (default: 1)
            explain (bool): launch gradient integration to have an explanation of the prediction (default: False)

        Returns:
        if explain is False:
            predictions (np.ndarray): A numpy array containing the top_k most likely codes to the query.
            confidence (np.ndarray): A numpy array containing the corresponding confidence scores.
        if explain is True:
            predictions (np.ndarray, shape (len(text), top_k)): Containing the top_k most likely codes to the query.
            confidence (np.ndarray, shape (len(text), top_k)): Corresponding confidence scores.
            all_attributions (torch.Tensor, shape (len(text), top_k, seq_len)): A tensor containing the attributions for each token in the text.
            x (torch.Tensor): A tensor containing the token indices of the text.
            id_to_token_dicts (List[Dict[int, str]]): A list of dictionaries mapping token indices to tokens (one for each sentence).
            token_to_id_dicts (List[Dict[str, int]]): A list of dictionaries mapping tokens to token indices: the reverse of those in id_to_token_dicts.
            df.text (pd.Series): A pandas Series containing the preprocessed text (one line for each sentence).
        """

        if explain:
            lig = LayerIntegratedGradients(self, self.embeddings) # initialize a Captum layer gradient integrator

        self.eval()
        batch_size = len(text)
        params["text"] = text
        
        df = pd.DataFrame(params)
        df = clean_text_feature(df, text_feature="text") # preprocess text

        indices_batch = []
        id_to_token_dicts = []
        token_to_id_dicts = []

        for sentence in df.text:
            all_ind, id_to_token, token_to_id = self.tokenizer.indices_matrix(sentence) # tokenize and convert to token indices
            indices_batch.append(all_ind)
            id_to_token_dicts.append(id_to_token)
            token_to_id_dicts.append(token_to_id)

        max_tokens = max([len(indices) for indices in indices_batch])

        padding_index = self.tokenizer.get_buckets() + self.tokenizer.get_nwords() # padding index, the integer value of the padding token
        padded_batch = [
            np.pad(
                indices,
                (0, max_tokens - len(indices)),
                "constant",
                constant_values=padding_index,
            )
            for indices in indices_batch
        ]
        padded_batch = np.stack(padded_batch)
        
        x = torch.LongTensor(padded_batch.astype(np.int32)).reshape(batch_size, -1) # (batch_size, seq_len) - Tokenized (int) + padded text
        other_features = []
        for key in params.keys():
            if key != "text":
                other_features.append(torch.LongTensor(params[key]).reshape(batch_size, -1))
        
        other_features = torch.stack(other_features).reshape(batch_size, -1).long()

        pred = self(x, other_features) # forward pass, contains the prediction scores (len(text), num_classes)
        label_scores = pred.detach().cpu().numpy()
        top_k_indices = np.argsort(label_scores, axis=1)[:, -top_k:] 

        if explain:
            all_attributions = []
            for k in range(top_k):
                attributions = lig.attribute((x, other_features), target=torch.Tensor(top_k_indices[:, k]).long()).sum(dim=-1) # (batch_size, seq_len)
                all_attributions.append(attributions.detach().cpu())
            all_attributions = torch.stack(all_attributions, dim = 1) # (batch_size, top_k, seq_len)

        confidence = np.take_along_axis(label_scores, top_k_indices, axis=1).round(2) # get the top_k most likely predictions
        predictions = np.empty((batch_size, top_k)).astype('str')

        for idx in range(batch_size):
            predictions[idx] = self.nace_encoder.inverse_transform(top_k_indices[idx]) # convert the indices to the corresponding NACE codes (str)
        
        if explain:
            
            return predictions, confidence, all_attributions, x, id_to_token_dicts, \
                   token_to_id_dicts, df.text, label_scores
        else:
            return predictions, confidence, label_scores

    def predict_and_explain(self, text, params, top_k=1, n=5, cutoff=0.65):
        """
        Args:
            text (List[str]): A list of sentences.
            params (Optional[Dict[str, Any]]): Additional parameters to
                pass to the model for inference.
            top_k (int): for each sentence, return the top_k most likely predictions (default: 1)
            n (int): mapping processed to original words: max number of candidate processed words to consider per original word (default: 5)
            cutoff (float): mapping processed to original words: minimum similarity score to consider a candidate processed word (default: 0.75)
        
        Returns:
            predictions (np.ndarray): A numpy array containing the top_k most likely codes to the query.
            confidence (np.ndarray): A numpy array containing the corresponding confidence scores.
            all_scores (List[List[List[float]]]): For each sentence, list of the top_k lists of attributions for each word in the sentence (one for each pred).
        """

        # Step 1: Get the predictions, confidence scores and attributions at token level
        pred, confidence, all_attr, tokenized_text, id_to_token_dicts, token_to_id_dicts, \
            processed_text, _ = self.predict(text=text, params=params, top_k=top_k, explain=True)

        tokenized_text_tokens = tokenized_text_in_tokens(tokenized_text, id_to_token_dicts)
        # Step 2: Map the attributions at token level to the processed words
        processed_word_to_score_dicts, processed_word_to_token_idx_dicts = \
            compute_preprocessed_word_score(
                            processed_text, tokenized_text_tokens,
                            all_attr, id_to_token_dicts, token_to_id_dicts,
                            padding_index=2009603, end_of_string_index=0
                            )
        # Step 3: Map the processed words to the original words
        all_scores, orig_to_processed_mappings = compute_word_score(processed_word_to_score_dicts, text, n=n, cutoff=cutoff)

        all_scores_letters = explain_continuous(
            text, processed_text, tokenized_text_tokens, orig_to_processed_mappings,
            processed_word_to_token_idx_dicts, all_attr, top_k
                                                     )

        return pred, confidence, all_scores, all_scores_letters


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
        return self.model(inputs[0], inputs[1])

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

