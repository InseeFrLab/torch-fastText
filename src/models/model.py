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

from config.preprocess import clean_text_feature
from explainability.utils import match_token_to_word, tokenized_text_in_tokens, \
                                 map_processed_to_original, compute_preprocessed_word_score



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
            inputs (List[torch.LongTensor]): Model inputs.

        Returns:
            torch.Tensor: Model output.
        """
        # Embed tokens

        x_1 = encoded_text # text list, of length batch_size

        if x_1.dtype != torch.long:
            x_1 = x_1.long()


        x_1 = self.embeddings(x_1) # (batch_size, seq_len, embedding_dim)

        x_cat = []
        for i, (variable, embedding_layer) in enumerate(
            self.categorical_embeddings.items()
        ):
            x_cat.append(embedding_layer(additional_inputs[i].long()).squeeze())

        # Aggregating the embeddings of each sequence 
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
    
    def predict(self, text: List, params:dict[str, any] = None, top_k = 1, explain = False):
        """
        Args:
            text (List): A list of text observations.
            params (Optional[Dict[str, Any]]): Additional parameters to
                pass to the model for inference.
            top_k (int): for each sentence, return the top_k most likely predictions (default: 1)
            explain (bool): launch gradient integration to have an explanation of the prediction (default: False)

        Returns:
            A tuple containing the k most likely codes to the query.
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
            all_ind, id_to_token, token_to_id = self.tokenizer.indices_matrix(sentence)
            indices_batch.append(all_ind)
            id_to_token_dicts.append(id_to_token)
            token_to_id_dicts.append(token_to_id)

        max_tokens = max([len(indices) for indices in indices_batch])

        padding_index = self.tokenizer.get_buckets() + self.tokenizer.get_nwords()
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
        

        # Cast
        x = torch.LongTensor(padded_batch.astype(np.int32)).reshape(batch_size, -1)
        other_features = []
        for key in params.keys():
            if key != "text":
                other_features.append(torch.LongTensor(params[key]).reshape(batch_size, -1))
        
        other_features = torch.stack(other_features).reshape(batch_size, -1).long()

        pred = self(x, other_features)
        label_scores = pred.detach().cpu().numpy()

        if explain:
            attributions = lig.attribute((x, other_features), target=pred.argmax(1)).sum(dim=-1)
        
        top_k_indices = np.argsort(label_scores, axis=1)[:, -top_k:]
        confidence = np.take_along_axis(label_scores, top_k_indices, axis=1)
        softmax_scores = softmax(confidence, axis=1).round(2)
        predictions = np.empty((batch_size, top_k)).astype('str')

        for idx in range(batch_size):
            predictions[idx] = self.nace_encoder.inverse_transform(top_k_indices[idx])
        
        if explain:
            return predictions, softmax_scores, attributions, x, id_to_token_dicts, token_to_id_dicts, df.text
        else:
            return predictions, softmax_scores

    def predict_and_explain(self, text, params, n=5, cutoff=0.75):
        pred, confidence, attr, tokenized_text, id_to_token_dicts, token_to_id_dicts, processed_text \
            = self.predict(text=text, params=params, top_k=1, explain=True)

        word_to_score_dicts = compute_preprocessed_word_score(self, processed_text, tokenized_text,
                            attr, id_to_token_dicts, token_to_id_dicts,
                            padding_index=2009603, end_of_string_index=0)

        all_scores = []
        for idx, word_to_score in enumerate(word_to_score_dicts):
            processed_words = list(word_to_score.keys())
            original_words = text[idx].split()        

            for i, word in enumerate(original_words):
                original_words[i] = word.replace(',', '')
                
            mapping = map_processed_to_original(processed_words, original_words, n=n, cutoff=cutoff)

            scores = {}
            for word in original_words:
                processed_words, distances = mapping[word]
                word_score = 0
                for i, potential_processed_word in enumerate(processed_words):
                    score = word_to_score[potential_processed_word]
                    word_score += score * distances[i] / np.sum(distances)

                scores[word] = word_score

            all_scores.append(scores)

        return pred, confidence, all_scores
        


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionLayer, self).__init__()
        # Learnable parameter for attention
        self.attention_weights = nn.Parameter(torch.randn(embedding_dim, 1))

    def forward(self, embeddings):
        # embeddings: (batch_size, seq_len, embedding_dim)
        
        # Compute attention scores
        scores = torch.matmul(embeddings, self.attention_weights)  # (batch_size, seq_len, 1)
        scores = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1), normalized across seq_len
        
        # Weighted sum of embeddings based on attention scores
        weighted_sum = torch.sum(embeddings * scores, dim=1)  # (batch_size, embedding_dim)
        
        return weighted_sum, scores.squeeze(-1)  # Return the weighted sum and the attention scores

class FastTextAttention(nn.Module):
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
        super(FastTextAttention, self).__init__()
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

        self.attention = AttentionLayer(embedding_dim)

        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, encoded_text, additional_inputs) -> torch.Tensor:
        """
        Forward method.

        Args:
            inputs (List[torch.LongTensor]): Model inputs.

        Returns:
            torch.Tensor: Model output.
        """
        # Embed tokens

        x_1 = encoded_text # text list, of length batch_size

        if x_1.dtype != torch.long:
            x_1 = x_1.long()


        x_1 = self.embeddings(x_1) # (batch_size, seq_len, embedding_dim)

        x_cat = []
        for i, (variable, embedding_layer) in enumerate(
            self.categorical_embeddings.items()
        ):
            x_cat.append(embedding_layer(additional_inputs[i].long()).squeeze())

        # Aggregating the embeddings of each sequence 
        # non_zero_tokens = x_1.sum(-1) != 0
        # non_zero_tokens = non_zero_tokens.sum(-1)
        # x_1 = x_1.sum(dim=-2)
        # x_1 /= non_zero_tokens.unsqueeze(-1)
        # x_1 = torch.nan_to_num(x_1) # (batch_size, embedding_dim)

        x_1, attention_scores = self.attention(x_1) # (batch_size, embedding_dim)

        # sum over all the categorical variables, output shape is (batch_size, embedding_dim)
        x_in = x_1 + torch.stack(x_cat, dim=0).sum(dim=0) 
        

        # Linear layer
        z = self.fc(x_in) #(batch_size, num_classes)
        return z
    
    def predict(self, text: List, params:dict[str, any] = None, top_k = 1):
        """
        Args:
            text (List): A list of text observations.
            params (Optional[Dict[str, Any]]): Additional parameters to
                pass to the model for inference.

        Returns:
            A tuple containing the k most likely codes to the query.
        """
        lig = LayerIntegratedGradients(self, self.embeddings)

        self.eval()
        batch_size = len(text)
        params["text"] = text
        
        df = pd.DataFrame(params)
        df = clean_text_feature(df, text_feature="text") # preprocess text

        indices_batch = []
        id_to_token_dicts = []
        token_to_id_dicts = []

        for sentence in df.text:
            all_ind, id_to_token, token_to_id = self.tokenizer.indices_matrix(sentence)
            indices_batch.append(all_ind)
            id_to_token_dicts.append(id_to_token)
            token_to_id_dicts.append(token_to_id)

        max_tokens = max([len(indices) for indices in indices_batch])

        padding_index = self.tokenizer.get_buckets() + self.tokenizer.get_nwords()
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
        

        # Cast
        x = torch.LongTensor(padded_batch.astype(np.int32)).reshape(batch_size, -1)
        other_features = []
        for key in params.keys():
            if key != "text":
                other_features.append(torch.LongTensor(params[key]).reshape(batch_size, -1))
        
        other_features = torch.stack(other_features).reshape(batch_size, -1).long()

        pred = self(x, other_features)
        label_scores = pred.detach().cpu().numpy()

        attributions = lig.attribute((x, other_features), target=pred.argmax(1)).sum(dim=-1)
        
        top_k_indices = np.argsort(label_scores, axis=1)[:, -top_k:]
        confidence = np.take_along_axis(label_scores, top_k_indices, axis=1)
        softmax_scores = softmax(confidence, axis=1).round(2)
        predictions = np.empty((batch_size, top_k)).astype('str')
        for idx in range(batch_size):
            predictions[idx] = self.nace_encoder.inverse_transform(top_k_indices[idx])
        return predictions, softmax_scores, attributions, x, id_to_token_dicts, token_to_id_dicts, df.text

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

