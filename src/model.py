"""
FastText model implemented with Pytorch.
Integrates additional categorical features.
"""
from typing import List
import torch
import pandas as pd
import numpy as np
from torchmetrics import Accuracy
from torch import nn
import pytorch_lightning as pl
from scipy.special import softmax
from captum.attr import IntegratedGradients, LayerIntegratedGradients

from preprocess import clean_text_feature



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
        #sum over all the categorical variables, output shape is (batch_size, embedding_dim)
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
        df = clean_text_feature(df, text_feature="text") #preprocess text

        indices_batch = []
        id_to_token_dicts = []

        for sentence in text:
            all_ind, id_to_token = self.tokenizer.indices_matrix(sentence)
            indices_batch.append(all_ind)
            id_to_token_dicts.append(id_to_token)

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

        attributions = lig.attribute((x, other_features), target=pred.argmax(1))
        
        top_k_indices = np.argsort(label_scores, axis = 1)[:, -top_k:]
        confidence = np.take_along_axis(label_scores, top_k_indices, axis=1)
        softmax_scores = softmax(confidence, axis  = 1).round(2)
        predictions = np.empty((batch_size, top_k)).astype('str')
        for idx in range(batch_size):
            predictions[idx] = self.nace_encoder.inverse_transform(top_k_indices[idx])
        return predictions, softmax_scores, attributions, x, id_to_token_dicts