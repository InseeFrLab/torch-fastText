"""
FastText model implemented with Pytorch.
Integrates additional categorical features.
"""

from typing import List

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from torch import nn

from config.preprocess import clean_text_feature
from explainability.utils import (
    compute_preprocessed_word_score,
    compute_word_score,
    explain_continuous,
    tokenized_text_in_tokens,
)


class FastTextModel(nn.Module):
    """
    FastText Pytorch Model.
    """

    def __init__(
        self,
        tokenizer,
        embedding_dim: int,
        vocab_size: int,
        num_classes: int,
        categorical_vocabulary_sizes: List[int] = None,
        padding_idx: int = 0,
        sparse: bool = True,
        direct_bagging: bool = False,
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
            direct_bagging (bool): Use EmbeddingBag instead of Embedding for the text embedding.
        """
        super(FastTextModel, self).__init__()
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.direct_bagging = direct_bagging
        self.vocab_size = vocab_size
        self.sparse = sparse

        self.embeddings = (
            nn.Embedding(
                embedding_dim=embedding_dim,
                num_embeddings=vocab_size,
                padding_idx=padding_idx,
                sparse=sparse,
            )
            if not direct_bagging
            else nn.EmbeddingBag(
                embedding_dim=embedding_dim, num_embeddings=vocab_size, sparse=sparse, mode="mean"
            )
        )

        self.categorical_embeddings = {}
        if categorical_vocabulary_sizes is not None:
            self.no_cat_var = False
            for var_idx, vocab_size in enumerate(categorical_vocabulary_sizes):
                emb = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size)
                self.categorical_embeddings[var_idx] = emb
                setattr(self, "emb_{}".format(var_idx), emb)
        else:
            self.no_cat_var = True

        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, encoded_text, additional_inputs) -> torch.Tensor:
        """
        Forward method.

        Args:
            encoded_text (torch.Tensor[Long]), shape (batch_size, seq_len): Tokenized + padded text (in integer (indices))
            additional_inputs (torch.Tensor[Long]): Additional categorical features, (batch_size , num_categorical_features)

        Returns:
            torch.Tensor: Model output: score for each class.
        """
        x_1 = encoded_text

        if x_1.dtype != torch.long:
            x_1 = x_1.long()

        # Embed tokens + averaging = sentence embedding if direct_bagging
        # No averaging if direct_bagging (handled directly by EmbeddingBag)
        x_1 = self.embeddings(
            x_1
        )  # (batch_size, embedding_dim) if direct_bagging otherwise (batch_size, seq_len, embedding_dim)

        if not self.direct_bagging:
            # Aggregate the embeddings of the text tokens
            non_zero_tokens = x_1.sum(-1) != 0
            non_zero_tokens = non_zero_tokens.sum(-1)
            x_1 = x_1.sum(dim=-2)  # (batch_size, embedding_dim)
            x_1 /= non_zero_tokens.unsqueeze(-1)
            x_1 = torch.nan_to_num(x_1)

        # Embed categorical variables
        x_cat = []
        if not self.no_cat_var:
            for i, (variable, embedding_layer) in enumerate(self.categorical_embeddings.items()):
                x_cat.append(embedding_layer(additional_inputs[:, i].long()).squeeze())

        if len(x_cat) > 0:  # if there are categorical variables
            # sum over all the categorical variables, output shape is (batch_size, embedding_dim)
            x_in = x_1 + torch.stack(x_cat, dim=0).sum(dim=0)

        else:
            x_in = x_1

        # Linear layer
        z = self.fc(x_in)  # (batch_size, num_classes)
        return z

    def predict(self, text: List[str], params: dict[str, any] = None, top_k=1, explain=False):
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
            text (list[str]): A plist containing the preprocessed text (one line for each sentence).
        """

        flag_change_embed = False
        if explain:
            if self.direct_bagging:
                # Get back the classical embedding layer for explainability
                new_embed_layer = nn.Embedding(
                    embedding_dim=self.embedding_dim,
                    num_embeddings=self.vocab_size,
                    padding_idx=self.padding_idx,
                    sparse=self.sparse,
                )
                new_embed_layer.load_state_dict(
                    self.embeddings.state_dict()
                )  # No issues, as exactly the same parameters
                self.embeddings = new_embed_layer
                self.direct_bagging = (
                    False  # To inform the forward pass that we are not using EmbeddingBag anymore
                )
                flag_change_embed = True

            lig = LayerIntegratedGradients(
                self, self.embeddings
            )  # initialize a Captum layer gradient integrator

        self.eval()
        batch_size = len(text)
        params["text"] = text

        text = clean_text_feature(text)  # preprocess text

        indices_batch = []
        id_to_token_dicts = []
        token_to_id_dicts = []

        for sentence in text:
            all_ind, id_to_token, token_to_id = self.tokenizer.indices_matrix(
                sentence
            )  # tokenize and convert to token indices
            indices_batch.append(all_ind)
            id_to_token_dicts.append(id_to_token)
            token_to_id_dicts.append(token_to_id)

        max_tokens = max([len(indices) for indices in indices_batch])

        padding_index = (
            self.tokenizer.get_buckets() + self.tokenizer.get_nwords()
        )  # padding index, the integer value of the padding token
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

        x = torch.LongTensor(padded_batch.astype(np.int32)).reshape(
            batch_size, -1
        )  # (batch_size, seq_len) - Tokenized (int) + padded text

        other_features = []
        for key in params.keys():
            if key != "text":
                other_features.append(
                    torch.tensor(params[key]).reshape(batch_size, -1).to(torch.int64)
                )

        other_features = torch.stack(other_features).reshape(batch_size, -1).long()

        pred = self(
            x, other_features
        )  # forward pass, contains the prediction scores (len(text), num_classes)
        label_scores = pred.detach().cpu().numpy()
        top_k_indices = np.argsort(label_scores, axis=1)[:, -top_k:]
        if explain:
            assert not self.direct_bagging, "Direct bagging should be False for explainability"
            all_attributions = []
            for k in range(top_k):
                attributions = lig.attribute(
                    (x, other_features), target=torch.Tensor(top_k_indices[:, k]).long()
                )  # (batch_size, seq_len)
                attributions = attributions.sum(dim=-1)
                all_attributions.append(attributions.detach().cpu())
            all_attributions = torch.stack(all_attributions, dim=1)  # (batch_size, top_k, seq_len)

        confidence = np.take_along_axis(label_scores, top_k_indices, axis=1).round(
            2
        )  # get the top_k most likely predictions
        predictions = np.empty((batch_size, top_k)).astype("str")

        if explain:
            assert not self.direct_bagging, "Direct bagging should be False for explainability"
            # Get back to initial embedding layer:
            # EmbeddingBag -> Embedding -> EmbeddingBag
            # or keep Embedding with no change
            if flag_change_embed:
                new_embed_layer = nn.EmbeddingBag(
                    embedding_dim=self.embedding_dim,
                    num_embeddings=self.vocab_size,
                    padding_idx=self.padding_idx,
                    sparse=self.sparse,
                )
                new_embed_layer.load_state_dict(
                    self.embeddings.state_dict()
                )  # No issues, as exactly the same parameters
                self.embeddings = new_embed_layer
                self.direct_bagging = True
            return (
                predictions,
                confidence,
                all_attributions,
                x,
                id_to_token_dicts,
                token_to_id_dicts,
                text,
                label_scores,
            )
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
        (
            pred,
            confidence,
            all_attr,
            tokenized_text,
            id_to_token_dicts,
            token_to_id_dicts,
            processed_text,
            _,
        ) = self.predict(text=text, params=params, top_k=top_k, explain=True)

        assert not self.direct_bagging, "Direct bagging should be False for explainability"

        tokenized_text_tokens = tokenized_text_in_tokens(tokenized_text, id_to_token_dicts)

        # Step 2: Map the attributions at token level to the processed words
        processed_word_to_score_dicts, processed_word_to_token_idx_dicts = (
            compute_preprocessed_word_score(
                processed_text,
                tokenized_text_tokens,
                all_attr,
                id_to_token_dicts,
                token_to_id_dicts,
                padding_index=2009603,
                end_of_string_index=0,
            )
        )

        # Step 3: Map the processed words to the original words
        all_scores, orig_to_processed_mappings = compute_word_score(
            processed_word_to_score_dicts, text, n=n, cutoff=cutoff
        )

        # Step 2bis: Get the attributions at letter level
        all_scores_letters = explain_continuous(
            text,
            processed_text,
            tokenized_text_tokens,
            orig_to_processed_mappings,
            processed_word_to_token_idx_dicts,
            all_attr,
            top_k,
        )

        return pred, confidence, all_scores, all_scores_letters
