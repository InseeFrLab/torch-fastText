"""
FastText model implemented with Pytorch.
Integrates additional categorical features.
"""

from typing import List, Union
import logging

import torch

try:
    from captum.attr import LayerIntegratedGradients

    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

from torch import nn

from ..utilities.utils import (
    compute_preprocessed_word_score,
    compute_word_score,
    explain_continuous,
    tokenized_text_in_tokens,
)
from ..utilities.checkers import validate_categorical_inputs

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


class FastTextModel(nn.Module):
    """
    FastText Pytorch Model.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        tokenizer=None,
        num_rows: int = None,
        categorical_vocabulary_sizes: List[int] = None,
        categorical_embedding_dims: Union[List[int], int] = None,
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

        if isinstance(categorical_embedding_dims, int):
            self.average_cat_embed = True  # if provided categorical embedding dims is an int, average the categorical embeddings before concatenating to sentence embedding
        else:
            self.average_cat_embed = False

        categorical_vocabulary_sizes, categorical_embedding_dims, num_categorical_features = (
            validate_categorical_inputs(
                categorical_vocabulary_sizes,
                categorical_embedding_dims,
                num_categorical_features=None,
            )
        )

        assert isinstance(categorical_embedding_dims, list) or categorical_embedding_dims is None, (
            "categorical_embedding_dims must be a list of int at this stage"
        )

        if categorical_embedding_dims is None:
            self.average_cat_embed = False

        if tokenizer is None:
            if num_rows is None:
                raise ValueError(
                    "Either tokenizer or num_rows must be provided (number of rows in the embedding matrix)."
                )
        else:
            if num_rows is not None:
                if num_rows != tokenizer.num_tokens:
                    logger.warning(
                        "num_rows is different from the number of tokens in the tokenizer. Using provided num_rows."
                    )

        self.num_rows = num_rows

        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.direct_bagging = direct_bagging
        self.sparse = sparse

        self.categorical_embedding_dims = categorical_embedding_dims

        self.embeddings = (
            nn.Embedding(
                embedding_dim=embedding_dim,
                num_embeddings=num_rows,
                padding_idx=padding_idx,
                sparse=sparse,
            )
            if not direct_bagging
            else nn.EmbeddingBag(
                embedding_dim=embedding_dim,
                num_embeddings=num_rows,
                padding_idx=padding_idx,
                sparse=sparse,
                mode="mean",
            )
        )

        self.categorical_embedding_layers = {}

        # Entry dim for the last layer:
        #   1. embedding_dim if no categorical variables or summing the categrical embeddings to sentence embedding
        #   2. embedding_dim + cat_embedding_dim if averaging the categorical embeddings before concatenating to sentence embedding (categorical_embedding_dims is a int)
        #   3. embedding_dim + sum(categorical_embedding_dims) if concatenating individually the categorical embeddings to sentence embedding (no averaging, categorical_embedding_dims is a list)
        dim_in_last_layer = embedding_dim
        if self.average_cat_embed:
            dim_in_last_layer += categorical_embedding_dims[0]

        if categorical_vocabulary_sizes is not None:
            self.no_cat_var = False
            for var_idx, num_rows in enumerate(categorical_vocabulary_sizes):
                if categorical_embedding_dims is not None:
                    emb = nn.Embedding(
                        embedding_dim=categorical_embedding_dims[var_idx], num_embeddings=num_rows
                    )  # concatenate to sentence embedding
                    if not self.average_cat_embed:
                        dim_in_last_layer += categorical_embedding_dims[var_idx]
                else:
                    emb = nn.Embedding(
                        embedding_dim=embedding_dim, num_embeddings=num_rows
                    )  # sum to sentence embedding
                self.categorical_embedding_layers[var_idx] = emb
                setattr(self, "emb_{}".format(var_idx), emb)
        else:
            self.no_cat_var = True

        self.fc = nn.Linear(dim_in_last_layer, num_classes)

    def forward(self, encoded_text: torch.Tensor, additional_inputs: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient forward pass implementation.

        Args:
            encoded_text (torch.Tensor[Long]), shape (batch_size, seq_len): Tokenized + padded text
            additional_inputs (torch.Tensor[Long]): Additional categorical features, (batch_size, num_categorical_features)

        Returns:
            torch.Tensor: Model output scores for each class
        """
        batch_size = encoded_text.size(0)

        # Ensure correct dtype and device once
        if encoded_text.dtype != torch.long:
            encoded_text = encoded_text.to(torch.long)

        # Compute text embeddings
        if self.direct_bagging:
            x_text = self.embeddings(encoded_text)  # (batch_size, embedding_dim)
        else:
            # Compute embeddings and averaging in a memory-efficient way
            x_text = self.embeddings(encoded_text)  # (batch_size, seq_len, embedding_dim)
            # Calculate non-zero tokens mask once
            non_zero_mask = (x_text.sum(-1) != 0).float()  # (batch_size, seq_len)
            token_counts = non_zero_mask.sum(-1, keepdim=True)  # (batch_size, 1)

            # Sum and average in place
            x_text = (x_text * non_zero_mask.unsqueeze(-1)).sum(
                dim=1
            )  # (batch_size, embedding_dim)
            x_text = torch.div(x_text, token_counts.clamp(min=1.0))
            x_text = torch.nan_to_num(x_text, 0.0)

        # Handle categorical variables efficiently
        if not self.no_cat_var and additional_inputs.numel() > 0:
            cat_embeds = []
            # Process categorical embeddings in batch
            for i, (_, embed_layer) in enumerate(self.categorical_embedding_layers.items()):
                cat_input = additional_inputs[:, i].long()
                cat_embed = embed_layer(cat_input)
                if cat_embed.dim() > 2:
                    cat_embed = cat_embed.squeeze(1)
                cat_embeds.append(cat_embed)

            if cat_embeds:  # If we have categorical embeddings
                if self.categorical_embedding_dims is not None:
                    if self.average_cat_embed:
                        # Stack and average in one operation
                        x_cat = torch.stack(cat_embeds, dim=0).mean(dim=0)
                        x_combined = torch.cat([x_text, x_cat], dim=1)
                    else:
                        # Optimize concatenation
                        x_combined = torch.cat([x_text] + cat_embeds, dim=1)
                else:
                    # Sum embeddings efficiently
                    x_combined = x_text + torch.stack(cat_embeds, dim=0).sum(dim=0)
            else:
                x_combined = x_text
        else:
            x_combined = x_text

        # Final linear layer
        return self.fc(x_combined)

    def predict(
        self,
        text: List[str],
        categorical_variables: List[List[int]],
        top_k=1,
        explain=False,
        preprocess=True,
    ):
        """
        Args:
            text (List[str]): A list of text observations.
            params (Optional[Dict[str, Any]]): Additional parameters to
                pass to the model for inference.
            top_k (int): for each sentence, return the top_k most likely predictions (default: 1)
            explain (bool): launch gradient integration to have an explanation of the prediction (default: False)
            preprocess (bool): If True, preprocess text. Needs unidecode library.

        Returns:
        if explain is False:
            predictions (torch.Tensor, shape (len(text), top_k)): A tensor containing the top_k most likely codes to the query.
            confidence (torch.Tensor, shape (len(text), top_k)): A tensor array containing the corresponding confidence scores.
        if explain is True:
            predictions (torch.Tensor, shape (len(text), top_k)): Containing the top_k most likely codes to the query.
            confidence (torch.Tensor, shape (len(text), top_k)): Corresponding confidence scores.
            all_attributions (torch.Tensor, shape (len(text), top_k, seq_len)): A tensor containing the attributions for each token in the text.
            x (torch.Tensor): A tensor containing the token indices of the text.
            id_to_token_dicts (List[Dict[int, str]]): A list of dictionaries mapping token indices to tokens (one for each sentence).
            token_to_id_dicts (List[Dict[str, int]]): A list of dictionaries mapping tokens to token indices: the reverse of those in id_to_token_dicts.
            text (list[str]): A plist containing the preprocessed text (one line for each sentence).
        """

        flag_change_embed = False
        if explain:
            if not HAS_CAPTUM:
                raise ImportError(
                    "Captum is not installed and is required for explainability. Run 'pip install torchFastText[explainability]'."
                )
            if self.direct_bagging:
                # Get back the classical embedding layer for explainability
                new_embed_layer = nn.Embedding(
                    embedding_dim=self.embedding_dim,
                    num_embeddings=self.num_rows,
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

        indices_batch, id_to_token_dicts, token_to_id_dicts = self.tokenizer.tokenize(
            text, text_tokens=False, preprocess=preprocess
        )

        padding_index = (
            self.tokenizer.get_buckets() + self.tokenizer.get_nwords()
        )  # padding index, the integer value of the padding token

        padded_batch = torch.nn.utils.rnn.pad_sequence(
            indices_batch,
            batch_first=True,
            padding_value=padding_index,
        )  # (batch_size, seq_len) - Tokenized (int) + padded text

        x = padded_batch

        if not self.no_cat_var:
            other_features = []
            for i, categorical_variable in enumerate(categorical_variables):
                other_features.append(
                    torch.tensor(categorical_variable).reshape(batch_size, -1).to(torch.int64)
                )

            other_features = torch.stack(other_features).reshape(batch_size, -1).long()
        else:
            other_features = torch.empty(batch_size)

        pred = self(
            x, other_features
        )  # forward pass, contains the prediction scores (len(text), num_classes)
        label_scores = pred.detach().cpu()
        label_scores_topk = torch.topk(label_scores, k=top_k, dim=1)

        predictions = label_scores_topk.indices  # get the top_k most likely predictions
        confidence = torch.round(label_scores_topk.values, decimals=2)  # and their scores

        if explain:
            assert not self.direct_bagging, "Direct bagging should be False for explainability"
            all_attributions = []
            for k in range(top_k):
                attributions = lig.attribute(
                    (x, other_features), target=torch.Tensor(predictions[:, k]).long()
                )  # (batch_size, seq_len)
                attributions = attributions.sum(dim=-1)
                all_attributions.append(attributions.detach().cpu())

            all_attributions = torch.stack(all_attributions, dim=1)  # (batch_size, top_k, seq_len)

            # Get back to initial embedding layer:
            # EmbeddingBag -> Embedding -> EmbeddingBag
            # or keep Embedding with no change
            if flag_change_embed:
                new_embed_layer = nn.EmbeddingBag(
                    embedding_dim=self.embedding_dim,
                    num_embeddings=self.num_rows,
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
            )
        else:
            return predictions, confidence

    def predict_and_explain(self, text, categorical_variables, top_k=1, n=5, cutoff=0.65):
        """
        Args:
            text (List[str]): A list of sentences.
            params (Optional[Dict[str, Any]]): Additional parameters to
                pass to the model for inference.
            top_k (int): for each sentence, return the top_k most likely predictions (default: 1)
            n (int): mapping processed to original words: max number of candidate processed words to consider per original word (default: 5)
            cutoff (float): mapping processed to original words: minimum similarity score to consider a candidate processed word (default: 0.75)

        Returns:
            predictions (torch.Tensor, shape (len(text), top_k)): Containing the top_k most likely codes to the query.
            confidence (torch.Tensor, shape (len(text), top_k)): Corresponding confidence scores.
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
        ) = self.predict(
            text=text, categorical_variables=categorical_variables, top_k=top_k, explain=True
        )

        tokenized_text_tokens = tokenized_text_in_tokens(tokenized_text, id_to_token_dicts)

        # Step 2: Map the attributions at token level to the processed words
        processed_word_to_score_dicts, processed_word_to_token_idx_dicts = (
            compute_preprocessed_word_score(
                processed_text,
                tokenized_text_tokens,
                all_attr,
                id_to_token_dicts,
                token_to_id_dicts,
                min_n=self.tokenizer.min_n,
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
