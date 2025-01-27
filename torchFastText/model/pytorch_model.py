"""
FastText model implemented with Pytorch.
Integrates additional categorical features.
"""

from typing import List

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
        categorical_embedding_dims: List[int] = None,
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
        

        if categorical_embedding_dims is not None:
            self.categorical_embedding_dims = categorical_embedding_dims

            if len(set(categorical_embedding_dims)) == 1:
                self.average_cat_embed = True # if categorical embedding dims are the same, we average them before concatenating to the sentence embedding


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

        self.categorical_embedding_layers = {}
        if categorical_vocabulary_sizes is not None:
            self.no_cat_var = False
            for var_idx, vocab_size in enumerate(categorical_vocabulary_sizes):
                if categorical_embedding_dims is not None:
                    emb = nn.Embedding(embedding_dim=categorical_embedding_dims[var_idx], num_embeddings=vocab_size) # concatenate to sentence embedding
                else:
                    emb = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size) # sum to sentence embedding
                self.categorical_embedding_layers[var_idx] = emb
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
            for i, (variable, embedding_layer) in enumerate(self.categorical_embedding_layers.items()):
                x_cat.append(embedding_layer(additional_inputs[:, i].long()).squeeze())

        if len(x_cat) > 0:  # if there are categorical variables

            if self.categorical_embedding_dims is not None: # concatenate to sentence embedding
                if self.average_cat_embed: # unique cat_embedding_dim for all categorical variables
                    x_cat = torch.stack(x_cat, dim=0).mean(dim=0) # average over all the categorical variables, output shape is (batch_size, cat_embedding_dim)
                    x_in = torch.cat([x_1, x_cat], dim=1)  # (batch_size, embedding_dim + cat_embedding_dim)
                else:
                    x_in = torch.cat([x_1] + x_cat, dim=1) # direct concat without averaging, output shape is (batch_size, embedding_dim + sum(cat_embedding_dims))
            
            else: # sum over all the categorical variables, output shape is (batch_size, embedding_dim)
                x_in = x_1 + torch.stack(x_cat, dim=0).sum(dim=0)

        else:
            x_in = x_1

        z = self.fc(x_in)  # (batch_size, num_classes)
        return z

    def predict(
        self, text: List[str], categorical_variables: List[List[int]], top_k=1, explain=False
    ):
        """
        Args:
            text (List[str]): A list of text observations.
            params (Optional[Dict[str, Any]]): Additional parameters to
                pass to the model for inference.
            top_k (int): for each sentence, return the top_k most likely predictions (default: 1)
            explain (bool): launch gradient integration to have an explanation of the prediction (default: False)

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

        indices_batch, id_to_token_dicts, token_to_id_dicts = self.tokenizer.tokenize(
            text, text_tokens=False
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
