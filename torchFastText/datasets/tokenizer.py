"""
NGramTokenizer class.
"""

import ctypes
import json
from typing import List, Tuple, Type, Dict

import numpy as np
import torch
from torch import Tensor
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
import multiprocessing

from ..preprocess import clean_text_feature


class NGramTokenizer:
    """
    NGramTokenizer class.
    """

    def __init__(
        self,
        min_count: int,
        min_n: int,
        max_n: int,
        num_tokens: int,
        len_word_ngrams: int,
        training_text: List[str],
        **kwargs,
    ):
        """
        Constructor for the NGramTokenizer class.

        Args:
            min_count (int): Minimum number of times a word has to be
                in the training data to be given an embedding.
            min_n (int): Minimum length of character n-grams.
            max_n (int): Maximum length of character n-grams.
            num_tokens (int): Number of rows in the embedding matrix.
            word_ngrams (int): Maximum length of word n-grams.
            training_text (List[str]): List of training texts.

        Raises:
            ValueError: If `min_n` is 1 or smaller.
            ValueError: If `max_n` is 7 or higher.
        """
        if min_n < 2:
            raise ValueError("`min_n` parameter must be greater than 1.")
        if max_n > 6:
            raise ValueError("`max_n` parameter must be smaller than 7.")

        self.min_count = min_count
        self.min_n = min_n
        self.max_n = max_n
        self.num_tokens = num_tokens
        self.word_ngrams = len_word_ngrams

        word_counts = {}
        for sentence in training_text:
            for word in sentence.split(" "):
                word_counts[word] = word_counts.setdefault(word, 0) + 1

        self.word_id_mapping = {}
        i = 1
        for word, counts in word_counts.items():
            if word_counts[word] >= min_count:
                self.word_id_mapping[word] = i
                i += 1
        self.nwords = len(self.word_id_mapping)

        self.padding_index = self.num_tokens + self.get_nwords()

    def __str__(self) -> str:
        """
        Returns description of the NGramTokenizer.

        Returns:
            str: Description.
        """
        return f"<NGramTokenizer(min_n={self.min_n}, max_n={self.max_n}, num_tokens={self.num_tokens}, word_ngrams={self.word_ngrams}, nwords={self.nwords})>"

    def get_nwords(self) -> int:
        """
        Return number of words kept in training data.

        Returns:
            int: Number of words.
        """
        return self.nwords

    def get_buckets(self) -> int:
        """
        Return number of buckets for tokenizer.

        Returns:
            int: Number of buckets.
        """
        return self.num_tokens

    @staticmethod
    def get_ngram_list(word: str, n: int) -> List[str]:
        """
        Return the list of character n-grams for a word with a
        given n.

        Args:
            word (str): Word.
            n (int): Length of the n-grams.

        Returns:
            List[str]: List of character n-grams.
        """
        return [word[i : i + n] for i in range(len(word) - n + 1)]

    @staticmethod
    def get_hash(subword: str) -> int:
        """
        Return hash for a given subword.

        Args:
            subword (str): Character n-gram.

        Returns:
            int: Corresponding hash.
        """
        h = ctypes.c_uint32(2166136261).value
        for c in subword:
            c = ctypes.c_int8(ord(c)).value
            h = ctypes.c_uint32(h ^ c).value
            h = ctypes.c_uint32(h * 16777619).value
        return h

    @staticmethod
    def get_word_ngram_id(hashes: Tuple[int], bucket: int, nwords: int) -> int:
        """
        Get word ngram hash.

        Args:
            hashes (Tuple[int]): Word hashes.
            bucket (int): Number of rows in embedding matrix.
            nwords (int): Number of words in the vocabulary.

        Returns:
            int: Word ngram hash.
        """
        hashes = [ctypes.c_int32(hash_value).value for hash_value in hashes]
        h = ctypes.c_uint64(hashes[0]).value
        for j in range(1, len(hashes)):
            h = ctypes.c_uint64((h * 116049371)).value
            h = ctypes.c_uint64(h + hashes[j]).value
        return h % bucket + nwords

    def get_subword_index(self, subword: str) -> int:
        """
        Return the row index from the embedding matrix which
        corresponds to a character n-gram.

        Args:
            subword (str): Character n-gram.

        Returns:
            int: Index.
        """
        return self.get_hash(subword) % self.num_tokens + self.nwords

    def get_word_index(self, word: str) -> int:
        """
        Return the row index from the embedding matrix which
        corresponds to a word.

        Args:
            word (str): Word.

        Returns:
            int: Index.
        """
        return self.word_id_mapping[word]

    def get_subwords(self, word: str) -> Tuple[List[str], List[int]]:
        """
        Return all subword tokens and indices for a given word.

        Args:
            word (str): Word.

        Returns:
            Tuple[List[str], List[int]]: Tuple of tokens and indices.
        """
        tokens = []
        word_with_tags = "<" + word + ">"
        for n in range(self.min_n, self.max_n + 1):
            tokens += self.get_ngram_list(word_with_tags, n)
        indices = [self.get_subword_index(token) for token in tokens]

        # Add word
        try:
            if word not in tokens:
                indices = [self.get_word_index(word)] + indices
                tokens = [word] + tokens

        except KeyError:
            # print("Token was not in mapping, not adding it to subwords.")
            pass
        return (tokens, indices)

    def indices_matrix(self, sentence: str) -> tuple[torch.Tensor, dict, dict]:
        """
        Returns an array of token indices for a text description.

        Args:
            sentence (str): Text description.

        Returns:
            tuple: (torch.Tensor of indices, id_to_token dict, token_to_id dict)
        """
        # Pre-split the sentence once
        words = sentence.split()
        words.append("</s>")  # Add end of string token

        indices = []
        all_tokens_id = {}

        # Process subwords in one batch
        for word in words[:-1]:  # Exclude </s> from subword processing
            tokens, ind = self.get_subwords(word)
            indices.extend(ind)
            # Update dictionary with zip for efficiency
            all_tokens_id.update(zip(tokens, ind))

        # Add </s> token
        indices.append(0)
        all_tokens_id["</s>"] = 0

        # Compute word n-grams more efficiently
        if self.word_ngrams > 1:
            # Pre-compute hashes for all words to avoid repeated computation
            word_hashes = [self.get_hash(word) for word in words]

            # Generate n-grams using sliding window
            word_ngram_ids = []
            for n in range(2, self.word_ngrams + 1):
                for i in range(len(words) - n + 1):
                    # Get slice of hashes for current n-gram
                    gram_hashes = tuple(word_hashes[i : i + n])

                    # Compute n-gram ID
                    word_ngram_id = int(
                        self.get_word_ngram_id(gram_hashes, self.num_tokens, self.nwords)
                    )

                    # Store gram and its ID
                    gram = " ".join(words[i : i + n])
                    all_tokens_id[gram] = word_ngram_id
                    word_ngram_ids.append(word_ngram_id)

            # Extend indices with n-gram IDs
            indices.extend(word_ngram_ids)

        # Create reverse mapping once at the end
        id_to_token = {v: k for k, v in all_tokens_id.items()}

        # Convert to tensor directly
        return torch.tensor(indices, dtype=torch.long), id_to_token, all_tokens_id

    def tokenize(self, text: list[str], text_tokens=True, preprocess=False):
        """
        Tokenize a list of sentences.

        Args:
            text (list[str]): List of sentences.
            text_tokens (bool): If True, return tokenized text in tokens.
            preprocess (bool): If True, preprocess text. Needs unidecode library.

        Returns:
            np.array: Array of indices.
        """

        if preprocess:
            text = clean_text_feature(text)

        tokenized_text = []
        id_to_token_dicts = []
        token_to_id_dicts = []
        for sentence in text:
            all_ind, id_to_token, token_to_id = self.indices_matrix(
                sentence
            )  # tokenize and convert to token indices
            tokenized_text.append(all_ind)
            id_to_token_dicts.append(id_to_token)
            token_to_id_dicts.append(token_to_id)

        if text_tokens:
            tokenized_text_tokens = self._tokenized_text_in_tokens(
                tokenized_text, id_to_token_dicts
            )
            return tokenized_text_tokens, tokenized_text, id_to_token_dicts, token_to_id_dicts
        else:
            return tokenized_text, id_to_token_dicts, token_to_id_dicts

    def _tokenized_text_in_tokens(self, tokenized_text, id_to_token_dicts):
        """
        Convert tokenized text in int format to tokens in str format (given a mapping dictionary).
        Private method. Used in tokenizer.tokenize and pytorch_model.predict()

        Args:
            tokenized_text (list): List of tokenized text in int format.
            id_to_token_dicts (list[Dict]): List of dictionaries mapping token indices to tokens.

            Both lists have the same length (number of sentences).

        Returns:
            list[list[str]]: List of tokenized text in str format.

        """

        return [
            [
                id_to_token_dicts[i][token_id.item()]
                for token_id in tokenized_sentence
                if token_id.item() not in {self.padding_index}
            ]
            for i, tokenized_sentence in enumerate(tokenized_text)
        ]

    def get_vocab(self):
        return self.word_id_mapping

    @classmethod
    def from_json(cls: Type["NGramTokenizer"], filepath: str, training_text) -> "NGramTokenizer":
        """
        Load a dataclass instance from a JSON file.
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data, training_text=training_text)
