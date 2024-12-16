"""
NGramTokenizer class.
"""

import ctypes
from typing import List, Tuple

import numpy as np
import torch

from ..preprocess import clean_text_feature
from ..utilities.utils import tokenized_text_in_tokens


class NGramTokenizer:
    """
    NGramTokenizer class.
    """

    def __init__(
        self,
        min_count: int,
        min_n: int,
        max_n: int,
        buckets: int,
        word_ngrams: int,
        training_text: List[str],
    ):
        """
        Constructor for the NGramTokenizer class.

        Args:
            min_count (int): Minimum number of times a word has to be
                in the training data to be given an embedding.
            min_n (int): Minimum length of character n-grams.
            max_n (int): Maximum length of character n-grams.
            buckets (int): Number of rows in the embedding matrix.
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
        self.min_n = min_n
        self.max_n = max_n
        self.num_buckets = buckets
        self.word_ngrams = word_ngrams

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

    def __str__(self) -> str:
        """
        Returns description of the NGramTokenizer.

        Returns:
            str: Description.
        """
        return f"<NGramTokenizer(min_n={self.min_n}, max_n={self.max_n}, num_buckets={self.num_buckets}, word_ngrams={self.word_ngrams}, nwords={self.nwords})>"

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
        return self.num_buckets

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
        return self.get_hash(subword) % self.num_buckets + self.nwords

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

    def indices_matrix(self, sentence: str) -> np.array:
        """
        Returns an array of token indices for a text description.

        Args:
            sentence (str): Text description.

        Returns:
            np.array: Array of indices.
        """
        indices = []
        words = []
        word_ngram_ids = []
        all_tokens_id = {}
        for word in sentence.split(" "):
            tokens, ind = self.get_subwords(word)
            indices += ind
            for idx, tok in enumerate(tokens):
                if tok not in all_tokens_id.keys():
                    all_tokens_id[tok] = ind[idx]

            words += [word]
        # Adding end of string token
        indices += [0]
        words += ["</s>"]
        all_tokens_id["</s>"] = 0

        # Adding word n-grams
        for word_ngram_len in range(2, self.word_ngrams + 1):
            for i in range(len(words) - word_ngram_len + 1):
                gram = words[i : i + word_ngram_len]
                gram = " ".join(gram)

                hashes = tuple(self.get_hash(word) for word in gram)
                word_ngram_id = int(self.get_word_ngram_id(hashes, self.num_buckets, self.nwords))
                all_tokens_id[gram] = word_ngram_id
                word_ngram_ids.append(word_ngram_id)

        all_indices = indices + word_ngram_ids

        id_to_token = {v: k for k, v in all_tokens_id.items()}

        return torch.Tensor(all_indices), id_to_token, all_tokens_id

    def tokenize(self, text: list[str], text_tokens=True):
        """
        Tokenize a list of sentences.

        Args:
            sentence (list[str]): List of sentences.

        Returns:
            np.array: Array of indices.
        """
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
            tokenized_text_tokens = tokenized_text_in_tokens(tokenized_text, id_to_token_dicts)
            return tokenized_text_tokens, tokenized_text, id_to_token_dicts, token_to_id_dicts
        else:
            return tokenized_text, id_to_token_dicts, token_to_id_dicts

    def get_vocab(self):
        return self.word_id_mapping
