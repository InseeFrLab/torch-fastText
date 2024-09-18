"""
NGramTokenizer class.
"""
import numpy as np
from typing import List, Tuple
from utils import get_hash, get_word_ngram_id


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
        self.buckets = buckets
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
        return self.buckets

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
        return [word[i: i + n] for i in range(len(word) - n + 1)]

    def get_subword_index(self, subword: str) -> int:
        """
        Return the row index from the embedding matrix which
        corresponds to a character n-gram.

        Args:
            subword (str): Character n-gram.

        Returns:
            int: Index.
        """
        return get_hash(subword) % self.buckets + self.nwords

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
            indices = [self.get_word_index(word)] + indices
            tokens = [word] + tokens
            
        except KeyError:
            #print("Token was not in mapping, not adding it to subwords.")
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
                all_tokens_id[tok] = ind[idx]

            words += [word]

        # Adding end of string token
        indices += [0]
        words += ["</s>"]

        # Adding word n-grams
        for word_ngram_len in range(2, self.word_ngrams + 1):
            for i in range(len(words) - word_ngram_len + 1):
                gram = words[i: i + word_ngram_len]
                gram = ' '.join(gram)

                hashes = tuple(get_hash(word) for word in gram)
                word_ngram_id = int(
                    get_word_ngram_id(hashes, self.buckets, self.nwords)
                )

                all_tokens_id[gram] = word_ngram_id
                word_ngram_ids.append(word_ngram_id)

        all_indices = indices + word_ngram_ids
        id_to_token = {v: k for k, v in all_tokens_id.items()}
        return np.asarray(all_indices), id_to_token
