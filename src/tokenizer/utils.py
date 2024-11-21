import ctypes
from typing import Tuple
import numpy as np


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
