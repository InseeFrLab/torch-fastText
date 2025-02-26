import pytest
from torchFastText.datasets.tokenizer import NGramTokenizer


def test_ngramtokenizer_init_valid():
    training_text = ["this is a test", "another test sentence"]
    tokenizer = NGramTokenizer(
        num_tokens=5,
        min_count=1,
        min_n=2,
        max_n=6,
        buckets=100,
        len_word_ngrams=2,
        training_text=training_text,
    )
    assert tokenizer.min_n == 2
    assert tokenizer.max_n == 6
    assert tokenizer.word_ngrams == 2
    assert tokenizer.nwords == 6  # "this", "is", "a", "test", "another", "sentence"


def test_ngramtokenizer_init_min_n_invalid():
    training_text = ["this is a test", "another test sentence"]
    with pytest.raises(ValueError, match="`min_n` parameter must be greater than 1."):
        NGramTokenizer(
            num_tokens=5,
            min_count=1,
            min_n=1,
            max_n=6,
            buckets=100,
            len_word_ngrams=2,
            training_text=training_text,
        )


def test_ngramtokenizer_init_max_n_invalid():
    training_text = ["this is a test", "another test sentence"]
    with pytest.raises(ValueError, match="`max_n` parameter must be smaller than 7."):
        NGramTokenizer(
            num_tokens=5,
            min_count=1,
            min_n=2,
            max_n=8,
            buckets=100,
            len_word_ngrams=2,
            training_text=training_text,
        )


def test_ngramtokenizer_init_min_count():
    training_text = ["this is a test", "this is another test"]
    tokenizer = NGramTokenizer(
        num_tokens=5,
        min_count=2,
        min_n=2,
        max_n=6,
        buckets=100,
        len_word_ngrams=2,
        training_text=training_text,
    )
    assert tokenizer.nwords == 3  # "this", "is", "test" (appears at least twice)


def test_ngramtokenizer_word_id_mapping():
    training_text = ["this is a test", "this is another test"]
    tokenizer = NGramTokenizer(
        num_tokens=5,
        min_count=1,
        min_n=2,
        max_n=6,
        buckets=100,
        len_word_ngrams=2,
        training_text=training_text,
    )
    expected_mapping = {"this": 1, "is": 2, "a": 3, "test": 4, "another": 5}
    assert tokenizer.word_id_mapping == expected_mapping


def test_ngramtokenizer_get_ngram_list():
    word = "test"
    n = 2
    ngrams = NGramTokenizer.get_ngram_list(word, n)
    print(ngrams)
    assert ngrams == ["te", "es", "st"]


def test_ngramtokenizer_get_subwords():
    training_text = ["this is a test", "this is another test"]
    tokenizer = NGramTokenizer(
        num_tokens=5,
        min_count=1,
        min_n=2,
        max_n=3,
        buckets=100,
        len_word_ngrams=2,
        training_text=training_text,
    )
    subwords = tokenizer.get_subwords("this is a test")
    print(subwords)
    assert subwords[0] == [
        "<t",
        "th",
        "hi",
        "is",
        "s ",
        " i",
        "is",
        "s ",
        " a",
        "a ",
        " t",
        "te",
        "es",
        "st",
        "t>",
        "<th",
        "thi",
        "his",
        "is ",
        "s i",
        " is",
        "is ",
        "s a",
        " a ",
        "a t",
        " te",
        "tes",
        "est",
        "st>",
    ]
