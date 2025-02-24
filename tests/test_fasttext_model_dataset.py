import pytest
from torchFastText.datasets.dataset import FastTextModelDataset
from torchFastText.datasets.tokenizer import NGramTokenizer


@pytest.fixture
def dataset():
    categorical_variables = [[1, 2], [3, 4], [5, 6]]
    texts = ["This is a test", "Another test", "Yet another test"]
    outputs = [0, 1, 0]
    tokenizer = NGramTokenizer(
        num_tokens=5,
        min_count=1,
        min_n=2,
        max_n=3,
        buckets=100,
        len_word_ngrams=2,
        training_text=texts,
    )
    return FastTextModelDataset(
        categorical_variables=categorical_variables,
        texts=texts,
        outputs=outputs,
        tokenizer=tokenizer,
    )


def test_getitem(dataset):
    text, cat_variable, y = dataset[0]
    assert text == "This is a test"
    assert cat_variable == [1, 2]
    assert y == 0
