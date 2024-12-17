"""
Processing fns.
"""

import string

import numpy as np

try:
    import nltk
    from nltk.corpus import stopwords as ntlk_stopwords
    from nltk.stem.snowball import SnowballStemmer

    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    import unidecode

    HAS_UNIDECODE = True
except ImportError:
    HAS_UNIDECODE = False


def clean_text_feature(text: list[str], remove_stop_words=True):
    """
    Cleans a text feature.

    Args:
        text (list[str]): List of text descriptions.
        remove_stop_words (bool): If True, remove stopwords.

    Returns:
        list[str]: List of cleaned text descriptions.

    """
    if not HAS_NLTK:
        raise ImportError(
            "nltk is not installed and is required for preprocessing. Run 'pip install torchFastText[preprocess]'."
        )
    if not HAS_UNIDECODE:
        raise ImportError(
            "unidecode is not installed and is required for preprocessing. Run 'pip install torchFastText[preprocess]'."
        )

    # Define stopwords and stemmer

    nltk.download("stopwords", quiet=True)
    stopwords = tuple(ntlk_stopwords.words("french")) + tuple(string.ascii_lowercase)
    stemmer = SnowballStemmer(language="french")

    # Remove of accented characters
    text = np.vectorize(unidecode.unidecode)(np.array(text))

    # To lowercase
    text = np.char.lower(text)

    # Remove one letter words
    def mylambda(x):
        return " ".join([w for w in x.split() if len(w) > 1])

    text = np.vectorize(mylambda)(text)

    # Remove duplicate words and stopwords in texts
    # Stem words
    libs_token = [lib.split() for lib in text.tolist()]
    libs_token = [
        sorted(set(libs_token[i]), key=libs_token[i].index) for i in range(len(libs_token))
    ]
    if remove_stop_words:
        text = [
            " ".join([stemmer.stem(word) for word in libs_token[i] if word not in stopwords])
            for i in range(len(libs_token))
        ]
    else:
        text = [
            " ".join([stemmer.stem(word) for word in libs_token[i]]) for i in range(len(libs_token))
        ]

    # Return clean DataFrame
    return text
