"""
Processing fns.
"""

import string

import nltk
import numpy as np
import unidecode
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split


def clean_text_feature(text: list[str], remove_stop_words=True):
    """
    Cleans a text feature.

    Args:
        text (list[str]): List of text descriptions.
        remove_stop_words (bool): If True, remove stopwords.

    Returns:
        list[str]: List of cleaned text descriptions.

    """
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


def stratified_split_rare_labels(X, y, test_size=0.2, min_train_samples=1):
    # Get unique labels and their frequencies
    unique_labels, label_counts = np.unique(y, return_counts=True)

    # Separate rare and common labels
    rare_labels = unique_labels[label_counts == 1]

    # Create initial mask for rare labels to go into training set
    rare_label_mask = np.isin(y, rare_labels)

    # Separate data into rare and common label datasets
    X_rare = X[rare_label_mask]
    y_rare = y[rare_label_mask]
    X_common = X[~rare_label_mask]
    y_common = y[~rare_label_mask]

    # Split common labels stratified
    X_common_train, X_common_test, y_common_train, y_common_test = train_test_split(
        X_common, y_common, test_size=test_size, stratify=y_common
    )

    # Combine rare labels with common labels split
    X_train = np.concatenate([X_rare, X_common_train])
    y_train = np.concatenate([y_rare, y_common_train])
    X_test = X_common_test
    y_test = y_common_test

    return X_train, X_test, y_train, y_test
