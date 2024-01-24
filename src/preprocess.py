"""
Processing fns.
"""
import string
import pandas as pd
import unidecode
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer


def clean_text_feature(
    df: pd.DataFrame, text_feature: str
) -> pd.DataFrame:
    """
    Cleans a text feature for pd.DataFrame `df`.

    Args:
        df (pd.DataFrame): DataFrame.
        text_feature (str): Name of the text feature.

    Returns:
        df (pd.DataFrame): DataFrame.
    """
    # Define stopwords and stemmer
    stopwords = tuple(ntlk_stopwords.words("french")) + tuple(string.ascii_lowercase)
    stemmer = SnowballStemmer(language="french")

    # Remove of accented characters
    df[text_feature] = df[text_feature].map(unidecode.unidecode)

    # To lowercase
    df[text_feature] = df[text_feature].str.lower()

    # Remove one letter words
    df[text_feature] = df[text_feature].apply(
        lambda x: " ".join([w for w in x.split() if len(w) > 1])
    )

    # Remove duplicate words and stopwords in texts
    # Stem words
    libs_token = [lib.split() for lib in df[text_feature].to_list()]
    libs_token = [
        sorted(set(libs_token[i]), key=libs_token[i].index)
        for i in range(len(libs_token))
    ]
    df[text_feature] = [
        " ".join(
            [
                stemmer.stem(word)
                for word in libs_token[i]
                if word not in stopwords
            ]
        )
        for i in range(len(libs_token))
    ]

    # Return clean DataFrame
    return df
