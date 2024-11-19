"""
Processing fns.
"""

import string
import pandas as pd
import unidecode
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
import numpy as np


def clean_text_feature(text: list[str], remove_stop_words=True) -> pd.DataFrame:
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
    text = np.vectorize(unidecode.unidecode)(np.array(text))

    # To lowercase
    text = np.char.lower(text)

    # Remove one letter words
    mylambda = lambda x: " ".join([w for w in x.split() if len(w) > 1])
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


def categorize_surface(
    df: pd.DataFrame, surface_feature_name: int, like_sirene_3: bool = True
) -> pd.DataFrame:
    """
    Categorize the surface of the activity.

    Args:
        df (pd.DataFrame): DataFrame to categorize.
        surface_feature_name (str): Name of the surface feature.
        like_sirene_3 (bool): If True, categorize like Sirene 3.

    Returns:
        pd.DataFrame: DataFrame with a new column "surf_cat".
    """
    df_copy = df.copy()
    df_copy[surface_feature_name] = df_copy[surface_feature_name].replace("nan", np.nan)
    df_copy[surface_feature_name] = df_copy[surface_feature_name].astype(float)
    # Check surface feature exists
    if surface_feature_name not in df.columns:
        raise ValueError(f"Surface feature {surface_feature_name} not found in DataFrame.")
    # Check surface feature is a float variable
    if not (pd.api.types.is_float_dtype(df_copy[surface_feature_name])):
        raise ValueError(f"Surface feature {surface_feature_name} must be a float variable.")

    if like_sirene_3:
        # Categorize the surface
        df_copy["surf_cat"] = pd.cut(
            df_copy[surface_feature_name],
            bins=[0, 120, 400, 2500, np.inf],
            labels=["1", "2", "3", "4"],
        ).astype(str)
    else:
        # Log transform the surface
        df_copy["surf_log"] = np.log(df[surface_feature_name])

        # Categorize the surface
        df_copy["surf_cat"] = pd.cut(
            df_copy.surf_log,
            bins=[0, 3, 4, 5, 12],
            labels=["1", "2", "3", "4"],
        ).astype(str)

    df_copy[surface_feature_name] = df_copy["surf_cat"].replace("nan", "0")
    df_copy[surface_feature_name] = df_copy[surface_feature_name].astype(int)
    df_copy = df_copy.drop(columns=["surf_log", "surf_cat"], errors="ignore")
    return df_copy


def clean_and_tokenize_df(df, categorical_features=["EVT", "CJ", "NAT", "TYP", "CRT"]):
    df.fillna("nan", inplace=True)

    df = df.rename(
        columns={
            "evenement_type": "EVT",
            "cj": "CJ",
            "activ_nat_et": "NAT",
            "liasse_type": "TYP",
            "activ_surf_et": "SRF",
            "activ_perm_et": "CRT",
        }
    )

    les = []
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        les.append(le)

    df = categorize_surface(df, "SRF", like_sirene_3=True)

    return df, les
