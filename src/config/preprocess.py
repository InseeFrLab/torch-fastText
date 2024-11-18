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


def clean_text_feature(df: pd.DataFrame, text_feature: str) -> pd.DataFrame:
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
        sorted(set(libs_token[i]), key=libs_token[i].index) for i in range(len(libs_token))
    ]
    df[text_feature] = [
        " ".join([stemmer.stem(word) for word in libs_token[i] if word not in stopwords])
        for i in range(len(libs_token))
    ]

    # Return clean DataFrame
    return df


def clean_text_input(text: list[str]) -> pd.DataFrame:
    """
    Cleans a text feature for pd.DataFrame `df`.

    Args:
        df (pd.DataFrame): DataFrame.
        text_feature (str): Name of the text feature.

    Returns:
        df (pd.DataFrame): DataFrame.
    """

    text = pd.Series(text)
    # Define stopwords and stemmer
    stopwords = tuple(ntlk_stopwords.words("french")) + tuple(string.ascii_lowercase)
    stemmer = SnowballStemmer(language="french")

    # Remove of accented characters
    text = text.map(unidecode.unidecode)

    # To lowercase
    text = text.str.lower()

    # Remove one letter words
    text = text.apply(lambda x: " ".join([w for w in x.split() if len(w) > 1]))

    # Remove duplicate words and stopwords in texts
    # Stem words
    libs_token = [lib.split() for lib in text.to_list()]
    libs_token = [
        sorted(set(libs_token[i]), key=libs_token[i].index) for i in range(len(libs_token))
    ]
    text = [
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


def clean_and_tokenize_df(df, categorical_features= ["EVT", "CJ", "NAT", "TYP", "CRT"]):
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
