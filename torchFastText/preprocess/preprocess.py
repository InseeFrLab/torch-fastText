"""
Processing fns.
"""

import string

import nltk
import numpy as np
import pandas as pd
import unidecode
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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


def clean_and_tokenize_df(
    df,
    categorical_features=["EVT", "CJ", "NAT", "TYP", "CRT"],
    text_feature="libelle_processed",
    label_col="apet_finale",
):
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
    df = df[[text_feature, "EVT", "CJ", "NAT", "TYP", "SRF", "CRT", label_col]]

    return df, les


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
