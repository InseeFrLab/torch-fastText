"""
Train the fastText model implemented with Pytorch.
"""
from pathlib import Path
import sys
import s3fs
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow
import pyarrow.parquet as pq
from preprocess import clean_text_feature
import fasttext


def get_root_path() -> Path:
    """
    Returns root path of project.

    Returns:
        Path: Root path.
    """
    return Path(__file__).parent.parent


def write_training_data(
    df: pd.DataFrame,
    y: str,
    text_feature: str,
    categorical_features: Optional[List[str]],
    label_prefix: str = "__label__",
) -> str:
    """
    Write training data to file.

    Args:
        df (pd.DataFrame): DataFrame.
        y (str): Output variable name.
        text_feature (str): Text feature.
        categorical_features (Optional[List[str]]): Categorical features.
        label_prefix (str, optional): Label prefix. Defaults to "__label__".

    Returns:
        str: Training data path.
    """
    training_data_path = get_root_path() / "data/training_data.txt"

    with open(training_data_path, "w", encoding="utf-8") as file:
        for _, item in df.iterrows():
            formatted_item = f"{label_prefix}{item[y]} {item[text_feature]}"
            for feature in categorical_features:
                formatted_item += f" {feature}_{item[feature]}"
            file.write(f"{formatted_item}\n")
    return training_data_path.as_posix()


def train_fasttext(
    df: pd.DataFrame,
    y: str,
    text_feature: str,
    categorical_features: Optional[List[str]],
    params: Dict,
):
    """
    Train method.

    Args:
        df (pd.DataFrame): Training data.
        y (str): Name of the variable to predict.
        text_feature (str): Name of the text feature.
        categorical_features (Optional[List[str]]): Names
            of the categorical features.
        params (Dict): Parameters for model and training.
    """
    max_epochs = params["max_epochs"]
    train_proportion = params["train_proportion"]
    lr = params["lr"]
    buckets = params["buckets"]
    embedding_dim = params["dim"]
    min_count = params["minCount"]
    min_n = params["minn"]
    max_n = params["maxn"]
    word_ngrams = params["wordNgrams"]

    # Train/val split
    features = [text_feature]
    if categorical_features is not None:
        features += categorical_features
    X_train, X_val, y_train, y_val = train_test_split(
        df[features],
        df[y],
        test_size=1 - train_proportion,
        random_state=0,
        shuffle=True,
    )

    df_train = pd.concat([X_train, y_train], axis=1)
    df_val = pd.concat([X_val, y_val], axis=1)

    # Train the model and log to MLflow tracking server
    params = {
        "dim": embedding_dim,
        "lr": lr,
        "epoch": max_epochs,
        "wordNgrams": word_ngrams,
        "minn": min_n,
        "maxn": max_n,
        "minCount": min_count,
        "bucket": buckets,
        "thread": 100,
        "loss": "ova",
        "label_prefix": "__label__",
    }

    # Write training data in a .txt file (fasttext-specific)
    training_data_path = write_training_data(
        df_train,
        y,
        text_feature,
        categorical_features,
    )

    # Train the fasttext model
    model = fasttext.train_supervised(
        training_data_path,
        **params,
        verbose=2
    )

    # Log parameters
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)

    # Evaluation
    val_texts = []
    for _, item in df_val.iterrows():
        formatted_item = item[text_feature]
        for feature in categorical_features:
            formatted_item += f" {feature}_{item[feature]}"
        val_texts.append(formatted_item)

    predictions = model.predict(val_texts, k=1)
    predictions = [x[0].replace("__label__", "") for x in predictions[0]]

    booleans = [
        prediction == str(label)
        for prediction, label in zip(predictions, df_val[y])
    ]
    accuracy = sum(booleans) / len(booleans)

    # Log accuracy
    mlflow.log_metric("accuracy", accuracy)


if __name__ == "__main__":
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]

    # Load data
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}, anon=True
    )
    df = (
        pq.ParquetDataset(
            "projet-formation/diffusion/mlops/data/firm_activity_data.parquet",
            filesystem=fs,
        )
        .read_pandas()
        .to_pandas()
    )
    # Subset of df to keep things short
    df = df.sample(frac=0.1)
    # Clean text feature
    df = clean_text_feature(df, text_feature="text")
    # Add fictitious additional variable
    df["additional_var"] = np.random.randint(0, 2, df.shape[0])
    # Encode classes
    encoder = LabelEncoder()
    df["nace"] = encoder.fit_transform(df["nace"])

    # Start MLflow run
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        train_fasttext(
            df=df,
            y="nace",
            text_feature="text",
            categorical_features=["additional_var"],
            params={
                "max_epochs": 50,
                "train_proportion": 0.8,
                "lr": 0.2,
                "buckets": 2000000,
                "dim": 50,
                "minCount": 1,
                "minn": 3,
                "maxn": 6,
                "wordNgrams": 3,
            },
        )
