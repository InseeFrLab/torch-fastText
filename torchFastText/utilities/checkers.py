import logging
import json
from typing import Optional, Union, Type, List

import numpy as np

logger = logging.getLogger(__name__)


def check_X(X):
    assert isinstance(X, np.ndarray), (
        "X must be a numpy array of shape (N,d), with the first column being the text and the rest being the categorical variables."
    )

    try:
        if X.ndim > 1:
            text = X[:, 0].astype(str)
        else:
            text = X[:].astype(str)
    except ValueError:
        logger.error("The first column of X must be castable in string format.")

    if len(X.shape) == 1 or (len(X.shape) == 2 and X.shape[1] == 1):
        no_cat_var = True
    else:
        no_cat_var = False

    if not no_cat_var:
        try:
            categorical_variables = X[:, 1:].astype(int)
        except ValueError:
            logger.error(
                f"Columns {1} to {X.shape[1] - 1} of X_train must be castable in integer format."
            )
    else:
        categorical_variables = None

    return text, categorical_variables, no_cat_var


def check_Y(Y):
    assert isinstance(Y, np.ndarray), "Y must be a numpy array of shape (N,) or (N,1)."
    assert len(Y.shape) == 1 or (len(Y.shape) == 2 and Y.shape[1] == 1), (
        "Y must be a numpy array of shape (N,) or (N,1)."
    )

    try:
        Y = Y.astype(int)
    except ValueError:
        logger.error("Y must be castable in integer format.")

    return Y

def validate_categorical_inputs(categorical_vocabulary_sizes: List[int],
                                categorical_embedding_dims: Union[List[int], int],
                                num_categorical_features: int = None):
    
    if categorical_vocabulary_sizes is None:
        logger.warning(
            "No categorical_vocabulary_sizes. It will be inferred later."
        )
        return None, None, None
    
    else:
        if not isinstance(categorical_vocabulary_sizes, list):
            raise TypeError("categorical_vocabulary_sizes must be a list of int")

        if isinstance(categorical_embedding_dims, list):
            if len(categorical_vocabulary_sizes) != len(categorical_embedding_dims):
                raise ValueError(
                    "Categorical vocabulary sizes and their embedding dimensions must have the same length"
                )

        if num_categorical_features is not None:
            if len(categorical_vocabulary_sizes) != num_categorical_features:
                raise ValueError(
                    "len(categorical_vocabulary_sizes) must be equal to num_categorical_features"
                )
        else:
            num_categorical_features = len(categorical_vocabulary_sizes)
        
    assert num_categorical_features is not None, "num_categorical_features should be inferred at this point."
    
    # "Transform" embedding dims into a suitable list, or stay None
    if categorical_embedding_dims is not None:
        if isinstance(categorical_embedding_dims, int):
            categorical_embedding_dims = [
                categorical_embedding_dims
            ] * num_categorical_features
        elif not isinstance(categorical_embedding_dims, list):
            raise TypeError("categorical_embedding_dims must be an int or a list of int")
    
    assert isinstance(categorical_embedding_dims, list) or categorical_embedding_dims is None, "categorical_embedding_dims must be a list of int at this point"

    return categorical_vocabulary_sizes, categorical_embedding_dims, num_categorical_features

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
