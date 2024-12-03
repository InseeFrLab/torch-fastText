import logging

import numpy as np

logger = logging.getLogger(__name__)


def check_X(X):
    assert isinstance(
        X, np.ndarray
    ), "X must be a numpy array of shape (N,d), with the first column being the text and the rest being the categorical variables."

    try:
        if X.ndim > 1:
            text = X[:, 0].astype(str)
        else:
            text=X[:].astype(str)
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
                f"Columns {1} to {X.shape[1]-1} of X_train must be castable in integer format."
            )
    else:
        categorical_variables = None

    return text, categorical_variables, no_cat_var


def check_Y(Y):
    assert isinstance(Y, np.ndarray), "Y must be a numpy array of shape (N,) or (N,1)."
    assert len(Y.shape) == 1 or (
        len(Y.shape) == 2 and Y.shape[1] == 1
    ), "Y must be a numpy array of shape (N,) or (N,1)."

    try:
        Y = Y.astype(int)
    except ValueError:
        logger.error("Y must be castable in integer format.")

    return Y
