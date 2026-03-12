# data/transform.py

import numpy as np
import pandas as pd


# --------------------------------------------------
# Basic array cleaning
# --------------------------------------------------

def flatten_array(arr):
    """
    Flatten MATLAB-style nested arrays like [[x],[y],[z]] → [x,y,z]
    """
    return np.array([
        x[0] if isinstance(x, (list, np.ndarray)) else x
        for x in arr
    ])


# --------------------------------------------------
# Convert one experiment/cell row → long DataFrame
# --------------------------------------------------

def row_to_dataframe(row, columns):
    """
    Convert one row from loaders dataframe into a clean long-format dataframe.

    Each column in `columns` becomes a time-series column.
    """
    clean = {}

    for name in columns:
        clean[name] = flatten_array(row[name])

    return pd.DataFrame(clean)
