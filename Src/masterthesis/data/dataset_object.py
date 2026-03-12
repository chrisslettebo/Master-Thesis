# Src/masterthesis/data/dataset_object.py

import numpy as np


# ==================================================
# Helper: stack environments
# ==================================================

def _stack_environments(env_dict):
    """
    Stack environment dictionary into one big numpy array.
    Assumes env_dict values are numpy arrays.
    """
    return np.vstack(list(env_dict.values()))


# ==================================================
# Create dataset from single numpy array
# ==================================================

def make_observational_dataset(X: np.ndarray):
    """
    Create dataset object for purely observational algorithms.
    """

    return {
        "X": X,
        "envs": None,
        "interventions": None,
    }


# ==================================================
# Create dataset from environment dictionary
# ==================================================

def make_environment_dataset(
    env_dict: dict,
    interventions: dict | None = None,
):
    """
    Create dataset object from environment dictionary.

    Parameters
    ----------
    env_dict : dict[int -> np.ndarray]
        Preprocessed environments.
    interventions : dict[int -> list[int]] | None
        Mapping env_id -> intervened variable indices.

    Returns
    -------
    dataset : dict
    """

    X_all = _stack_environments(env_dict)

    return {
        "X": X_all,
        "envs": env_dict,
        "interventions": interventions,
    }
