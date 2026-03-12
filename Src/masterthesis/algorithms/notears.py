# Src/masterthesis/algorithms/notears.py

import numpy as np
from causal_discovery.algos.notears import NoTears


# --------------------------------------------------
# Helper: convert weight matrix → binary DAG
# --------------------------------------------------

def weights_to_dag(W: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert NOTEARS weight matrix to binary adjacency matrix.
    """
    return (np.abs(W) > threshold).astype(int)


# --------------------------------------------------
# NOTEARS wrapper (unified interface)
# --------------------------------------------------

def run_notears(
    dataset: dict,
    n_outer: int = 20,
    n_inner: int = 100,
    threshold: float = 0.3,
):
    """
    Parameters
    ----------
    dataset : dict
        Standard dataset object:
        {
            "X": numpy array (required),
            "envs": optional,
            "interventions": optional
        }

    Returns
    -------
    dag : np.ndarray (binary adjacency matrix)
    """

    # NOTEARS only needs observational data
    if dataset.get("X") is None:
        raise ValueError("NOTEARS requires dataset['X'].")

    X = dataset["X"]

    model = NoTears(
        rho=1,
        alpha=0.1,
        l1_reg=0,
        lr=1e-2,
    )

    model.learn(X, n_outer_iter=n_outer, n_inner_iter=n_inner)
    W_est = model.get_result()

    dag = weights_to_dag(W_est, threshold)
    return dag


# --------------------------------------------------
# Large dataset variant (subsampling)
# --------------------------------------------------

def run_notears_large(
    dataset: dict,
    subset_size: int = 5_250_000,
    n_repeats: int = 5,
    n_outer: int = 20,
    n_inner: int = 100,
    threshold: float = 0.3,
):
    """
    NOTEARS for very large datasets via subsampling.
    """

    if dataset.get("X") is None:
        raise ValueError("NOTEARS requires dataset['X'].")

    X_full = dataset["X"]
    n_samples, n_features = X_full.shape

    W_accum = np.zeros((n_features, n_features))

    for _ in range(n_repeats):
        idx = np.random.choice(
            n_samples,
            size=min(subset_size, n_samples),
            replace=False,
        )
        X_subset = X_full[idx]

        model = NoTears(rho=1, alpha=0.1, l1_reg=0, lr=1e-2)
        model.learn(X_subset, n_outer_iter=n_outer, n_inner_iter=n_inner)
        W_accum += model.get_result()

    W_mean = W_accum / n_repeats
    dag = weights_to_dag(W_mean, threshold)

    return dag

