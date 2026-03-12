# Src/masterthesis/algorithms/notears.py

import numpy as np
from causal_discovery.algos.notears import NoTears


def run_notears(X, n_outer=20, n_inner=100, threshold=0.3):
    """
    Unified NOTEARS wrapper.

    Returns
    -------
    dag : np.ndarray (binary adjacency matrix)
    """

    model = NoTears(rho=1, alpha=0.1, l1_reg=0, lr=1e-2)
    model.learn(X, n_outer_iter=n_outer, n_inner_iter=n_inner)

    W = model.get_result()
    dag = (np.abs(W) > threshold).astype(int)

    return dag