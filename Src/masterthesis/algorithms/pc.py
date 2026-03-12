# Src/masterthesis/algorithms/pc.py

import numpy as np
from causallearn.search.ConstraintBased.PC import pc


# --------------------------------------------------
# Convert causal-learn graph → adjacency matrix
# --------------------------------------------------

def _causal_learn_graph_to_adj(G) -> np.ndarray:
    """
    Convert causal-learn Graph object to adjacency matrix.
    """
    # causal-learn stores adjacency as numpy matrix already
    adj = np.array(G.graph)

    # Convert to binary adjacency matrix (remove weights/orientation codes)
    adj = (adj != 0).astype(int)

    return adj


# --------------------------------------------------
# PC wrapper (unified interface)
# --------------------------------------------------

def run_pc(
    dataset: dict,
    alpha: float = 0.05,
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

    if dataset.get("X") is None:
        raise ValueError("PC requires dataset['X'].")

    X = dataset["X"]

    result = pc(X, alpha=alpha)

    dag = _causal_learn_graph_to_adj(result.G)
    return dag
