# Src/masterthesis/evaluation.py

import numpy as np
from pathlib import Path


# ==================================================
# Load saved DAGs
# ==================================================

def load_all_dags(result_folder: str):
    """
    Load all DAGs saved by experiments.

    Returns
    -------
    dag_list : list[np.ndarray]
    """

    result_folder = Path(result_folder)
    dag_list = []

    for path in result_folder.rglob("dag.npy"):
        dag = np.load(path)
        dag_list.append(dag)

    if len(dag_list) == 0:
        raise ValueError("No DAGs found in folder.")

    return dag_list


# ==================================================
# DAG frequency matrix (your original idea ⭐)
# ==================================================

def dag_frequency_matrix(dag_list):
    """
    Compute edge frequency matrix across many DAGs.
    """

    dag_stack = np.stack(dag_list, axis=0)
    freq = np.sum(dag_stack, axis=0) / len(dag_list)

    return freq


# ==================================================
# Simple metrics (optional but useful)
# ==================================================

def edge_count(dag: np.ndarray) -> int:
    return int(np.sum(dag))


def average_edge_count(dag_list):
    return float(np.mean([edge_count(d) for d in dag_list]))
