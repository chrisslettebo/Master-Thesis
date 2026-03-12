# Src/masterthesis/experiments.py

import json
import numpy as np
from pathlib import Path
from typing import Dict, Callable

from masterthesis.data.dataset_object import make_environment_dataset
from masterthesis.visualization import save_graph_plot


# ==================================================
# Helpers
# ==================================================

def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _hash_dag(dag: np.ndarray) -> str:
    import hashlib
    return hashlib.md5(dag.tobytes()).hexdigest()


def _save_dag(save_dir: Path, dag: np.ndarray):
    np.save(save_dir / "dag.npy", dag)


def _save_config(save_dir: Path, config: dict):
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


# ==================================================
# Run single algorithm
# ==================================================

def run_algorithm(
    dataset: dict,
    algorithm_fn: Callable,
    algorithm_name: str,
    save_dir: Path,
    algo_params: dict | None = None,
):
    """
    Run one algorithm and save results.
    """

    if algo_params is None:
        algo_params = {}

    _ensure_dir(save_dir)

    dag = algorithm_fn(dataset, **algo_params)

    _save_dag(save_dir, dag)
    _save_config(save_dir, algo_params)

    return dag


# ==================================================
# Run multiple algorithms
# ==================================================

def run_algorithms(
    dataset: dict,
    algorithms: Dict[str, Callable],
    base_save_dir: str,
    algo_params: Dict[str, dict] | None = None,
):
    """
    Run several algorithms on the same dataset.
    """

    if algo_params is None:
        algo_params = {}

    results = {}
    base_save_dir = Path(base_save_dir)

    for name, algo in algorithms.items():
        print(f"Running {name}...")

        params = algo_params.get(name, {})
        save_dir = base_save_dir / name

        dag = run_algorithm(dataset, algo, name, save_dir, params)
        results[name] = dag

    return results


# ==================================================
# Run algorithm per environment (your old NoTearsByGroup)
# ==================================================

def run_algorithm_by_environment(
    env_dict: dict,
    algorithm_fn: Callable,
    algorithm_name: str,
    save_root: str,
    interventions: dict | None = None,
    algo_params: dict | None = None,
    column_names=None,
):
    """
    Run algorithm independently on each environment.

    Returns:
        dag_counter : dict[str -> int]
            Count of identical DAGs across environments.
    """

    if algo_params is None:
        algo_params = {}

    dag_counter = {}
    save_root = Path(save_root)

    for env_id, X in env_dict.items():
        print(f"Running {algorithm_name} on environment {env_id}")

        dataset = make_environment_dataset(
            env_dict={env_id: X},
            interventions=interventions,
        )

        save_dir = save_root / f"env_{env_id}" / algorithm_name
        _ensure_dir(save_dir)

        dag = algorithm_fn(dataset, **algo_params)

        # count identical DAGs (your original idea ⭐)
        dag_hash = _hash_dag(dag)
        dag_counter[dag_hash] = dag_counter.get(dag_hash, 0) + 1

        _save_dag(save_dir, dag)

        # save graph image if column names available
        if column_names is not None:
            save_graph_plot(column_names, dag, save_dir / "dag.png")

    return dag_counter
