# data/datasets.py

import pandas as pd
from masterthesis.data.transform import row_to_dataframe
import numpy as np

# --------------------------------------------------
# Helper: build Step_Index → group mapping
# --------------------------------------------------

def _build_group_mapping(groups):
    """
    groups: list of lists, e.g. [[1,2,3],[4,5],[6]]
    Returns dict mapping step_index_value → group_id
    """
    mapping = {}
    for group_id, group in enumerate(groups):
        for value in group:
            mapping[value] = group_id
    return mapping


# --------------------------------------------------
# Single environment dataset (concatenated)
# --------------------------------------------------

def build_env_dataset(data, columns, groups, sort_by="StepIndex"):

    mapping = {}
    for i, group in enumerate(groups):
        for val in group:
            mapping[val] = i

    df = data[columns + [sort_by]].copy()

    df["Group"] = df[sort_by].map(mapping)

    if df["Group"].isnull().any():
        print("Dropping unmapped StepIndex rows")
        df = df.dropna(subset=["Group"])

    df = df.drop(columns=[sort_by])
    df["Group"] = df["Group"].astype(int)

    return df


# --------------------------------------------------
# Multi-environment dataset (dict per env)
# --------------------------------------------------

def build_env_dict(data, columns, groups, sort_by="StepIndex"):

    full_df = build_env_dataset(data, columns, groups, sort_by)

    env_dict = {}

    for group_id in sorted(full_df["Group"].unique()):
        X = full_df[full_df["Group"] == group_id].drop(columns=["Group"]).values

        if len(X) > 0:   # skip empty envs
            env_dict[group_id] = X

    return env_dict

def balanced_env_dict(data, columns, groups, sort_by="StepIndex"):

    full_df = build_env_dataset(data, columns, groups, sort_by)

    env_dict = {}

    for group_id in sorted(full_df["Group"].unique()):
        X = full_df[full_df["Group"] == group_id].drop(columns=["Group"]).values

        if len(X) > 0:
            env_dict[group_id] = X

    # balance environments
    min_size = min(len(v) for v in env_dict.values())

    for k in env_dict:
        env_dict[k] = env_dict[k][:min_size]

    return env_dict

def subsampled_env_dict(
    data,
    columns,
    groups,
    sort_by="StepIndex",
    max_samples_per_env=None,
    seed=None
):
    """
    Build environment dictionary with optional random subsampling.

    Parameters
    ----------
    max_samples_per_env : int | None
        Maximum samples per environment.
        If None → no subsampling.
    seed : int | None
        Random seed.

    Returns
    -------
    env_dict : dict[int -> np.ndarray]
    """

    if seed is not None:
        np.random.seed(seed)

    full_df = build_env_dataset(data, columns, groups, sort_by)
    print(full_df["Group"].value_counts(dropna=False))
    env_dict = {}

    for group_id in sorted(full_df["Group"].unique()):

        X = full_df[full_df["Group"] == group_id].drop(columns=["Group"]).values

        if len(X) == 0:
            continue

        if max_samples_per_env is not None and len(X) > max_samples_per_env:

            idx = np.random.choice(len(X), max_samples_per_env, replace=False)
            X = X[idx]

        env_dict[group_id] = X

    return env_dict


def balanced_subsampled_env_dict(
    data,
    columns,
    groups,
    sort_by="StepIndex",
    max_samples_per_env=None,
    seed=None
):
    """
    Build balanced environment dictionary using random sampling.

    Parameters
    ----------
    max_samples_per_env : int | None
        Optional upper bound for samples per environment.
    seed : int | None
        Random seed.
    """

    if seed is not None:
        np.random.seed(seed)

    full_df = build_env_dataset(data, columns, groups, sort_by)

    env_dict = {}

    for group_id in sorted(full_df["Group"].unique()):

        X = full_df[full_df["Group"] == group_id].drop(columns=["Group"]).values

        if len(X) == 0:
            continue

        env_dict[group_id] = X

    # determine target size
    min_size = min(len(v) for v in env_dict.values())

    if max_samples_per_env is not None:
        min_size = min(min_size, max_samples_per_env)

    # random balancing
    for k in env_dict:

        X = env_dict[k]

        if len(X) > min_size:

            idx = np.random.choice(len(X), min_size, replace=False)
            env_dict[k] = X[idx]

    return env_dict

# --------------------------------------------------
# Split observational vs interventional datasets
# --------------------------------------------------

def split_observational_interventional(data, columns, groups):
    """
    For each original row (cell/experiment), split data into:
        - observational dataset (first group)
        - interventional datasets (all groups)

    Returns:
        list of tuples:
            [(obs_list, intervention_list), ...]
    """
    results = []

    for _, row in data.iterrows():
        print(row)
        df = row_to_dataframe(row, columns)

        obs_datasets = []
        intervention_datasets = []

        for i, group in enumerate(groups):
            subset = df[df["Step_Index"].isin(group)].reset_index(drop=True)

            if i == 0:
                obs_datasets.append(subset)

            intervention_datasets.append(subset)

        results.append((obs_datasets, intervention_datasets))

    return results
