# Src/masterthesis/data/preprocessing.py
import numpy as np
from sklearn.preprocessing import StandardScaler


# ==================================================
# Reproducibility
# ==================================================

def set_random_seed(seed: int):
    """
    Set numpy random seed for reproducible experiments.
    """
    np.random.seed(seed)


# ==================================================
# Scaling
# ==================================================


def scale_across_environments(env_dict):
    """
    Scale all environments using statistics from concatenated data.
    """

    # Stack all environments
    all_data = np.vstack(list(env_dict.values()))

    scaler = StandardScaler()
    scaler.fit(all_data)

    scaled_envs = {}

    for env, X in env_dict.items():
        scaled_envs[env] = scaler.transform(X)

    return scaled_envs


# ==================================================
# Noise injection
# ==================================================

def add_gaussian_noise(env_dict, noise_std: float):
    """
    Add Gaussian noise to each environment dataset.

    Parameters
    ----------
    env_dict : dict[int -> np.ndarray]
    noise_std : float

    Returns
    -------
    noisy_env_dict : dict[int -> np.ndarray]
    """

    noisy_envs = {}

    for env, X in env_dict.items():
        noise = np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=X.shape
        )
        noisy_envs[env] = X + noise

    return noisy_envs


# ==================================================
# Convenience pipeline
# ==================================================

def preprocess_env_data(
    env_dict,
    scale: bool = True,
    noise_std: float | None = None,
    seed: int | None = None,
):
    """
    Full preprocessing pipeline used by experiments.

    This is the function your experiments will call.

    Steps:
        1) set seed
        2) scale
        3) add noise

    Returns
    -------
    env_dict_preprocessed : dict[int -> np.ndarray]
    """

    if seed is not None:
        set_random_seed(seed)

    # Step 1 — scaling
    if scale:
        env_dict = scale_across_environments(env_dict)

    print(env_dict.keys())
    print("after scale:", env_dict[0].std(axis=0))

    # Step 2 — noise injection
    if noise_std is not None:
        env_dict = add_gaussian_noise(env_dict, noise_std)

    print("after noise:", env_dict[0].std(axis=0))

    return env_dict

