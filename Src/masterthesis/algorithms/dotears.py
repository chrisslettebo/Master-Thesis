import numpy as np
from .dotears_impl import DOTEARS   # your class file
from .test_tears import TEST_DOTEARS, TEST_DOTEARS2


def weights_to_dag(W, threshold):
    return (np.abs(W) > threshold).astype(int)


def run_dotears(
    dataset: dict,
    lambda1: float = 0.1,
    threshold: float = 0.3,
    scaled: bool = False,
):
    """
    Unified wrapper for DO-TEARS.

    Parameters
    ----------
    dataset : dict
        dataset object with envs
    """
    envs = dataset["envs"]
    interventions = dataset["interventions"]

    data = {}

    for env_id, X in envs.items():

        targets = interventions.get(env_id, [])

        if len(targets) == 0:
            data["obs"] = X
        else:
            for t in targets:
                key = f"{t}_int"
                data[key] = X

    model = DOTEARS(data, lambda1=lambda1, scaled=scaled)
    W = model.fit()

    dag = (np.abs(W) > threshold).astype(int)

    return dag


def run_dotears_test(dataset, lambda1: float = 0.1, w_threshold: float = 0.3,):

    env_dict = dataset["envs"]
    interventions = dataset["interventions"]

    model = TEST_DOTEARS(
        env_dict,
        interventions=interventions,
        lambda1=lambda1,
        w_threshold=w_threshold
    )

    W = model.fit()

    dag = (np.abs(W) > 0).astype(int)

    return dag

def run_dotears_test2(dataset, lambda1: float = 0.1, w_threshold: float = 0.3,):

    env_dict = dataset["envs"]
    interventions = dataset["interventions"]

    model = TEST_DOTEARS2(
        env_dict,
        interventions=interventions,
        lambda1=lambda1,
        w_threshold=w_threshold
    )

    W = model.fit()

    dag = (np.abs(W) > 0).astype(int)

    return dag