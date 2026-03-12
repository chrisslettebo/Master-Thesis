# Src/masterthesis/visualization.py

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns


# ==================================================
# Plot and save DAG graph
# ==================================================

def save_graph_plot(cols, dag, save_path: str | None = None):
    """
    Save DAG as network graph image.
    """

    G = nx.DiGraph()
    G.add_nodes_from(cols)

    for i, src in enumerate(cols):
        for j, tgt in enumerate(cols):
            if dag[i, j] == 1:
                G.add_edge(src, tgt)

    plt.figure(figsize=(7, 6))
    pos = nx.circular_layout(G)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2000,
        font_size=10,
        arrows=True,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.1",
    )
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()
    plt.close()


# ==================================================
# DAG frequency heatmap
# ==================================================

def plot_dag_frequency_heatmap(freq_matrix, cols):
    """
    Plot frequency heatmap of edges across DAGs.
    """

    freq_matrix = np.array(freq_matrix)
    n = len(cols)

    size = max(8, n * 2)
    plt.figure(figsize=(size, size), dpi=200)

    ax = sns.heatmap(
        freq_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=cols,
        yticklabels=cols,
        cmap="viridis",
        linewidths=0.5,
        square=True,
        vmin=0,
        vmax=1,
        cbar=True,
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
