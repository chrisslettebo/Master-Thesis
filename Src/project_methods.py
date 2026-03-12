import os
import numpy as np
import networkx as nx
from causal_discovery.algos.notears import NoTears
import hashlib
from matplotlib import pyplot as plt
import seaborn as sns

class ProjectMethods:

    @staticmethod
    def NoTearsByGroup(subsets, save_point, threshold=0.3, n_outer = 20, n_inner=100):
        os.makedirs(save_point, exist_ok=True)
        dag_counter = {}

        for key, data in subsets.items():

            # --- directory for group
            group_dir = os.path.join(save_point, f"group_{key}")
            os.makedirs(group_dir, exist_ok=True)

            # --- prepare data
            subdata = data.drop(columns=["Group"])
            X = subdata.to_numpy(dtype=float)
            cols = subdata.columns.tolist()

            # --- run NoTears
            no_tears_alg = NoTears(rho=1, alpha=0.1, l1_reg=0, lr=1e-2)
            no_tears_alg.learn(X, n_outer_iter=n_outer, n_inner_iter=n_inner)
            W_est = no_tears_alg.get_result()

            # --- convert to DAG adjacency
            dag = (np.abs(W_est) > threshold).astype(int)

            # count identical DAGs
            dag_hash = hashlib.md5(dag.tobytes()).hexdigest()
            dag_counter[dag_hash] = dag_counter.get(dag_hash, 0) + 1

            # --- save DAG & graph
            ProjectMethods.makeAndSaveGraph(cols, dag, group_dir)
            np.save(os.path.join(group_dir, "dag.npy"), dag)
            np.save(os.path.join(group_dir, "cols.npy"), np.array(cols))

        return dag_counter


    @staticmethod
    def makeAndSaveGraph(cols, dag, group_dir):
        G = nx.DiGraph()
        G.add_nodes_from(cols)  # ensure all nodes are drawn

        for i, src in enumerate(cols):
            for j, tgt in enumerate(cols):
                if dag[i, j] == 1:
                    G.add_edge(src, tgt)

        plt.figure(figsize=(7, 6))
        pos = nx.circular_layout(G)  
        nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, arrows=True, arrowstyle='->', connectionstyle='arc3,rad=0.1')
        plt.savefig(os.path.join(group_dir, "dag.png"), dpi=200, bbox_inches="tight")
        plt.close()

    
    @staticmethod
    def load_all_dags(result_folder):
        dag_list = []
        cols = None

        for folder in os.listdir(result_folder):
            group_dir = os.path.join(result_folder, folder)

            dag_path = os.path.join(group_dir, "dag.npy")
            cols_path = os.path.join(group_dir, "cols.npy")

            if os.path.exists(dag_path):
                dag = np.load(dag_path)
                dag_list.append(dag)

                if cols is None and os.path.exists(cols_path):
                    cols = np.load(cols_path).tolist()

        return dag_list, cols
    
    @staticmethod
    def dag_frequency_matrix(dag_list):
        freq = np.sum(dag_list, axis=0)   # sum occurrences edge-wise
        freq = freq / len(dag_list)       # convert to frequency 0–1
        return freq
    

    @staticmethod
    def plot_dag_frequency_heatmap(freq_matrix, cols, save_path=None):
        """
        Plots a DAG frequency heatmap with annotations and adaptive text colors.

        Parameters:
        - freq_matrix: 2D array-like, frequency values between 0 and 1
        - cols: list of column/row labels
        - save_path: optional path to save the figure
        """
        # Ensure it's a numpy array
        freq_matrix = np.array(freq_matrix)
        n = len(cols)

        # Set figure size proportional to number of columns
        size = max(8, n * 2)
        plt.figure(figsize=(size, size), dpi=200)

        # Create heatmap
        ax = sns.heatmap(
            freq_matrix,
            annot=True,           # Let seaborn handle annotations
            fmt=".3f",            # 3 decimals for clarity
            xticklabels=cols,
            yticklabels=cols,
            cmap="viridis",
            linewidths=0.5,
            square=True,
            vmin=0, vmax=1,
            cbar=True,
            annot_kws={"size": 12}  # base font size
        )

        # Adjust text color based on cell background
        for text in ax.texts:
            try:
                val = float(text.get_text())
            except ValueError:
                val = 0
            # Use white text for dark cells, black for light cells
            text.set_color("white" if val < 0.2 else "black")

        # Ticks formatting
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


    @staticmethod
    def NoTearsLargeDataset(X_full, cols, save_point, threshold=0.3,
                            subset_size=250000, n_repeats=5, n_outer=20, n_inner=100):
        """
        Run NoTears on a very large dataset using repeated random subsamples.

        Parameters:
        - X_full: np.array, full dataset (samples x features)
        - cols: list of column names
        - save_point: directory to save results
        - threshold: edge inclusion threshold
        - subset_size: number of rows per subsample
        - n_repeats: number of random subsamples to average results
        - n_outer, n_inner: NOTEARS iterations per subsample

        Returns:
        - dag: final adjacency matrix (binary)
        """
        os.makedirs(save_point, exist_ok=True)
        n_samples, n_features = X_full.shape
        print(n_samples)

        W_accum = np.zeros((n_features, n_features))

        for repeat in range(n_repeats):
            # Random subsample
            idx = np.random.choice(n_samples, size=min(subset_size, n_samples), replace=False)
            X_subset = X_full[idx]

            print(f"Running subsample {repeat+1}/{n_repeats} ({X_subset.shape[0]} rows)...")

            # Initialize NoTears
            no_tears_alg = NoTears(rho=1, alpha=0.1, l1_reg=0, lr=1e-2)
            no_tears_alg.learn(X_subset, n_outer_iter=n_outer, n_inner_iter=n_inner)
            W_est = no_tears_alg.get_result()

            # Accumulate adjacency weights
            W_accum += W_est

        # Average over repeats
        W_mean = W_accum / n_repeats

        # Threshold to get final DAG
        dag = (np.abs(W_mean) > threshold).astype(int)
        
        print(cols)
        ProjectMethods.makeAndSaveGraph(cols, dag, save_point)
        
        # Save results
        np.save(os.path.join(save_point, "dag.npy"), dag)
        np.save(os.path.join(save_point, "cols.npy"), np.array(cols))

        print(f"Saved DAG with threshold {threshold} at {save_point}")
        return dag

