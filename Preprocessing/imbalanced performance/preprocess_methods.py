import numpy as np
from causal_discovery.algos.notears import NoTears
import networkx as nx
import hashlib
import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import dagma
from dagma.nonlinear import DagmaMLP, DagmaNonlinear
import dagma.utils as du
import torch
from torch import nn 

class ModuleLevelInfoInSteps:
    def getRowIndexOfStepIndex(data):
        stepStartEnd = []
        first = 0
        stepindex = 1
        for i in range(len(data['Data_processed']['Data']['Step_Index'])):
            if(data['Data_processed']['Data']['Step_Index'][i]!=stepindex):
                stepStartEnd.append([stepindex, first, i-1])
                first = i
                stepindex +=1
        stepStartEnd.append([stepindex, first, len(data['Data_processed']['Data']['Step_Index'])-1])
        return stepStartEnd

    def getTimeInSteps(data, stepStartEnd):
        timeframes = []
        for i in range(len(stepStartEnd)):
            time_in_step = data['Data_processed']['Data']['Test_Times'][stepStartEnd[i][2]]-data['Data_processed']['Data']['Test_Times'][stepStartEnd[i][1]]
            timeframes.append([i+1, time_in_step])
        return timeframes

    def getCellNames(data):
        cell1 = data['Data_processed']['Cells_name']['Position_1']
        cell2 = data['Data_processed']['Cells_name']['Position_2']
        cell3 = data['Data_processed']['Cells_name']['Position_3']
        cell4 = data['Data_processed']['Cells_name']['Position_4']
        return cell1, cell2, cell3, cell4
    
    def getDataForCellInStepIntevall(data, stepStartEnd, data_col):
        data_insteps = []
        valid = ['Step_Times', 'Cycle_Index', 'VoltageV', 'CurrentA', 'TemperatureC_Cell_1',
                 'TemperatureC_Cell_2', 'TemperatureC_Cell_3', 'TemperatureC_Cell_4', 'Ambient_TemperatureC', 
                 'CurrentA_Cell_1', 'CurrentA_Cell_2', 'CurrentA_Cell_3', 'CurrentA_Cell_4']
        if data_col in valid:
            for i in range(len(stepStartEnd)):
                data_insteps.append([i+1, data['Data_processed']['Data'][data_col][stepStartEnd[i][1]], data['Data_processed']['Data'][data_col][stepStartEnd[i][2]]])
            return data_insteps
        else: print(f'not valid data name, must be in {valid}')

    def getDataMeanForCellInStepIntevall(data, stepStartEnd, data_col):
        data_insteps = []
        valid = ['VoltageV', 'CurrentA', 'TemperatureC_Cell_1',
                 'TemperatureC_Cell_2', 'TemperatureC_Cell_3', 'TemperatureC_Cell_4', 'Ambient_TemperatureC', 
                 'CurrentA_Cell_1', 'CurrentA_Cell_2', 'CurrentA_Cell_3', 'CurrentA_Cell_4']
        if data_col in valid:
            for i in range(len(stepStartEnd)):
                data_insteps.append(np.mean([data['Data_processed']['Data'][data_col][stepStartEnd[i][1]:stepStartEnd[i][2]]]))
            return data_insteps
        else: print(f'not valid data name, must be in {valid}')

class HPPCCellCharacteristicsInSteps:
    def getRowIndexOfStepIndexForCell(data):
        stepStartEnd = []
        first = data['Step_Index'][0]
        stepindex = 1
        for i in range(len(data['Step_Index'])):
            if(data['Step_Index'][i]!=stepindex):
                stepStartEnd.append([stepindex, first, i-1])
                first = i
                stepindex +=1
        stepStartEnd.append([stepindex, first, len(data['Step_Index'])-1])
        return stepStartEnd
    
    def getTimeInStepsForCell(data, stepStartEnd):
        timeframes = []
        for i in range(len(stepStartEnd)):
            timeframes.append([i+1, data['TimeData'][stepStartEnd[i][2]]-data['TimeData'][stepStartEnd[i][1]]])
        return timeframes

    def getDataForCellInStepIntevall(data, stepStartEnd, data_col):
        data_insteps = []
        valid = ['CurrentData', 'VoltageData', 'CycleIndex', 'TempData']
        if data_col in valid:
            for i in range(len(stepStartEnd)):
                data_insteps.append([i+1, data[data_col][stepStartEnd[i][1]], data[data_col][stepStartEnd[i][2]]])
            return data_insteps
        else: print(f'not valid data name, must be in {valid}')

    def getDataMeanForCellInStepIntevall(data, stepStartEnd, data_col):
        data_insteps = []
        valid = ['CurrentData', 'VoltageData', 'TempData']
        if data_col in valid:
            for i in range(len(stepStartEnd)):
                data_insteps.append(np.mean([data[data_col][stepStartEnd[i][1]:stepStartEnd[i][2]]]))
            return data_insteps
        else: print(f'not valid data name, must be in {valid}')


class OCVCellCharacteristics:
    
    """ OCV dooes not have timesteps"""

    def getTime(data):
        ... 

    def getDataForCellInStepIntevall(data, stepStartEnd, data_col):
        data_insteps = []
        valid = ['TimeData', 'CurrentData', 'OCV', 'TempData']


    def getDataMeanForCellInStepIntevall(data, stepStartEnd, data_col):
        data_insteps = []
        valid = ['TimeData', 'CurrentData', 'OCV', 'TempData']

class HallSensorInSteps:

    def getRowIndexOfStepIndexForCell(data):
        stepStartEnd = []
        first = data['Step_Index'][0]
        stepindex = 1
        for i in range(len(data['Step_Index'])):
            if(data['Step_Index'][i]!=stepindex):
                stepStartEnd.append([stepindex, first, i-1])
                first = i
                stepindex +=1
        stepStartEnd.append([stepindex, first, len(data['Step_Index'])-1])
        return stepStartEnd
    
    def getTimeForCellInStepIntevall(data, stepStartEnd):
        data_insteps = []
        for i in range(len(stepStartEnd)):
            data_insteps.append([i+1, data['Test_Times'][stepStartEnd[i][2]]-data['Test_Times'][stepStartEnd[i][1]]])
        return data_insteps
    
    def getDataForCellInStepIntevall(data, stepStartEnd, data_col):
        data_insteps = []
        valid = ['Step_Times', 'Cycle_Index', 'CurrentA', 'HallVoltage', 'PowerSupplyVoltage', 'CellTempData', 'AmbientTempData']
        if data_col in valid:
            for i in range(len(stepStartEnd)):
                data_insteps.append([i+1, data[data_col][stepStartEnd[i][1]], data[data_col][stepStartEnd[i][2]]])
            return data_insteps
        else: print(f'not valid data name, must be in {valid}')

    def getDataMeanForCellInStepIntevall(data, stepStartEnd, data_col):
        data_insteps = []
        valid = ['CurrentA', 'HallVoltage', 'PowerSupplyVoltage', 'CellTempData', 'AmbientTempData']
        if data_col in valid:
            for i in range(len(stepStartEnd)):
                data_insteps.append(np.mean([data[data_col][stepStartEnd[i][1]:stepStartEnd[i][2]]]))
            return data_insteps
        else: print(f'not valid data name, must be in {valid}')


class CausalIndividualLevel:

    @staticmethod
    def observational_struct_learning(cell_list, obs_or_int, save_point, sort_by="Step_Index", type="cell"):
        os.makedirs(save_point, exist_ok=True)
        dag_counter = {}

        for idx, cell in enumerate(cell_list):

            cell_dir = os.path.join(save_point, f"{type}_{idx}")
            os.makedirs(cell_dir, exist_ok=True)

            for group_idx, group in enumerate(cell[obs_or_int]):

                df_cell = group.copy()
                if df_cell.empty:
                    continue

                if sort_by in df_cell.columns:
                    df_cell = df_cell.drop(columns=[sort_by])

                df_cell = df_cell.loc[:, df_cell.nunique() > 1]
                if df_cell.shape[1] < 2:
                    continue

                X = df_cell.to_numpy().astype(float)
                cols = df_cell.columns.tolist()

                no_tears_alg = NoTears(rho=1, alpha=0.1, l1_reg=0, lr=1e-2)
                no_tears_alg.learn(X, n_outer_iter=30)
                W_est = no_tears_alg.get_result()
                dag = (np.abs(W_est) > 0.3).astype(int)

                dag_hash = hashlib.md5(dag.tobytes()).hexdigest()
                dag_counter[dag_hash] = dag_counter.get(dag_hash, 0) + 1

                group_dir = os.path.join(cell_dir, f"group_{group_idx}")
                os.makedirs(group_dir, exist_ok=True)

                np.save(os.path.join(group_dir, "dag.npy"), dag)

                G = nx.DiGraph()
                for i, src in enumerate(cols):
                    for j, tgt in enumerate(cols):
                        if dag[i, j] == 1:
                            G.add_edge(src, tgt)

                plt.figure(figsize=(7, 6))
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, arrows=True)
                plt.savefig(os.path.join(group_dir, "dag.png"), dpi=200, bbox_inches="tight")
                plt.close()

        with open(os.path.join(save_point, "dag_summary.txt"), "w") as f:
            for h, count in dag_counter.items():
                f.write(f"{h}: {count}\n")

    @staticmethod
    def load_and_visualize_unique_dags(save_point):
        """
        Reads dag_summary.txt, reconstructs unique DAGs from stored dag.npy files,
        and shows them visually with their counts.
        """

        # ---- Load summary ----
        summary_path = os.path.join(save_point, "dag_summary.txt")
        if not os.path.exists(summary_path):
            print("ERROR: No dag_summary.txt found in:", save_point)
            return

        summary = {}
        with open(summary_path, "r") as f:
            for line in f:
                h, count = line.strip().split(": ")
                summary[h] = int(count)

        unique_dags = {}

        # ---- Traverse folder structure ----
        for cell in os.listdir(save_point):
            cell_path = os.path.join(save_point, cell)
            if not os.path.isdir(cell_path):
                continue

            for group in os.listdir(cell_path):
                group_path = os.path.join(cell_path, group)
                dag_path = os.path.join(group_path, "dag.npy")

                if not os.path.exists(dag_path):
                    continue

                dag = np.load(dag_path)
                dag_hash = hashlib.md5(dag.tobytes()).hexdigest()

                # Only keep DAGs mentioned in the summary
                if dag_hash in summary:
                    unique_dags[dag_hash] = dag

        # ---- Visualization ----
        print("Found", len(unique_dags), "unique DAGs.")

        n = len(unique_dags)
        plt.figure(figsize=(6 * n, 6))

        for i, (dag_hash, dag) in enumerate(unique_dags.items(), 1):
            G = nx.DiGraph()
            num_nodes = dag.shape[0]
            nodes = [f"X{i}" for i in range(num_nodes)]
            G.add_nodes_from(nodes)

            for s in range(num_nodes):
                for t in range(num_nodes):
                    if dag[s, t] == 1:
                        G.add_edge(nodes[s], nodes[t])

            plt.subplot(1, n, i)
            pos = nx.spring_layout(G, seed=0)
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_size=2000,
                font_size=10,
                arrows=True
            )
            plt.title(f"DAG {i}\nHash: {dag_hash[:6]}\nCount: {summary[dag_hash]}")

        plt.tight_layout()
        plt.show()

        return unique_dags, summary

    @staticmethod
    def flatten_array(arr):
        return np.array([x[0] if isinstance(x, (list, np.ndarray)) else x for x in arr])

    @staticmethod
    def make_cell_dataframe(row, cols):
        clean = {name: CausalIndividualLevel.flatten_array(row[name]) for name in cols}
        return pd.DataFrame(clean)

    @staticmethod
    def splitDataForEachByStep(data, cols, groups, sort_by='Step_Index'):
        cell_list = []
        for idx, row in data.iterrows():

            cell_df = CausalIndividualLevel.make_cell_dataframe(row, cols)
            obs_datasets = []
            interventional_datasets = []

            for i in range(len(groups)):
                subset = cell_df[cell_df[sort_by].isin(groups[i])].reset_index(drop=True)

                if i == 0:
                    obs_datasets.append(subset)

                interventional_datasets.append(subset)

            cell_list.append([obs_datasets, interventional_datasets])

        return cell_list

    @staticmethod
    def makeDataFrameOfExperiment(experiments, controlVarIndex, controlVar):
        experiments_seperated = []
        for i in controlVar:
            experiment_data = []
            for experiment in experiments:
                #'Test_Times', 'Step_Times', 'Step_Index', 'Cycle_Index', 'VoltageV', 'CurrentA', 'TemperatureC_Cell_1', 'TemperatureC_Cell_2', 'TemperatureC_Cell_3', 'TemperatureC_Cell_4', 
                # 'Ambient_TemperatureC', 'CurrentA_Cell_1', 'CurrentA_Cell_2', 'CurrentA_Cell_3', 'CurrentA_Cell_4'
                if(experiment[controlVarIndex]==i):
                    experiment_info = {
                        'Test_Times': experiment[5]['Data_processed']["Data"]['Test_Times'], 
                        'Step_Times': experiment[5]['Data_processed']["Data"]['Step_Times'],
                        'Cycle_Index': experiment[5]['Data_processed']["Data"]['Cycle_Index'],
                        'Step_Index': experiment[5]['Data_processed']["Data"]['Step_Index'],
                        'VoltageV' : experiment[5]['Data_processed']["Data"]['VoltageV'],
                        'CurrentA' : experiment[5]['Data_processed']["Data"]['CurrentA'],
                        'TemperatureC_Cell_1' : experiment[5]['Data_processed']["Data"]['TemperatureC_Cell_1'],
                        'TemperatureC_Cell_2' : experiment[5]['Data_processed']["Data"]['TemperatureC_Cell_2'],
                        'TemperatureC_Cell_3' : experiment[5]['Data_processed']["Data"]['TemperatureC_Cell_3'],
                        'TemperatureC_Cell_4' : experiment[5]['Data_processed']["Data"]['TemperatureC_Cell_4'],
                        'Ambient_TemperatureC' : experiment[5]['Data_processed']["Data"]['Ambient_TemperatureC'],
                        'CurrentA_Cell_1' : experiment[5]['Data_processed']["Data"]['CurrentA_Cell_1'],
                        'CurrentA_Cell_2' : experiment[5]['Data_processed']["Data"]['CurrentA_Cell_2'],
                        'CurrentA_Cell_3' : experiment[5]['Data_processed']["Data"]['CurrentA_Cell_3'],
                        'CurrentA_Cell_4' : experiment[5]['Data_processed']["Data"]['CurrentA_Cell_4'],
                    }
                    experiment_data.append(experiment_info)

            cell_dataset = pd.DataFrame(experiment_data)
            experiments_seperated.append(cell_dataset)
        return experiments_seperated
    
    @staticmethod
    def multienvcausaldiscoverydataset(data, cols, groups, sort_by='Step_Index'):

        # ---- Map StepIndex values → group number ----
        mapping = {}
        for i, group in enumerate(groups):
            for val in group:
                mapping[val] = i

        dfs = []  # collect all cell DataFrames here

        for idx, row in data.iterrows():
            cell_df = CausalIndividualLevel.make_cell_dataframe(row, cols)

            # map StepIndex → group index
            cell_df["Group"] = cell_df[sort_by].map(mapping)

            # drop original StepIndex
            cell_df = cell_df.drop(columns=[sort_by])

            dfs.append(cell_df)

        # build final dataset
        new_df = pd.concat(dfs, ignore_index=True)
        return new_df


    @staticmethod
    def multienvcausaldiscovery(data, save_point):
        os.makedirs(save_point, exist_ok=True)

        # --- Remove zero-variance columns ---
        data = data.loc[:, data.nunique() > 1]
        if "Group" in data.columns:
            data = data.drop(columns=["Group"])

        X = data.to_numpy().astype(float)
        cols = data.columns.tolist()

        # --- Run NoTears ---
        l1_reg = 0.01
        alpha = 0.05
        rho = 0.5
        no_tears_alg = NoTears(rho=rho, alpha=alpha, l1_reg=l1_reg, lr=2e-5)
        no_tears_alg.learn(X, n_outer_iter=30,  n_inner_iter=300)
        W_est = no_tears_alg.get_result()
        dag = (np.abs(W_est) > 0.3).astype(int)
        print(dag)

        G = nx.DiGraph()
        for i, src in enumerate(cols):
            for j, tgt in enumerate(cols):
                if dag[i, j] == 1:
                    G.add_edge(src, tgt)

        plt.figure(figsize=(7, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, arrows=True)
        plt.savefig(os.path.join(save_point, "dag.png"), dpi=200, bbox_inches="tight")
        plt.close()

    @staticmethod
    def multienvcausaldiscoverypergroup(data, save_point):
        os.makedirs(save_point, exist_ok=True)
        groups = data["Group"].unique()
        # --- Remove zero-variance columns ---
        for index, g in enumerate(groups):
            os.makedirs(f'{save_point}/{index}', exist_ok=True)
            subset = data[data["Group"] == g].reset_index(drop=True)
            #subset = subset.loc[:, subset.nunique() > 1]
            if "Group" in subset.columns:
                subset = subset.drop(columns=["Group"])

            X = subset.to_numpy().astype(float)
            cols = subset.columns.tolist()

            # --- Run NoTears ---
            l1_reg = 0.01
            alpha = 0.05
            rho = 0.5
            no_tears_alg = NoTears(rho=rho, alpha=alpha, l1_reg=l1_reg, lr=2e-5)
            no_tears_alg.learn(X, n_outer_iter=30,  n_inner_iter=300)
            W_est = no_tears_alg.get_result()
            dag = (np.abs(W_est) > 0.3).astype(int)
            print(dag)

            G = nx.DiGraph()
            for i, src in enumerate(cols):
                for j, tgt in enumerate(cols):
                    if dag[i, j] == 1:
                        G.add_edge(src, tgt)

            plt.figure(figsize=(7, 6))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, arrows=True)
            plt.savefig(os.path.join(f'{save_point}/{index}', "dag.png"), dpi=200, bbox_inches="tight")
            plt.close()

    @staticmethod
    def multienvcausaldiscoveryChat(data, save_point):
        os.makedirs(save_point, exist_ok=True)

        # --- Remove non-causal columns ---
        #if "Group" in data.columns:
            #data = data.drop(columns=["Group"])

        # --- Remove zero-variance columns ---
        data = data.loc[:, data.nunique() > 1]
        cols = data.columns.tolist()
        X = data.to_numpy().astype(float)
        # --- Run NoTears ---
        dagma_model = DagmaMLP(dims=[4, 16, 16, 4])  # input 4 → hidden → output 4
        model_nonlinear = DagmaNonlinear(dagma_model)
        tensor_data = torch.tensor(X, dtype=torch.float32)
        W_est = model_nonlinear.fit(tensor_data, lambda1 = 0.0005, lambda2=0.0001, lr=5e-4, warm_iter=1000, max_iter=3000)

        print("Learned adjacency matrix:\n", W_est)
        W = W_est.detach().cpu().numpy()
        dag = (np.abs(W) > 0.3).astype(int)

        print("Adjacency matrix:")
        print(dag)

        # --- Draw DAG ---
        G = nx.DiGraph()
        for i, src in enumerate(cols):
            for j, tgt in enumerate(cols):
                if dag[i, j] == 1:
                    G.add_edge(src, tgt)

        plt.figure(figsize=(7, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, arrows=True)
        plt.savefig(os.path.join(save_point, "ChatDag.png"), dpi=200, bbox_inches="tight")
        plt.close()
        


    @staticmethod
    def normalize_per_cell(list_of_cell_dfs):
        scaled_cells = []
        for df in list_of_cell_dfs:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df.values)
            scaled_df = pd.DataFrame(scaled, columns=df.columns)
            scaled_cells.append(scaled_df)
        return pd.concat(scaled_cells, ignore_index=True)
    
    @staticmethod
    def run_bigdata_notears(df, save_point):
        os.makedirs(save_point, exist_ok=True)
        if "Step_Index" in df.columns:
            df = df.drop(columns=["Step_Index"])
        
        X = df.to_numpy().astype(float)
        cols = df.columns.tolist()

        alg = NoTears(
            rho=0.5,        # weaker penalty for acyclicity
            alpha=0.05,     # smaller constraint
            l1_reg=0.01,    # allow edges to grow
            lr=1e-3,        # smaller learning rate
            #h_tol=1e-8,     # more precise DAG constraint
            #w_threshold=0.2 # cutoff for small weights
        )
        
        alg.learn(
            X,
            n_outer_iter=40,    # more iterations
            n_inner_iter=200,
            #grad_clip=5.0       # IMPORTANT: stabilizes large data
        )

        W = alg.get_result()
        dag = (np.abs(W) > 0.15).astype(int)   # lower threshold

        # Build graph
        G = nx.DiGraph()
        for i, src in enumerate(cols):
            for j, tgt in enumerate(cols):
                if dag[i, j] == 1:
                    G.add_edge(src, tgt)

        plt.figure(figsize=(7, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=2000)
        plt.savefig(os.path.join(save_point, "DAG_bigdata.png"))
        plt.close()

        return G, W

    @staticmethod
    def count_edge_frequencies(unique_dags, summary, node_names):
        """
        Count how often each directed edge appears across DAGs of potentially different sizes.
        node_names must be the full list of variable names in consistent order.
        """

        edge_counts = {}
        full_dim = len(node_names)

        for dag_hash, dag in unique_dags.items():
            dag_count = summary[dag_hash]
            dim = dag.shape[0]  # DAG size for this run

            for i in range(dim):
                for j in range(dim):
                    if dag[i, j] == 1:
                        edge = (node_names[i], node_names[j])
                        edge_counts[edge] = edge_counts.get(edge, 0) + dag_count

        return edge_counts


            