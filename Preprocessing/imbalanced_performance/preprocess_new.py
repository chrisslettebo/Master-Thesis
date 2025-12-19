import numpy as np
import hashlib
import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from pathlib import Path
from itertools import product

class IndividualProcess:
    @staticmethod
    def getCells():
        base = "Data/imbalanced performance/" \
               "Full factorial design of experiments dataset for parallel-connected lithium-ion cells imbalanced performance investigation/" \
               "Parallel-connected module experimental campaign/" \
               "1_Single_cell_characterisation/"

        cells = []

        # --- Specific files ---
        cells.append(["GS3", CommonProcesses.load_mat(base + "Aged_cells/HPPC_MS/HPPC_MultiSine_GS3.mat")])
        cells.append(["Y1",  CommonProcesses.load_mat(base + "Aged_cells/HPPC_MS/HPPC_MultiSine_Y1.mat")])

        # --- Load P2–P20 ---
        for i in range(2, 21):
            cells.append([f"P{i}", CommonProcesses.load_mat(base + f"NMC_cells/HPPC_MS/HPPC_MultiSine_P{i}.mat")])

        # --- Load F1–F18 ---
        for i in range(1, 19):
            cells.append([f"F{i}", CommonProcesses.load_mat(base + f"NCA_cells/HPPC_MS/HPPC_MultiSine_F{i}.mat")])

        return cells
    
    @staticmethod
    def getCellDataset():
        cell_data = []
        cells = IndividualProcess.getCells()
        for cell in cells:
            cell_info = {
                'CurrentData': cell[1]['CurrentData'], 
                'VoltageData': cell[1]['VoltageData'],
                'TempData': cell[1]['TempData'],
                'Step_Index': cell[1]['StepIndex'],
                'CycleIndex' : cell[1]['CycleIndex'],
                'TimeData' : cell[1]['TimeData']
            }
            cell_data.append(cell_info)

        cell_dataset = pd.DataFrame(cell_data)
        return cell_dataset


class CommonProcesses:

    @staticmethod
    def flatten_array(arr):
    # Works for arrays like [[x], [y], [z]]
        return np.array([x[0] if isinstance(x, (list, np.ndarray)) else x for x in arr])

    @staticmethod
    def make_cell_dataframe(row, cols):
        clean = {}

        # Flatten all nested arrays
        for name in cols:
            clean[name] = CommonProcesses.flatten_array(row[name])

        # Build long-format dataframe
        return pd.DataFrame(clean)

    @staticmethod
    def load_mat(relative_path: str):
        """
        Safe loader that builds paths relative to project root.
        Example usage:
            DataLoader.load_mat("Data/.../HPPC_MultiSine_GS3.mat")
        """
        root = Path(__file__).resolve().parents[2]   # <-- adjust if needed
        file_path = root / relative_path

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return loadmat(file_path, struct_as_record=False, simplify_cells=True)

    @staticmethod
    def splitDataForEachByStep(data, cols, groups):
        cell_list = []
        for idx, row in data.iterrows():
            #cell_info = {name : row[name] for name in cols}
            #new_pd = pd.DataFrame([cell_info])
            cell_df = CommonProcesses.make_cell_dataframe(row, cols)
            obs_datasets = []
            interventional_datasets = []
            for i in range(len(groups)):
                subset = cell_df[cell_df["Step_Index"].isin(groups[i])].reset_index(drop=True)

                if i == 0:
                    obs_datasets.append(subset)

                interventional_datasets.append(subset)
            cell_list.append([obs_datasets, interventional_datasets])
        return cell_list
    
    @staticmethod
    def multienvcausaldiscoverydataset(data, cols, groups, sort_by='Step_Index'):

        # ---- Map StepIndex values → group number ----
        mapping = {}
        for i, group in enumerate(groups):
            for val in group:
                mapping[val] = i

        dfs = []  # collect all cell DataFrames here
        null_info = []  # collect info about rows where mapping failed

        for idx, row in data.iterrows():
            cell_df = CommonProcesses.make_cell_dataframe(row, cols)

            # map StepIndex → group index
            cell_df["Group"] = cell_df[sort_by].map(mapping)

                # check for nulls created by mapping
            if cell_df["Group"].isnull().any():
                null_rows = cell_df[cell_df["Group"].isnull()]
                null_info.append((idx, null_rows))  # store original Step_Index values causing NaN

            # drop original StepIndex
            cell_df = cell_df.drop(columns=[sort_by])

            dfs.append(cell_df)

        # build final dataset
        new_df = pd.concat(dfs, ignore_index=True)


        if null_info:
            print("Warning: Some rows could not be mapped to groups:")
            for row_idx, missing_values in null_info:
                print(f"Original data row {row_idx} has unmapped {sort_by} values:\n{missing_values}")
        return new_df


class ModularProcesses:
    
    @staticmethod
    def getExperiments():
        base = Path("Data/imbalanced performance/") / \
               "Full factorial design of experiments dataset for parallel-connected lithium-ion cells imbalanced performance investigation/" / \
               "Parallel-connected module experimental campaign/" / \
               "3_Module_level_experiments_fixed/Processed_data/"

        cell_paths = ['Mixed', 'NCA', 'NMC']
        is_aged = ['Aged', 'Unaged']
        Res = [0, 1, 3]
        Temp = [10, 25, 40]
        machines = ['M1', 'M2']

        experiments = []

        for c, a, r, t, m in product(cell_paths, is_aged, Res, Temp, machines):
            relative_path = base / f"{c}_cells" / a / f"R_{r}" / f"T_{t}" / f"{m}_{c}_{a}_R{r}_T{t}.mat"
            root = Path(__file__).resolve().parents[2]   # <-- adjust if needed
            file_path = root / relative_path
            if file_path.exists():
                data = CommonProcesses.load_mat(relative_path)
                experiments.append([c, a, r, t, m, data])

        return experiments

    @staticmethod
    def getExperimentDataSet():
        experiment_data = []
        experiments = ModularProcesses.getExperiments()
        for experiment in experiments:
            #'Test_Times', 'Step_Times', 'Step_Index', 'Cycle_Index', 'VoltageV', 'CurrentA', 'TemperatureC_Cell_1', 'TemperatureC_Cell_2', 'TemperatureC_Cell_3', 'TemperatureC_Cell_4', 
            # 'Ambient_TemperatureC', 'CurrentA_Cell_1', 'CurrentA_Cell_2', 'CurrentA_Cell_3', 'CurrentA_Cell_4'
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
        return cell_dataset
    
    @staticmethod
    def getExperimentDataSetExtraInfo():
        experiment_data = []
        experiments = ModularProcesses.getExperiments()
        for experiment in experiments:
            #'Test_Times', 'Step_Times', 'Step_Index', 'Cycle_Index', 'VoltageV', 'CurrentA', 'TemperatureC_Cell_1', 'TemperatureC_Cell_2', 'TemperatureC_Cell_3', 'TemperatureC_Cell_4', 
            # 'Ambient_TemperatureC', 'CurrentA_Cell_1', 'CurrentA_Cell_2', 'CurrentA_Cell_3', 'CurrentA_Cell_4'
            data = experiment[5]['Data_processed']["Data"]
            n = len(data['Test_Times'])  # number of rows per experiment

            experiment_info = {
                'Type_Cell':      [experiment[0]] * n,
                'Age':            [experiment[1]] * n,
                'Resistance':     [experiment[2]] * n,
                'Temperature':    [experiment[3]] * n,

                'Test_Times':     data['Test_Times'],
                'Step_Times':     data['Step_Times'],
                'Cycle_Index':    data['Cycle_Index'],
                'Step_Index':     data['Step_Index'],
                'VoltageV':       data['VoltageV'],
                'CurrentA':       data['CurrentA'],
                'TemperatureC_Cell_1': data['TemperatureC_Cell_1'],
                'TemperatureC_Cell_2': data['TemperatureC_Cell_2'],
                'TemperatureC_Cell_3': data['TemperatureC_Cell_3'],
                'TemperatureC_Cell_4': data['TemperatureC_Cell_4'],
                'Ambient_TemperatureC': data['Ambient_TemperatureC'],
                'CurrentA_Cell_1': data['CurrentA_Cell_1'],
                'CurrentA_Cell_2': data['CurrentA_Cell_2'],
                'CurrentA_Cell_3': data['CurrentA_Cell_3'],
                'CurrentA_Cell_4': data['CurrentA_Cell_4'],
            }
            experiment_data.append(experiment_info)

        cell_dataset = pd.DataFrame(experiment_data)
        return cell_dataset
    