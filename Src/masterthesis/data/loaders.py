# data/loaders.py

from pathlib import Path
from itertools import product
import pandas as pd
from scipy.io import loadmat


# --------------------------------------------------
# Helper
# --------------------------------------------------

def load_mat(relative_path: str):
    """
    Safe loader that builds paths relative to project root.
    """
    root = Path(__file__).resolve().parents[3]
    file_path = root / relative_path

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return loadmat(file_path, struct_as_record=False, simplify_cells=True)


# ==================================================
# CELL-LEVEL DATA
# ==================================================

def load_cells():
    """
    Load all individual cell .mat files.
    Returns list of tuples: (cell_name, mat_dict)
    """
    base = (
        "Data/imbalanced performance/"
        "Full factorial design of experiments dataset for parallel-connected lithium-ion cells imbalanced performance investigation/"
        "Parallel-connected module experimental campaign/"
        "1_Single_cell_characterisation/"
    )

    cells = []

    # Specific aged cells
    cells.append(("GS3", load_mat(base + "Aged_cells/HPPC_MS/HPPC_MultiSine_GS3.mat")))
    #cells.append(("Y1",  load_mat(base + "Aged_cells/HPPC_MS/HPPC_MultiSine_Y1.mat")))

    # P2–P20
    for i in range(2, 21):
        cells.append((f"P{i}", load_mat(base + f"NMC_cells/HPPC_MS/HPPC_MultiSine_P{i}.mat")))

    # F1–F18
    for i in range(1, 19):
        cells.append((f"F{i}", load_mat(base + f"NCA_cells/HPPC_MS/HPPC_MultiSine_F{i}.mat")))

    return cells


def load_cell_dataframe():
    """
    Returns a pandas DataFrame where each row contains raw arrays for one cell.
    """
    cells = load_cells()
    cell_data = []

    for cell_name, cell in cells:
        T = len(cell['CurrentData'])
        for t in range(T):
            cell_data.append({
                "CellID": cell_name,
                "CurrentData": cell['CurrentData'][t],
                "VoltageData": cell['VoltageData'][t],
                "TempData": cell['TempData'][t],
                "StepIndex": cell['StepIndex'][t],
                "CycleIndex": cell['CycleIndex'][t],
                "TimeData": cell['TimeData'][t]
            })

    return pd.DataFrame(cell_data)

# ==================================================
# MODULE-LEVEL DATA
# ==================================================

def load_module_experiments():
    """
    Load all module-level experiment .mat files.
    Returns list of tuples with metadata.
    """
    base = (
        "Data/imbalanced performance/"
        "Full factorial design of experiments dataset for parallel-connected lithium-ion cells imbalanced performance investigation/"
        "Parallel-connected module experimental campaign/"
        "3_Module_level_experiments_fixed/Processed_data/"
    )

    cell_types = ["Mixed", "NCA", "NMC"]
    ages = ["Aged", "Unaged"]
    resistances = [0, 1, 3]
    temps = [10, 25, 40]
    machines = ["M1", "M2"]

    experiments = []

    for c, a, r, t, m in product(cell_types, ages, resistances, temps, machines):
        relative_path = base + f"/{c}_cells/" + a + f"/R_{r}/" + f"T_{t}/" + f"{m}_{c}_{a}_R{r}_T{t}.mat"
        root = Path(__file__).resolve().parents[3]
        file_path = root / relative_path

        if file_path.exists():
            data = load_mat(relative_path)
            experiments.append((c, a, r, t, m, data))

    return experiments


def load_module_dataframe():
    """
    Returns DataFrame where each row corresponds to one experiment.
    """
    experiments = load_module_experiments()
    cell_data = []

    for c, a, r, t, m, data in experiments:
            L = len(data['Data_processed']["Data"]['CurrentA'])
            for i in range(L):
                cell_data.append({
                    'Cell': c,
                    'aged': a,
                    'Resistance': r,
                    'Temp': t,
                    'CurrentA' : data['Data_processed']["Data"]['CurrentA'][i],
                    'TestTimes': data['Data_processed']["Data"]['Test_Times'][i], 
                    'StepTimes': data['Data_processed']["Data"]['Step_Times'][i],
                    'CycleIndex': data['Data_processed']["Data"]['Cycle_Index'][i],
                    'StepIndex': data['Data_processed']["Data"]['Step_Index'][i],
                    'VoltageV' : data['Data_processed']["Data"]['VoltageV'][i],                   
                    'TemperatureC_Cell_1' : data['Data_processed']["Data"]['TemperatureC_Cell_1'][i],
                    'TemperatureC_Cell_2' : data['Data_processed']["Data"]['TemperatureC_Cell_2'][i],
                    'TemperatureC_Cell_3' : data['Data_processed']["Data"]['TemperatureC_Cell_3'][i],
                    'TemperatureC_Cell_4' : data['Data_processed']["Data"]['TemperatureC_Cell_4'][i],
                    'Ambient_TemperatureC' : data['Data_processed']["Data"]['Ambient_TemperatureC'][i],
                    'CurrentA_Cell_1' : data['Data_processed']["Data"]['CurrentA_Cell_1'][i],
                    'CurrentA_Cell_2' : data['Data_processed']["Data"]['CurrentA_Cell_2'][i],
                    'CurrentA_Cell_3' : data['Data_processed']["Data"]['CurrentA_Cell_3'][i],
                    'CurrentA_Cell_4' : data['Data_processed']["Data"]['CurrentA_Cell_4'][i],
                })

    return pd.DataFrame(cell_data)
