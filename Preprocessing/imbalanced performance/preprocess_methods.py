import numpy as np

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
    
    