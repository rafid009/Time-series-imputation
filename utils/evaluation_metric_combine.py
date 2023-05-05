import numpy as np
import pandas as pd
import os
import json

v = 'v2'
data_name = 'agaid'#f"synth{'_' + v if v != '' else ''}"
input_folder = f"results_{data_name}_qual"
metric = 'crps'
correction = 1
def get_std(data):
    return np.std(data, ddof=1) / np.sqrt(np.size(data))

if metric == 'crps':
    input_dir = f"{input_folder}/crps"
    print(f"\nCRPS in {data_name}\n")
    for file_name in os.listdir(input_dir):
        mode = None
        if 'forecasting-True' in file_name:
            mode = 'forecast'
        elif 'blackout-True' in file_name:
            mode = 'blackout'
        else:
            mode = 'random'
        measure = -1
        if mode == 'forecast':
            measure = 0
        elif mode == 'random':
            start = file_name.index('miss_') + len('miss_')
            end = start + 3
            measure = float(file_name[start:end])
        else:
            start = file_name.index('l_') + len('l_')
            end = file_name.index('_miss_')
            measure = float(file_name[start:end])
        fp = open(f"{input_dir}/{file_name}", "r")
        content = json.load(fp)
        fp.close()
        mean_csdi = np.round(content['csdi'] / correction, 4)
        mean_dfs = np.round(content['diffsaits'], 4)
        trials_csdi = content['csdi_trials']
        trials_dfs = content['diffsaits_trials']

        data = np.array([i for i in trials_csdi.values()]) / correction
        std_csdi = np.round(get_std(data), 5)

        data = np.array([i for i in trials_dfs.values()])
        std_dfs = np.round(get_std(data), 5)

        print(f"Mode: {mode}\n\tMeasure: {measure}\n\t\tCSDI: {mean_csdi}({std_csdi})\
              \n\t\tOur Model: {mean_dfs}({std_dfs})\n\n")
else:
    mean_folder, trials_folder = f"{input_folder}/{metric}-mean", f"{input_folder}/{metric}-trials"
    for file_name in os.listdir(mean_folder):
        mode = None
        if 'forecasting-True' in file_name:
            mode = 'forecast'
        elif 'blackout-True' in file_name:
            mode = 'blackout'
        else:
            mode = 'random'
        measure = -1
        if mode == 'forecast':
            measure = 0
        elif mode == 'random':
            start = file_name.index('miss_') + len('miss_')
            end = start + 3
            measure = float(file_name[start:end])
        else:
            start = file_name.index('l_') + len('l_')
            end = file_name.index('_miss_')
            measure = float(file_name[start:end])
        l = len(file_name)
        trial_file_name = file_name[0:4]+'trials'+'-'+file_name[4:]
        fp_mean = open(f"{mean_folder}/{file_name}", "r")
        mean_info = json.load(fp_mean)
        fp_mean.close()
        fp_trial = open(f"{trials_folder}/{trial_file_name}", "r")
        std_info = json.load(fp_trial)
        fp_trial.close()

        mean_csdi = np.round(mean_info['csdi'], 5)
        mean_diffsaits = np.round(mean_info['diffsaits'], 5)
        mean_saits = np.round(mean_info['saits'], 5)

        trials_csdi = std_info['csdi']
        data = np.array([i for i in trials_csdi.values()])
        std_csdi = np.round(get_std(data), 5)

        trials_diffsaits = std_info['diffsaits']
        data = np.array([i for i in trials_diffsaits.values()])
        std_diffsaits = np.round(get_std(data), 5)

        trials_saits = std_info['saits']
        data = np.array([i for i in trials_saits.values()])
        std_saits = np.round(get_std(data), 5)

        print(f"Mode: {mode}\n\tMeasure: {measure}\n\t\tCSDI: {mean_csdi}({std_csdi})\
              \n\t\tOur Model: {mean_diffsaits}({std_diffsaits})\n\t\tSAITS: {mean_saits}({std_saits})\n\n")