from models.main_model import CSDI_Synth
from datasets.dataset_synth import get_dataloader, get_testloader
from utils.utils import train, get_num_params, calc_quantile_CRPS, evaluate_imputation_all
import numpy as np
import torch
import sys
import os
from pypots.imputation import SAITS
import matplotlib.pyplot as plt
import matplotlib
import pickle
from datasets.synthetic_data import create_synthetic_data_v2, feats_v2
import json
from json import JSONEncoder
import math
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
# torch.manual_seed(42)
np.set_printoptions(threshold=sys.maxsize)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

given_features = feats_v2


seed = 10
config_dict_csdi = {
    'train': {
        'epochs': 2500,
        'batch_size': 16 ,
        'lr': 1.0e-3
    },      
    'diffusion': {
        'layers': 4, 
        'channels': 64,
        'nheads': 8,
        'diffusion_embedding_dim': 128,
        'beta_start': 0.0001,
        'beta_end': 0.5,
        'num_steps': 50,
        'schedule': "quad",
         'is_fast': False,
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "random",
        'type': 'CSDI',
        'n_layers': 3, 
        'd_time': 100,
        'n_feature': len(given_features),
        'd_model': 128,
        'd_inner': 128,
        'n_head': 8,
        'd_k': 64,
        'd_v': 64,
        'dropout': 0.1,
        'diagonal_attention_mask': True
    },
}

nsample = 50

n_steps = 100
n_features = len(given_features)
num_seasons = 32
noise = False
train_loader, valid_loader = get_dataloader(n_steps, n_features, num_seasons, batch_size=16, missing_ratio=0.1, seed=10, is_test=False, v2='v2', noise=noise)

model_csdi = CSDI_Synth(config_dict_csdi, device, target_dim=len(given_features)).to(device)
model_folder = "./saved_model_synth_v2"
filename = f"model_csdi_synth_v2.pth"
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)
print(f"\n\nCSDI training starts.....\n")
# train(
#     model_csdi,
#     config_dict_csdi["train"],
#     train_loader,
#     valid_loader=valid_loader,
#     foldername=model_folder,
#     filename=f"{filename}",
#     is_saits=False
# )
# model_csdi.load_state_dict(torch.load(f"{model_folder}/{filename}"))
print(f"CSDI params: {get_num_params(model_csdi)}")


# saits_model_file = f"{model_folder}/saits_model_synth_v4{'_noise' if noise else ''}.pkl"
# saits = SAITS(n_steps=n_steps, n_features=n_features, n_layers=3, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=3000, patience=400, device=device)
# X, mean, std = create_synthetic_data_v4(n_steps, num_seasons, seed=10, noise=noise)
# print(f"\n\SAITS training starts.....\n")
# saits.fit(X)
# pickle.dump(saits, open(saits_model_file, 'wb'))


# saits = pickle.load(open(saits_model_file, 'rb'))

config_dict_diffsaits = {
    'train': {
        'epochs':5000, # 3000 -> ds3
        'batch_size': 16 ,
        'lr': 5.0e-5
    },      
    'diffusion': {
        'layers': 4, 
        'channels': 64,
        'nheads': 8,
        'diffusion_embedding_dim': 128,
        'beta_start': 0.0001,
        'beta_end': 0.5,
        'num_steps': 50,
        'schedule': "quad",
         'is_fast': False,
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "mix", # noise mix
        'type': 'SAITS',
        'n_layers': 6,
        'loss_weight_p': 1,
        'loss_weight_f': 1,
        'd_time': n_steps,
        'n_feature': len(given_features),
        'd_model': 128,
        'd_inner': 128,
        'n_head': 8,
        'd_k': 128, # 64, #len(given_features),
        'd_v': 128, # 64, #len(given_features),
        'dropout': 0.1,
        'diagonal_attention_mask': False
    },
    'ablation': {
        'fde-choice': 'fde-conv-multi',
        'fde-layers': 4,
        'is_fde': False,
        'weight_combine': False,
        'no-mask': False,
        'fde-diagonal': False,
        'is_fde_2nd': False,
        'reduce-type': 'linear',
        'is_2nd_block': False
    }
}
print(f"config: {config_dict_diffsaits}")
name = 'fde-conv-multi'
model_diff_saits = CSDI_Synth(config_dict_diffsaits, device, target_dim=len(given_features)).to(device)

filename = f"model_diffsaits_synth_v2_{name}_new_2.pth"
print(f"\n\DiffSAITS training starts.....\n")

# model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))

train(
    model_diff_saits,
    config_dict_diffsaits["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=model_folder,
    filename=f"{filename}",
    is_saits=True
)

# model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))
print(f"DiffSAITS params: {get_num_params(model_diff_saits)}")

models = {
    # 'CSDI': model_csdi,
    # 'SAITS': saits,
    'DiffSAITS': model_diff_saits
}
mse_folder = f"results_synth_v2_{name}_new_2/metric"
data_folder = f"results_synth_v2_{name}_new_2/data"
lengths = [10, 50, 90]
for l in lengths:
    print(f"\nlength = {l}")
    print(f"\nBlackout:")
    evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='synth_v2', batch_size=32, length=l, noise=noise)
    # evaluate_imputation_all(models=models, mse_folder=data_folder, dataset_name='synth_v4', length=l, trials=1, batch_size=1, data=True, noise=noise)

print(f"\nForecasting:")
evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='synth_v2', batch_size=32, length=(10, 80), forecasting=True, noise=noise)
# evaluate_imputation_all(models=models, mse_folder=data_folder, forecasting=True, dataset_name='synth_v4', length=l, trials=1, batch_size=1, data=True, noise=noise)

miss_ratios = [0.1, 0.5, 0.9]
for ratio in miss_ratios:
    print(f"\nRandom Missing: ratio ({ratio})")
    evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='synth_v2', batch_size=32, missing_ratio=ratio, random_trial=True, noise=noise)
    # evaluate_imputation_all(models=models, mse_folder=data_folder, dataset_name='synth_v4', trials=1, batch_size=1, data=True, missing_ratio=ratio, random_trial=True, noise=noise)
