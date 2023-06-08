from models.main_model import CSDI_AWN
from datasets.dataset_awn import get_dataloader
from utils.utils import train, get_num_params, evaluate_imputation_all
import numpy as np
import torch
import sys
import os
from pypots.imputation import SAITS
import matplotlib.pyplot as plt
import matplotlib
import pickle
from datasets.dataset_awn import get_dataloader
from datasets.preprocess_awn import features
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

given_features = features #['sin', 'cos2', 'harmonic', 'weight', 'lin_comb', 'non_lin_comb', 'mixed_history']

miss_type = 'random'
seed = 10
config_dict_csdi_pattern = {
    'train': {
        'epochs': 2000,
        'batch_size': 4,
        'lr': 1.0e-4
        # 'lr': 1.0e-4
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
        'target_strategy': 'pattern',
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
        'diagonal_attention_mask': True,
        'num_patterns': 15000,
        'num_val_patterns': 5000,
        'pattern_dir': './data/Daily/miss_pattern'
    },
}

config_dict_csdi_random = {
    'train': {
        'epochs': 3000,
        'batch_size': 4,
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
        'target_strategy': 'random',
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
        'diagonal_attention_mask': True,
        'num_patterns': 10000,
        'num_val_patterns': 5000,
        'pattern_dir': './data/Daily/miss_pattern'
    },
}

# filename: Any, is_year: bool = True, n_steps: int = 366

dataset_name = 'awn_daily_year'
data_file = './data/Daily/data_yy.npy'
nsample = 50

n_steps = 366
n_features = len(given_features)
test_season = 32
is_year = True
train_loader, valid_loader = get_dataloader(data_file, is_year=is_year, batch_size=4, test_index=test_season, missing_ratio=0.1, is_test=False, is_pattern=(miss_type == 'pattern'))

config_dict_csdi = config_dict_csdi_pattern if miss_type == 'pattern' else config_dict_csdi_random
model_csdi = CSDI_AWN(config_dict_csdi, device, target_dim=len(given_features)).to(device)
model_folder = f"./saved_model_{dataset_name}"
filename = f"model_csdi_{dataset_name}_{miss_type}.pth"
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)
print(f"\n\nCSDI training starts.....\n")
model_csdi.load_state_dict(torch.load(f"{model_folder}/{filename}"))
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

# config_dict_diffsaits = {
#     'train': {
#         'epochs':4000, # 3000 -> ds3
#         'batch_size': 16 ,
#         'lr': 1.0e-3
#     },      
#     'diffusion': {
#         'layers': 4, 
#         'channels': 64,
#         'nheads': 8,
#         'diffusion_embedding_dim': 128,
#         'beta_start': 0.0001,
#         'beta_end': 0.5,
#         'num_steps': 50,
#         'schedule': "quad",
#          'is_fast': False,
#     },
#     'model': {
#         'is_unconditional': 0,
#         'timeemb': 128,
#         'featureemb': 16,
#         'target_strategy': "mix", # noise mix
#         'type': 'SAITS',
#         'n_layers': 4,
#         'loss_weight_p': 1,
#         'loss_weight_f': 1,
#         'd_time': n_steps,
#         'n_feature': len(given_features),
#         'd_model': 128,
#         'd_inner': 128,
#         'n_head': 8,
#         'd_k': 64, #len(given_features),
#         'd_v': 64, #len(given_features),
#         'dropout': 0.1,
#         'diagonal_attention_mask': False,
#     },
#     'ablation': {
#         'fde-choice': 'fde-conv-multi',
#         'fde-layers': 4,
#         'is_fde': True,
#         'weight_combine': False,
#         'no-mask': False,
#         'fde-diagonal': True,
#         'is_fde_2nd': False,
#         'reduce-type': 'linear',
#         'is_2nd_block': True
#     }
# }
# print(f"config: {config_dict_diffsaits}")
# name = 'fde-conv-multi'
# model_diff_saits = CSDI_AWN(config_dict_diffsaits, device, target_dim=len(given_features)).to(device)

# filename = f"model_diffsaits_{dataset_name}_{name}_new.pth"
# print(f"\n\DiffSAITS training starts.....\n")

# model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))

# train(
#     model_diff_saits,
#     config_dict_diffsaits["train"],
#     train_loader,
#     valid_loader=valid_loader,
#     foldername=model_folder,
#     filename=f"{filename}",
#     is_saits=True
# )

# model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))
# print(f"DiffSAITS params: {get_num_params(model_diff_saits)}")

models = {
    'CSDI': model_csdi,
    # 'SAITS': saits,
    # 'DiffSAITS': model_diff_saits
}
# mse_folder = f"results_{dataset_name}_{name}_new/metric"
# data_folder = f"results_{dataset_name}_{name}_new/data"
name = miss_type
mse_folder = f"results_{dataset_name}_{name}/metric"
data_folder = f"results_{dataset_name}_{name}/data"

test_patterns_start = 15001
num_test_patterns = 5000

test_pattern_config = {
    'start': test_patterns_start,
    'num_patterns': num_test_patterns,
    'pattern_dir': './data/Daily/miss_pattern'
}

evaluate_imputation_all(models=models, trials=20, mse_folder=mse_folder, dataset_name='awn', batch_size=4, pattern=test_pattern_config, test_indices=test_season)
# lengths = [50, 100, 200]
# for l in lengths:
#     print(f"\nlength = {l}")
#     print(f"\nBlackout:")
#     evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='awn', batch_size=4, length=l, test_indices=test_season)
    # evaluate_imputation_all(models=models, mse_folder=data_folder, dataset_name='synth_v4', length=l, trials=1, batch_size=1, data=True)

print(f"\nForecasting:")
evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='awn', batch_size=4, length=(50, 200), forecasting=True, test_indices=test_season)
# evaluate_imputation_all(models=models, mse_folder=data_folder, forecasting=True, dataset_name='synth_v4', length=l, trials=1, batch_size=1, data=True)

miss_ratios = [0.1, 0.5, 0.9]
for ratio in miss_ratios:
    print(f"\nRandom Missing: ratio ({ratio})")
    evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='awn', batch_size=4, missing_ratio=ratio, random_trial=True, test_indices=test_season)
    # evaluate_imputation_all(models=models, mse_folder=data_folder, dataset_name='synth_v4', trials=1, batch_size=1, data=True, missing_ratio=ratio, random_trial=True)

print(f"new, not new_2")