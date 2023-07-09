import argparse
import torch
import datetime
import json
import yaml
import os
from datasets.dataset_pm25 import get_dataloader
from models.main_model import CSDI_PM25
from config_ablation import common_config
from utils.utils import train, evaluate_imputation_all, get_num_params
import pickle
import numpy as np
from pypots.imputation import SAITS


miss_pattern = 'random'
args = {
    'config': 'base.yaml',
    'device': 'cuda:0',
    'seed': 1,
    'testmissingratio': 0.1,
    'nfold': 0,
    'unconditional': False,
    'modelfolder': f'saved_model_pm25_{miss_pattern}',
    'nsample': 50
}
print(args)

path = "config/" + args["config"]
with open(path, "r") as f:
    config = yaml.safe_load(f)

# train:
#   epochs: 2000
#   batch_size: 16 
#   lr: 5.0e-3

# diffusion:
#   layers: 4 
#   channels: 64 
#   nheads: 8
#   diffusion_embedding_dim: 128
#   beta_start: 0.0001
#   beta_end: 0.5
#   num_steps: 50
#   schedule: "quad"

# model:
#   is_unconditional: 0
#   timeemb: 128
#   featureemb: 16
#   target_strategy: "random"
config["train"]["lr"] = 1.0e-4
config["model"]["is_unconditional"] = args["unconditional"]
config["model"]["test_missing_ratio"] = args["testmissingratio"]
config["model"]['target_strategy'] = 'mix' if miss_pattern == 'random' else 'pattern'
args['validationindex'] = 0
print(f"config_csdi:\n")
print(json.dumps(config, indent=4))

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    config["train"]["batch_size"], device=args['device'], validindex=args['validationindex']
)

config['model']['type'] = 'CSDI'
config['diffusion']['is_fast'] = False
# config['model']['num_patterns'] = 20000
# config['model']['num_val_patterns'] = 5000
# config['model']['pattern_dir'] = './data/pm25/miss_patterns'

model_csdi = CSDI_PM25(config, args['device']).to(args['device'])

model_folder = "saved_model_pm25"
filename = "model_csdi_pm25.pth"
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)

train(
    model_csdi,
    config["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=model_folder,
    filename=filename
)
# model_csdi.load_state_dict(torch.load(f"{model_folder}/{filename}"))
config_dict_diffsaits = {
    'train': {
        'epochs': 3500,
        'batch_size': 16 ,
        'lr': 1.0e-4
    },      
    'diffusion': {
        'layers': 4, 
        'channels': 64,
        'nheads': 8,
        'diffusion_embedding_dim': 128,
        'beta_start': 0.0001,
        'beta_end': 0.7,
        'num_steps': 50,
        'schedule': "quad"
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "mix",
        'type': 'SAITS',
        'n_layers': 3,
        'loss_weight_p': 0.5,
        'loss_weight_f': 1,
        'd_time': 36,
        'n_feature': 36, #len(attributes),
        'd_model': 128,
        'd_inner': 128,
        'n_head': 8,
        'd_k': 64,
        'd_v': 64,
        'dropout': 0.1,
        'diagonal_attention_mask': False
    }
}
config_dict_diffsaits['ablation'] = common_config['ablation']
config_dict_diffsaits['model']['n_layers'] = common_config['n_layers']
config_dict_diffsaits['name'] = common_config['name']
name = config_dict_diffsaits['name']
print(f"config: {config_dict_diffsaits}")
model_diff_saits = CSDI_PM25(config_dict_diffsaits, args['device'], is_simple=False).to(args['device'])
# filename_simple = 'model_diff_saits_simple.pth'
filename = f'model_diff_saits_pm25_{name}.pth'

# # model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))
# # 
train(
    model_diff_saits,
    config_dict_diffsaits["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=model_folder,
    filename=f"{filename}",
    is_saits=True,
    data_type='pm25'
)
# nsample = 100
# model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))
# print(f"DiffSAITS params: {get_num_params(model_diff_saits)}")

# saits_model_file = f"{model_folder}/model_saits_pm25.pth" # don't change it
# saits = SAITS(n_steps=36, n_features=36, n_layers=3, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=2000, patience=200, device=args['device'])

# X = []
# masks = []
# for j, train_batch in enumerate(train_loader, start=1):
#     observed_data, observed_mask, _, _, _, _, _, _ = model_diff_saits.process_data(train_batch)
#     observed_data = observed_data.permute(0, 2, 1)
#     observed_mask = observed_mask.permute(0, 2, 1)
#     if isinstance(observed_data, torch.Tensor):
#         X.append(observed_data.detach().cpu().numpy())
#         masks.append(observed_mask.detach().cpu().numpy())
#     elif isinstance(observed_data, list):
#         X.append(np.asarray(observed_data))
#         masks.append(np.asarray(observed_mask))
#     else:
#         X.append(observed_data)
#         masks.append(observed_mask)
    
# X = np.concatenate(X, axis=0)
# masks = np.concatenate(masks, axis=0)
# masks = np.ma.make_mask(masks, copy=True, shrink=False)
# shp = X.shape
# print(f"X shape: {shp}")
# X = X.reshape(-1).copy()
# masks = masks.reshape(-1)
# masks = ~masks
# X[masks] = np.nan
# X = X.reshape(shp)
# saits.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
# pickle.dump(saits, open(saits_model_file, 'wb'))
# saits = pickle.load(open(saits_model_file, 'rb'))
models = {
    'CSDI': model_csdi,
    # 'SAITS': saits,
    'DiffSAITS': model_diff_saits
}
mse_folder = f"results_pm25_{name}/metric"
data_folder = f"results_pm25_{name}/data"

test_patterns_start = 25001
num_test_patterns = 5000

test_pattern_config = {
    'start': test_patterns_start,
    'num_patterns': num_test_patterns,
    'pattern_dir': config['model']['pattern_dir']
}

evaluate_imputation_all(models=models, trials=5, mse_folder=mse_folder, dataset_name='pm25', batch_size=32, test_indices=test_loader, pattern=test_pattern_config)
evaluate_imputation_all(models=models, trials=5, mse_folder=mse_folder, dataset_name='pm25', batch_size=32, test_indices=test_loader)

# lengths = [10, 20, 30]
# for l in lengths:
#     print(f"\nlength = {l}")
#     print(f"\nBlackout:")
#     evaluate_imputation_all(models=models, trials=3, mse_folder=mse_folder, dataset_name='physio', batch_size=32, length=l, test_indices=test_indices)

# print(f"\nForecasting:")
# evaluate_imputation_all(models=models, trials=3, mse_folder=mse_folder, dataset_name='physio', batch_size=32, length=(10, 30), forecasting=True, test_indices=test_indices)

# miss_ratios = [0.1, 0.5, 0.9]
# for ratio in miss_ratios:
#     print(f"\nRandom Missing: ratio ({ratio})")
#     evaluate_imputation_all(models=models, trials=3, mse_folder=mse_folder, dataset_name='physio', batch_size=32, missing_ratio=ratio, random_trial=True, test_indices=test_indices)