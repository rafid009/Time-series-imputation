from models.main_model import CSDI_Agaid
from datasets.dataset_agaid import get_dataloader
from utils.utils import *
import numpy as np
import torch
import sys
import os
from pypots.imputation import SAITS
from datasets.process_data import *
import pickle

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = np.random.randint(10, 100)

miss_type = 'random'

config_dict_csdi_pattern = {
    'train': {
        'epochs': 2500,
        'batch_size': 8,
        'lr': 1.0e-4
    },      
    'diffusion': {
        'layers': 4, 
        'channels': 64,
        'nheads': 8,
        'diffusion_embedding_dim': 128,
        'beta_start': 0.0001,
        'beta_end': 0.5,
        'num_steps': 70,
        'schedule': "quad",
        'is_fast': False,
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': miss_type,
        'type': 'CSDI',
        'n_layers': 3, 
        'd_time': 100,
        'n_feature': len(features),
        'd_model': 128,
        'd_inner': 128,
        'n_head': 8,
        'd_k': 64,
        'd_v': 64,
        'dropout': 0.1,
        'diagonal_attention_mask': True,
        'num_patterns': 15000,
        'num_val_patterns': 5000,
        'pattern_dir': './data/AgAid/miss_pattern'
    },
}

config_dict_csdi_random = {
    'train': {
        'epochs': 3000,
        'batch_size': 8,
        'lr': 1.0e-4
    },      
    'diffusion': {
        'layers': 4, 
        'channels': 64,
        'nheads': 8,
        'diffusion_embedding_dim': 128,
        'beta_start': 0.0001,
        'beta_end': 0.5,
        'num_steps': 70,
        'schedule': "quad",
        'is_fast': False,
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "mix",
        'type': 'CSDI',
        'n_layers': 3, 
        'd_time': 252,
        'n_feature': len(features),
        'd_model': 128,
        'd_inner': 128,
        'n_head': 4,
        'd_k': 64,
        'd_v': 64,
        'dropout': 0.1,
        'diagonal_attention_mask': False,
        'num_patterns': 15000,
        'num_val_patterns': 5000,
        'pattern_dir': './data/AgAid/miss_pattern'
    }
}

data_file = 'ColdHardiness_Grape_Merlot_2.csv'
config_dict_csdi = config_dict_csdi_pattern if miss_type.startswith('pattern') else config_dict_csdi_random

train_loader, valid_loader, mean, std = get_dataloader(
    seed=seed,
    filename=data_file,
    batch_size=config_dict_csdi["train"]["batch_size"],
    missing_ratio=0.2,
    season_idx=[32, 33]
)
# 

model_csdi = CSDI_Agaid(config_dict_csdi, device).to(device)
model_folder = "./saved_model_agaid"
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)
filename = f'model_csdi_{miss_type}.pth'
# train(
#     model_csdi,
#     config_dict_csdi["train"],
#     train_loader,
#     valid_loader=valid_loader,
#     foldername=model_folder,
#     filename=filename
# )
# nsample = 50
model_csdi.load_state_dict(torch.load(f"{model_folder}/{filename}"))
print(f"CSDI params: {get_num_params(model_csdi)}")
# evaluate(model_csdi, valid_loader, nsample=nsample, scaler=1, foldername=model_folder)
# model_folder_exp = "./saved_model_explode"
# if not os.path.isdir(model_folder_exp):
#     os.makedirs(model_folder_exp)
    
# config_dict_diffsaits = {
#     'train': {
#         'epochs': 5000,
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
#         'schedule': "quad"
#     },
#     'model': {
#         'is_unconditional': 0,
#         'timeemb': 128,
#         'featureemb': 16,
#         'target_strategy': "random",
#         'type': 'SAITS',
#         'n_layers': 3,
#         'loss_weight_p': 1,
#         'loss_weight_f': 1,
#         'd_time': 252,
#         'n_feature': len(features),
#         'd_model': 128,
#         'd_inner': 128,
#         'n_head': 8,
#         'd_k': 64,
#         'd_v': 64,
#         'dropout': 0.2,
#         'diagonal_attention_mask': True
#     }
# }
# print(config_dict_diffsaits)
# # model_diff_saits_simple = CSDI_Agaid(config_dict, device, is_simple=True).to(device)
# model_diff_saits = CSDI_Agaid(config_dict_diffsaits, device, is_simple=False).to(device)
# filename_simple = 'model_diff_saits_simple.pth'
# filename = 'model_diff_saits_final.pth'
# config_info = 'model_diff_saits_final.pth'

# model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))
# 
# train(
#     model_diff_saits,
#     config_dict_diffsaits["train"],
#     train_loader,
#     valid_loader=valid_loader,
#     foldername=model_folder,
#     filename=f"{filename}",
#     is_saits=True,
#     data_type='agaid'
# )
# nsample = 100
# model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))
# print(f"DiffSAITS params: {get_num_params(model_diff_saits)}")


# filename = "ColdHardiness_Grape_Merlot_2.csv"
# df = pd.read_csv(filename)
# modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)
# season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)
# train_season_df = season_df.drop(season_array[-1], axis=0)
# # train_season_df = train_season_df.drop(season_array[-2], axis=0)
# mean, std = get_mean_std(train_season_df, features)
# X, Y = split_XY(season_df, max_length, season_array, features)

# # # observed_mask = ~np.isnan(X)

# X = X[:-1]
# X = (X - mean) / std
# saits_model_file = f"{model_folder}/model_saits.pth"
# saits = SAITS(n_steps=252, n_features=len(features), n_layers=3, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=3000, patience=200, device=device)

# saits.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
# pickle.dump(saits, open(saits_model_file, 'wb'))

# saits = pickle.load(open(saits_model_file, 'rb'))

models = {
    'CSDI': model_csdi,
    # 'SAITS': saits,
    # 'DiffSAITS': model_diff_saits#,
    # 'DiffSAITSsimple': model_diff_saits_simple
}
mse_folder = f"results_agaid_qual_{miss_type}"
data_folder = f"results_data_agaid_qual_{miss_type}"

name = miss_type
test_patterns_start = 15001
num_test_patterns = 5000

test_pattern_config = {
    'start': test_patterns_start,
    'num_patterns': num_test_patterns,
    'pattern_dir': './data/AgAid/miss_pattern'
}

evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='agaid', batch_size=8, pattern=test_pattern_config, test_indices=[32,33], mean=mean, std=std)


# lengths = [ 50, 100, 200]

# for l in lengths:
#     print(f"length = {l}")
#     print(f"\nBlackout:\n")
#     evaluate_imputation_all(models=models, trials=20, mse_folder=mse_folder, dataset_name='agaid', batch_size=16, length=l)
#     # evaluate_imputation(models, data_folder, length=l, trials=1, data=True)
#     evaluate_imputation_all(models=models, mse_folder=data_folder, dataset_name='agaid', length=l, trials=1, batch_size=1, data=True)

print(f"\nForecasting:\n")
evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='agaid', batch_size=8, length=(30, 150), forecasting=True, test_indices=[32,33], mean=mean, std=std)
    # evaluate_imputation(models, mse_folder=data_folder, length=l, forecasting=True, trials=1, data=True)
# evaluate_imputation_all(models=models, mse_folder=data_folder, forecasting=True, dataset_name='agaid', length=100, trials=1, batch_size=1, data=True)

miss_ratios = [0.1, 0.5, 0.9]
for ratio in miss_ratios:
    print(f"\nRandom Missing: ratio ({ratio})\n")
    evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='agaid', batch_size=8, missing_ratio=ratio, random_trial=True, test_indices=[32,33], mean=mean, std=std)
    # evaluate_imputation(models, mse_folder=data_folder, random_trial=True, trials=1, data=True, missing_ratio=ratio)
    # evaluate_imputation_all(models=models, mse_folder=data_folder, dataset_name='agaid', trials=1, batch_size=1, data=True, missing_ratio=ratio, random_trial=True)


# print("For All")
# for l in lengths:
#     print(f"For length: {l}")
#     # evaluate_imputation(models, mse_folder, length=l, trials=1, season_idx=33)
#     print(f"Blackout Missing:\n")
#     evaluate_imputation(models, mse_folder, length=l, trials=10, season_idx=33)
#     evaluate_imputation(models, data_folder, length=l, trials=1, data=True)
#     print(f"Forecasting:\n")
#     evaluate_imputation(models, mse_folder, length=l, trials=1, season_idx=33, forecasting=True)
#     evaluate_imputation(models, mse_folder=data_folder, length=l, forecasting=True, trials=1, data=True)
#     print(f"Random Missing:\n")
#     evaluate_imputation(models, mse_folder, length=l, trials=10, season_idx=33, random_trial=True)
#     evaluate_imputation(models, mse_folder=data_folder, length=l, random_trial=True, trials=1, data=True)

    # evaluate_imputation_data(models, length=l)

# feature_combinations = {
    # "temp": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT"],
    # "hum": ["AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY", "MAX_REL_HUMIDITY"],
    # "dew": ["AVG_DEWPT", "MIN_DEWPT", "MAX_DEWPT"],
    # "pinch": ["P_INCHES"],
    # "wind": ["WS_MPH", "MAX_WS_MPH"],
    # "sr": ["SR_WM2"],q
    # "leaf": ["LW_UNITY"],
    # "et": ["ETO", "ETR"],
    # "st": ["ST8", "MIN_ST8", "MAX_ST8"],
#     "temp-hum": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT", "AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY",
#          "MAX_REL_HUMIDITY"],
#     "temp-hum-dew": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT", "AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY",
#          "MAX_REL_HUMIDITY", "AVG_DEWPT", "MIN_DEWPT", "MAX_DEWPT"],
#     "for-lte": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT", "AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY",
#          "MAX_REL_HUMIDITY", "AVG_DEWPT", "MIN_DEWPT", "MAX_DEWPT", "P_INCHES", "WS_MPH", "MAX_WS_MPH",
#          "SR_WM2", "LW_UNITY", "ETO", "ETR", "ST8", "MIN_ST8", "MAX_ST8"],
#     "for-temp": ["AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY", "MAX_REL_HUMIDITY", "AVG_DEWPT", "MIN_DEWPT",
#          "MAX_DEWPT", "P_INCHES", "WS_MPH", "MAX_WS_MPH",
#          "SR_WM2", "LW_UNITY", "ETO", "ETR", "ST8", "MIN_ST8", "MAX_ST8"],
#     "for-hum": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT", 
#          "AVG_DEWPT", "MIN_DEWPT", "MAX_DEWPT", "P_INCHES", "WS_MPH", "MAX_WS_MPH",
#          "SR_WM2", "LW_UNITY", "ETO", "ETR", "ST8", "MIN_ST8", "MAX_ST8"],
#     "for-dew": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT", "AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY",
#          "MAX_REL_HUMIDITY", "P_INCHES", "WS_MPH", "MAX_WS_MPH",
#          "SR_WM2", "LW_UNITY", "ETO", "ETR", "ST8", "MIN_ST8", "MAX_ST8"],
#     "for-et": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT", "AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY",
#          "MAX_REL_HUMIDITY", "AVG_DEWPT", "MIN_DEWPT", "MAX_DEWPT", "P_INCHES", "WS_MPH", "MAX_WS_MPH",
#          "SR_WM2", "LW_UNITY", "ST8", "MIN_ST8", "MAX_ST8"],
#     "for-sr": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT", "AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY",
#          "MAX_REL_HUMIDITY", "AVG_DEWPT", "MIN_DEWPT", "MAX_DEWPT", "P_INCHES", "WS_MPH", "MAX_WS_MPH",
#          "LW_UNITY", "ETO", "ETR", "ST8", "MIN_ST8", "MAX_ST8"]
# }
# print(f"The exclusions")
# for key in feature_combinations.keys():
#     for l in lengths:
#         print(f"For length: {l}")
#         evaluate_imputation(models, mse_folder, exclude_key=key, exclude_features=feature_combinations[key], length=l, trials=1)
#         evaluate_imputation(models, mse_folder, exclude_key=key, exclude_features=feature_combinations[key], length=l, trials=20)
        # evaluate_imputation_data(models, exclude_key=key, exclude_features=feature_combinations[key], length=l)
# forward_evaluation(models, filename, features)

# input_file = "ColdHardiness_Grape_Merlot_2.csv"

# cross_validate(input_file, config_dict_csdi, config_dict_diffsaits, seed=10)