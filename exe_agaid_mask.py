from models.mask_main_model import Mask_Agaid
from datasets.dataset_agaid_mask import get_dataloader
from utils.utils import train, get_num_params, calc_quantile_CRPS, evaluate_imputation_all, clip_pattern_mask
import numpy as np
import torch
import sys
import os
from pypots.imputation import SAITS
import matplotlib.pyplot as plt
import matplotlib
from datasets.process_data import features
import pickle
import json
from json import JSONEncoder
import math
from tqdm import tqdm
import xskillscore as xs
import xarray as xr
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
# torch.manual_seed(42)
np.set_printoptions(threshold=np.inf)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
   

seed = np.random.randint(10, 100)
n_features = len(features)
n_sample = 100
d_time = 252

config_dict_csdi = {
    'train': {
        'epochs': 500,
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
        'is_unconditional': True,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "random",
        'type': 'CSDI',
        'n_layers': 3, 
        'd_time': d_time,
        'n_feature': len(features),
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

n_steps = d_time

data_file = 'ColdHardiness_Grape_Merlot_2.csv'

train_loader, valid_loader, mean, std = get_dataloader(
    seed=seed,
    filename=data_file,
    batch_size=config_dict_csdi["train"]["batch_size"],
    missing_ratio=0.01,
    season_idx=[32, 33]
)

model_csdi = Mask_Agaid(config_dict_csdi, device, target_dim=len(features)).to(device)
model_folder = "./saved_model_mask_agaid"
filename = f"model_csdi_mask_agaid.pth"
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)
print(f"\n\nCSDI Masked training starts.....\n")
# model_csdi.load_state_dict(torch.load(f"{model_folder}/{filename}"))
train(
    model_csdi,
    config_dict_csdi["train"],
    train_loader,
    valid_loader=None,
    foldername=model_folder,
    filename=f"{filename}",
    is_saits=False
)
# model_csdi.load_state_dict(torch.load(f"{model_folder}/{filename}"))
print(f"CSDI params: {get_num_params(model_csdi)}")


def quantile_loss(target, forecast, q: float) -> float:
    # print(f"in quant: target: {target.shape}, forecast: {forecast.shape}")
    return 2 * torch.sum(
        torch.abs((forecast - target) * ((target <= forecast) * 1.0 - q))
    )

def calc_denominator(target):
    return torch.sum(torch.abs(target))

def calc_quantile_CRPS(target, forecast, mean_scaler, scaler):
    # print(f"target: {target.shape}\nforecast: {forecast.shape}")
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i])
        # print(f"q_loss: {q_loss}, denom: {denom}")
        CRPS += q_loss / denom
        # print(f"CRPS each qunatile: {CRPS}")
    return CRPS.item() / len(quantiles)


nsample = 20000 # 3000 * 4 * 8
ground = 0
for i, val in enumerate(valid_loader):
    ground = val['observed_mask'].to(device).float() # (B, L, K)
    # ground = ground.reshape(ground.shape[0], -1).cpu().numpy()

sample_folder = './data/AgAid/miss_pattern'

if not os.path.isdir(sample_folder):
    os.makedirs(sample_folder)

with torch.no_grad():
    output = model_csdi.evaluate(nsample, (1, n_features, d_time))
    samples = output

    samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
    # samples = samples.reshape(samples.shape[0], samples.shape[1], -1).cpu().numpy()
    # samples = (samples > 0).float()
    # print(f"sample 1: {samples[0][1].cpu().numpy()}")
    # print(f"sample 2: {samples[0][2].cpu().numpy()}")
    # print(f"sample 3: {samples[0][3].cpu().numpy()}")
    save_samples = clip_pattern_mask(samples.cpu().numpy())
    save_samples = save_samples.squeeze(0)
    # print(f"sample 1: {save_samples[1].cpu().numpy()}")
    # print(f"sample 2: {save_samples[2].cpu().numpy()}")
    # print(f"sample 3: {save_samples[3].cpu().numpy()}")
    for i in range(save_samples.shape[0]):
        print(f"samples: {samples[0][i].cpu().numpy()}")
        print(f"save samples: {save_samples[i]}")
        np.save(f"{sample_folder}/pattern_{i}.npy", save_samples[i])

    crps_avg = 0
    num = 0
    for i in range(len(ground)):
        crps = calc_quantile_CRPS(ground[i].unsqueeze(0), samples, 0, 1)
        print(f"CRPS for {i} : {crps}")
        crps_avg += crps
        num += 1
    print(f"final CRPS: {crps_avg / num}")
    

    # forecasts = xr.DataArray(samples, coords=[('member', np.arange(samples.shape[0])), ('b', np.arange(1)), ('x', np.arange(n_features * d_time))])
    # forecast_mean = np.mean(samples, axis=0)
    # forcast_std = np.std(samples, axis=0)
    # forecasts = st.norm.cdf(samples, ground.shape[0], ground.shape[1], )
    # brier = xs.brier_score(observations, (forecasts).mean('member'), dim="b")
    # crps = xs.crps_ensemble(observations, forecasts, member_dim='member', dim='b')

    # print(f"brier: {brier}\ncrps: {crps}")
    # print(f"CRPS: {crps}")




    


