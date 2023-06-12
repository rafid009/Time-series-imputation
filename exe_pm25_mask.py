import argparse
import torch
import datetime
import json
import yaml
import os
from pypots.imputation import SAITS
import pickle
import numpy as np
import time
np.set_printoptions(threshold=np.inf)
from models.mask_main_model import Mask_PM25
from datasets.dataset_pm25_mask import get_dataloader
from utils.utils import train

def quantile_loss(target, forecast, q: float) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target):
    return torch.sum(torch.abs(target))


def calc_quantile_CRPS(target, forecast, mean_scaler, scaler):
    # print(f"target: {target.shape}\nforecast: {forecast.shape}")
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    # print(f"target: {target}")
    # print(f"forecasts: {forecast[0:10]}")

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

d_time = 36
args = {
    'train': {
        'epochs': 300,
        'batch_size': 16,
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
        'is_unconditional': True,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "mix",
        'type': 'CSDI',
        'n_layers': 3, 
        'd_time': d_time,
        'n_feature': 36,#len(attributes),
        'd_model': 128,
        'd_inner': 128,
        'n_head': 8,
        'd_k': 64,
        'd_v': 64,
        'dropout': 0.1,
        'diagonal_attention_mask': True,
        "test_missing_ratio": 0.1
    }
}

print(f"config: {args}")
args['validationindex'] = 0
args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, valid_loader, test_loader, std, mean = get_dataloader(
    args["train"]["batch_size"], device=args['device'], validindex=args['validationindex']
)
args['model']['type'] = 'CSDI'
args['diffusion']['is_fast'] = False
args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_csdi = Mask_PM25(args, args['device']).to(args['device'])
model_folder = "saved_model_pm25_mask"
filename = "model_csdi_pm25_mask.pth"
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)
train(
    model_csdi,
    args["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=model_folder,
    filename=filename
)

model_csdi.load_state_dict(torch.load(f"{model_folder}/{filename}"))

nsample = 30000 # 3000 * 4 * 8
ground = 0
for i, val in enumerate(test_loader):
    ground = val['observed_mask'].to(args["device"]).float() # (B, L, K)

sample_folder = './data/pm25/miss_patterns'

if not os.path.isdir(sample_folder):
    os.makedirs(sample_folder)
# L = 48, K = 35
with torch.no_grad():
    output = model_csdi.evaluate(nsample, shape=(1, 36, 36))
    samples = output
    samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)

    print(f"sample 1: {samples[0][0].cpu().numpy()}")
    print(f"sample 1: {samples[0][2].cpu().numpy()}")
    print(f"sample 1: {samples[0][3].cpu().numpy()}")
    samples = torch.round(torch.abs(samples))
    save_samples = samples.squeeze(0)
    print(f"sample 1: {save_samples[0].cpu().numpy()}")
    print(f"sample 1: {save_samples[2].cpu().numpy()}")
    print(f"sample 1: {save_samples[3].cpu().numpy()}")
    for i in range(save_samples.shape[0]):
        np.save(f"{sample_folder}/pattern_{i}.npy", save_samples[i].cpu().numpy())

    crps_avg = 0
    num = 0
    for i in range(len(ground)):
        crps = calc_quantile_CRPS(ground[i].unsqueeze(0), samples, 0, 1)
        print(f"CRPS for {i} : {crps}")
        crps_avg += crps
        num += 1
    print(f"final CRPS: {crps_avg / num}")