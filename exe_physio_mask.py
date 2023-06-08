import argparse
import torch
import datetime
import json
import yaml
import os
from pypots.imputation import SAITS
import pickle
import numpy as np
np.set_printoptions(threshold=np.inf)
from models.mask_main_model import Mask_Physio
from datasets.dataset_physio_mask import get_dataloader, attributes
from utils.utils import train, evaluate, get_num_params, evaluate_imputation_all

def quantile_loss(target, forecast, q: float) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target):
    return torch.sum(torch.abs(target))


def calc_quantile_CRPS(target, forecast, mean_scaler, scaler):
    print(f"target: {target.shape}\nforecast: {forecast.shape}")
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    print(f"target: {target}")
    print(f"forecasts: {forecast[0:10]}")

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i])
        print(f"q_loss: {q_loss}, denom: {denom}")
        CRPS += q_loss / denom
        print(f"CRPS each qunatile: {CRPS}")
    return CRPS.item() / len(quantiles)

args = {
    'config': 'base.yaml',
    'device': 'cuda:0',
    'seed': 1,
    'testmissingratio': 0.1,
    'nfold': 0,
    'unconditional': False,
    'modelfolder': 'saved_model_physio',
    'nsample': 50
}

path = "config/" + args["config"]
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = True #args["unconditional"]
config["model"]["test_missing_ratio"] = args["testmissingratio"]
print(f"config_csdi:\n")
print(json.dumps(config, indent=4))



train_loader, valid_loader, test_loader, test_indices = get_dataloader(
    seed=args["seed"],
    nfold=args["nfold"],
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)
config['model']['type'] = 'CSDI'
config['diffusion']['is_fast'] = False

model_csdi = Mask_Physio(config, args['device']).to(args['device'])
model_folder = "saved_model_physio_mask"
filename = "model_csdi_physio_mask.pth"
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

nsample = 40000 # 3000 * 4 * 8
ground = 0
for i, val in enumerate(test_loader):
    ground = val['observed_mask'].to(args["device"]).float() # (B, L, K)
    # ground = ground.reshape(ground.shape[0], -1).cpu().numpy()

sample_folder = './data/physio/miss_patterns'

if not os.path.isdir(sample_folder):
    os.makedirs(sample_folder)
# L = 48, K = 35
with torch.no_grad():
    output = model_csdi.evaluate(nsample, shape=(1, len(attributes), 48))
    samples = output

    samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)

    # samples = (samples > 0).float()
    # np.save(f"{sample_folder}/patterns.npy", samples.cpu().numpy())
    print(f"sample 1: {samples[0][0].cpu().numpy()}")
    print(f"sample 1: {samples[0][2].cpu().numpy()}")
    print(f"sample 1: {samples[0][3].cpu().numpy()}")
    samples = torch.round(torch.abs(samples))
    # samples = samples.reshape(samples.shape[0], samples.shape[1], -1).cpu().numpy()
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
    