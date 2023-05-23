from models.mask_main_model import Mask_AWN
from datasets.dataset_awn_mask import get_dataloader
from utils.utils import train, get_num_params, calc_quantile_CRPS, evaluate_imputation_all
import numpy as np
import torch
import sys
import os
from pypots.imputation import SAITS
import matplotlib.pyplot as plt
import matplotlib
from datasets.preprocess_awn import features
import pickle
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
        'is_unconditional': True,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "random",
        'type': 'CSDI',
        'n_layers': 4, 
        'd_time': 100,
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

n_steps = 100
n_features = len(features)
num_seasons = 32
noise = False
filename = './data/Daily/miss_data_yy.npy'
train_loader, valid_loader = get_dataloader(filename, 8, 0.2, is_year=True, type='Daily')

model_csdi = Mask_AWN(config_dict_csdi, device, target_dim=len(features)).to(device)
model_folder = "./saved_model_mask_awn"
filename = f"model_csdi_mask_awn.pth"
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)
print(f"\n\nCSDI Masked training starts.....\n")
train(
    model_csdi,
    config_dict_csdi["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=model_folder,
    filename=f"{filename}",
    is_saits=False
)
# model_csdi.load_state_dict(torch.load(f"{model_folder}/{filename}"))
print(f"CSDI params: {get_num_params(model_csdi)}")