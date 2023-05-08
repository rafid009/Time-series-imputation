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
from datasets.synthetic_data import create_synthetic_data_v3, feats_v3
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

given_features = feats_v3 #['sin', 'cos2', 'harmonic', 'weight', 'lin_comb', 'non_lin_comb', 'mixed_history']

def evaluate_imputation(models, mse_folder, exclude_key='', exclude_features=None, trials=20, length=100, season_idx=None, random_trial=False, forward_trial=False, data=False, missing_ratio=0.2):
    # given_features = given_features = ['sin', 'cos2', 'harmonic', 'weight', 'inv'] 
    nsample = 30
    # trials = 30
    season_avg_mse = {}
    num_seasons = 1
    n_steps = 100
    # exclude_features = ['MEAN_AT', 'MIN_AT', 'AVG_AT', 'MAX_AT']
    results = {}
    models['CSDI'].eval()
    models['DiffSAITS'].eval()

    for season in range(num_seasons):
        print(f"For season: {season}")
        mse_csdi_total = {}
        mse_saits_total = {}
        mse_diff_saits_total = {}
        CRPS_csdi = 0
        CRPS_diff_saits = 0
        for i in range(trials):
            test_loader = get_testloader(n_steps, len(given_features), 1, exclude_features=exclude_features, length=length, seed=5*i, forward_trial=forward_trial, random_trial=random_trial, missing_ratio=missing_ratio)
            
            for j, test_batch in enumerate(test_loader, start=1):
                if 'CSDI' in models.keys():
                    output = models['CSDI'].evaluate(test_batch, nsample)
                    samples, c_target, eval_points, observed_points, observed_time, obs_intact, gt_intact = output
                    samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)
                    samples_median = samples.median(dim=1)
                
                if 'DiffSAITS' in models.keys():
                    output_diff_saits = models['DiffSAITS'].evaluate(test_batch, nsample)
                    if 'CSDI' not in models.keys():
                        samples_diff_saits, c_target, eval_points, observed_points, observed_time, obs_intact, gt_intact = output_diff_saits
                        c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                        eval_points = eval_points.permute(0, 2, 1)
                        observed_points = observed_points.permute(0, 2, 1)
                    else:
                        samples_diff_saits, _, _, _, _, _, _ = output_diff_saits
                    samples_diff_saits = samples_diff_saits.permute(0, 1, 3, 2)
                    samples_diff_saits_median = samples_diff_saits.median(dim=1)
                    samples_diff_saits_mean = samples_diff_saits.mean(dim=1)

                # gt_intact = gt_intact.squeeze(axis=0)
                saits_X = gt_intact #test_batch['obs_data_intact']
                saits_output = models['SAITS'].impute(saits_X)
                
                if data:
                    if 'CSDI' in models.keys():
                        results[season] = {
                            'target mask': eval_points[0, :, :].cpu().numpy(),
                            'target': c_target[0, :, :].cpu().numpy(),
                            # 'csdi_mean': samples_mean[0, :, :].cpu().numpy(),
                            'csdi_median': samples_median.values[0, :, :].cpu().numpy(),
                            'csdi_samples': samples[0].cpu().numpy(),
                            'saits': saits_output[0, :, :],
                            'diff_saits_mean': samples_diff_saits_mean[0, :, :].cpu().numpy(),
                            'diff_saits_median': samples_diff_saits_median.values[0, :, :].cpu().numpy(),
                            'diff_saits_samples': samples_diff_saits[0].cpu().numpy(),
                            # 'diff_saits_median_simple': samples_diff_saits_median_simple.values[0, :, :].cpu().numpy(),
                            # 'diff_saits_samples_simple': samples_diff_saits_simple[0].cpu().numpy()
                            }
                    else:
                         results[season] = {
                            'target mask': eval_points[0, :, :].cpu().numpy(),
                            'target': c_target[0, :, :].cpu().numpy(),
                            'saits': saits_output[0, :, :],
                            'diff_saits_mean': samples_diff_saits_mean[0, :, :].cpu().numpy(),
                            # 'diff_saits_median': samples_diff_saits_median.values[0, :, :].cpu().numpy(),
                            'diff_saits_samples': samples_diff_saits[0].cpu().numpy(),
                        }
                else:
                    for feature in given_features:
                        if exclude_features is not None and feature in exclude_features:
                            continue
                        # print(f"For feature: {feature}, for length: {length}, trial: {random_trial}")
                        feature_idx = given_features.index(feature)
                        if eval_points[0, :, feature_idx].sum().item() == 0:
                            continue
                        if 'CSDI' in models.keys():
                            mse_csdi = ((samples_median.values[0, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                            mse_csdi = math.sqrt(mse_csdi.sum().item() / eval_points[0, :, feature_idx].sum().item())

                            mae_csdi = torch.abs((samples_median.values[0, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx])
                            mae_csdi = mae_csdi.sum().item() / eval_points[0, :, feature_idx].sum().item()
                            
                            if feature not in mse_csdi_total.keys():
                                mse_csdi_total[feature] = {'rmse': 0, 'mae': 0}
                            
                            mse_csdi_total[feature]["rmse"] += mse_csdi
                            mse_csdi_total[feature]['mae'] += mae_csdi

                        if feature not in mse_diff_saits_total.keys():
                            mse_diff_saits_total[feature] = {'rmse': 0, 'mae': 0, 'diff_rmse_med': 0}

                        mse_diff_saits = ((samples_diff_saits_mean[0, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                        mse_diff_saits = math.sqrt(mse_diff_saits.sum().item() / eval_points[0, :, feature_idx].sum().item())

                        mse_diff_saits_median = ((samples_diff_saits_median.values[0, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                        mse_diff_saits_median = math.sqrt(mse_diff_saits_median.sum().item() / eval_points[0, :, feature_idx].sum().item())

                        mae_diff_saits = torch.abs((samples_diff_saits_mean[0, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx])
                        mae_diff_saits = mae_diff_saits.sum().item() / eval_points[0, :, feature_idx].sum().item()
                        
                        mse_diff_saits_total[feature]["rmse"] += mse_diff_saits
                        mse_diff_saits_total[feature]["mae"] += mae_diff_saits
                        mse_diff_saits_total[feature]["diff_rmse_med"] += mse_diff_saits_median


                        mse_saits = ((torch.tensor(saits_output[0, :, feature_idx], device=device)- c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                        mae_saits = torch.abs((torch.tensor(saits_output[0, :, feature_idx], device=device)- c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx])
                        mse_saits = math.sqrt(mse_saits.sum().item() / eval_points[0, :, feature_idx].sum().item())
                        mae_saits = mae_saits.sum().item() / eval_points[0, :, feature_idx].sum().item()

                        if feature not in mse_saits_total.keys():
                            mse_saits_total[feature] = {'rmse': 0, 'mae': 0}

                        mse_saits_total[feature]['rmse'] += mse_saits
                        mse_saits_total[feature]['mae'] += mae_saits
                CRPS_csdi += calc_quantile_CRPS(c_target, samples, eval_points, 0, 1)
                CRPS_diff_saits += calc_quantile_CRPS(c_target, samples_diff_saits, eval_points, 0, 1)
        print(f"CSDI CRPS: {CRPS_csdi/trials}")
        print(f"DiffSAITS CRPS: {CRPS_diff_saits/trials}")
        if not data:
            print(f"For season = {season}:")
            for feature in given_features:
                if exclude_features is not None and feature in exclude_features:
                    continue
                # if feature not in mse_csdi_total.keys() or feature not in mse_diff_saits_total.keys():
                #     continue
                if 'CSDI' in models.keys():
                    for i in mse_csdi_total[feature].keys():
                        mse_csdi_total[feature][i] /= trials
                for i in mse_diff_saits_total[feature].keys():
                    mse_diff_saits_total[feature][i] /= trials
                for i in mse_saits_total[feature].keys():
                    mse_saits_total[feature][i] /= trials
                if 'CSDI' in models.keys():
                    print(f"\n\tFor feature = {feature}\n\tCSDI mae: {mse_csdi_total[feature]['mae']}\n\tDiffSAITS mae: {mse_diff_saits_total[feature]['mae']}")
                    print(f"\n\tFor feature = {feature}\n\tCSDI rmse: {mse_csdi_total[feature]['rmse']}\n\tDiffSAITS rmse: {mse_diff_saits_total[feature]['rmse']}\n\tDiffSAITS median: {mse_diff_saits_total[feature]['diff_rmse_med']}\n\tSAITS mse: {mse_saits_total[feature]['rmse']}")# \
                else:
                    print(f"\n\tFor feature = {feature}\n\tDiffSAITS mae: {mse_diff_saits_total[feature]['mae']}")
                    print(f"\n\tFor feature = {feature}\n\tDiffSAITS mse: {mse_diff_saits_total[feature]['rmse']}")# \
                
                # DiffSAITSsimple mse: {mse_diff_saits_simple_total[feature]}")
                # except:
                #     continue
            if 'CSDI' in models.keys():
                season_avg_mse[season] = {
                    'CSDI': mse_csdi_total,
                    'SAITS': mse_saits_total,
                    'DiffSAITS': mse_diff_saits_total#,
                    # 'DiffSAITSsimple': mse_diff_saits_simple_total
                }
            else:
                season_avg_mse[season] = {
                    'SAITS': mse_saits_total,
                    'DiffSAITS': mse_diff_saits_total#,
                }

    if not os.path.isdir(mse_folder):
        os.makedirs(mse_folder)
    if data:
        fp = open(f"{mse_folder}/samples-{exclude_key if len(exclude_key) != 0 else 'all'}-{length}_random_{random_trial}_forecast_{forward_trial}_miss_{missing_ratio}.json", "w")
        json.dump(results, fp=fp, indent=4, cls=NumpyArrayEncoder)
        fp.close()
    else:
        out_file = open(f"{mse_folder}/mse_mae_{exclude_key if len(exclude_key) != 0 else 'all'}_{length}_random_{random_trial}_forecast_{forward_trial}_miss_{missing_ratio}.json", "w")
        json.dump(season_avg_mse, out_file, indent = 4)
        out_file.close()


def draw_data_plot(results, f, season, folder='subplots', num_missing=100):
    
    plt.figure(figsize=(40,28))
    plt.title(f"For feature = {f} in Season {season}", fontsize=30)

    ax = plt.subplot(411)
    ax.set_title(f'Feature = {f} Season = {season} original data', fontsize=27)
    plt.plot(np.arange(results['real'].shape[0]), results['real'], 'tab:blue')
    ax.set_xlabel('Days', fontsize=25)
    ax.set_ylabel('Values', fontsize=25)
    ax = plt.subplot(412)
    ax.set_title(f'Feature = {f} Season = {season} missing data data', fontsize=27)
    plt.plot(np.arange(results['missing'].shape[0]), results['missing'], 'tab:blue')
    ax.set_xlabel('Days', fontsize=25)
    ax.set_ylabel('Values', fontsize=25)
    ax = plt.subplot(413)
    ax.set_title(f'Feature = {f} Season = {season} CSDI data', fontsize=27)
    plt.plot(np.arange(results['csdi'].shape[0]), results['csdi'], 'tab:orange')
    ax.set_xlabel('Days', fontsize=25)
    ax.set_ylabel('Values', fontsize=25)
    ax = plt.subplot(414)
    ax.set_title(f'Feature = {f} Season = {season} SAITS data', fontsize=27)
    plt.plot(np.arange(results['saits'].shape[0]), results['saits'], 'tab:green')
    ax.set_xlabel('Days', fontsize=25)
    ax.set_ylabel('Values', fontsize=25)
    
    plt.tight_layout(pad=5)
    folder = f"{folder}/{season}"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    plt.savefig(f"{folder}/{f}-imputations-season-{season}-{num_missing}.png", dpi=300)
    plt.close()


def evaluate_imputation_data(models, exclude_key='', exclude_features=None, length=50):
    n_samples = 2
    nsample = 30
    i = 0
    data_folder = "data_synth"
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    for season in range(n_samples):
        print(f"For season: {season}")
        i += 1
        test_loader = get_testloader(50, len(given_features), 1, length=length, exclude_features=exclude_features, seed=50+10*i)
        for i, test_batch in enumerate(test_loader, start=1):
            output = models['CSDI'].evaluate(test_batch, nsample)
            samples, c_target, eval_points, observed_points, observed_time, obs_data_intact, gt_intact = output
            samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
            c_target = c_target.permute(0, 2, 1)  # (B,L,K)
            eval_points = eval_points.permute(0, 2, 1)
            observed_points = observed_points.permute(0, 2, 1)
            samples_median = samples.median(dim=1)
            gt_intact = gt_intact.squeeze(axis=0)
            # print(f"gt_intact: {gt_intact.shape}")
            saits_output = models['SAITS'].impute(gt_intact)

            for feature in given_features:
                if exclude_features is not None and feature in exclude_features:
                    continue
                # print(f"For feature: {feature}")
                feature_idx = given_features.index(feature)
                # cond_mask = observed_points - eval_points
                missing = gt_intact
                results = {
                    'real': obs_data_intact[0, :, feature_idx].cpu().numpy(),
                    'missing': missing[0, :, feature_idx].cpu().numpy(),
                    'csdi': samples_median.values[0, :, feature_idx].cpu().numpy(),
                    'saits': saits_output[0, :, feature_idx]
                }
                draw_data_plot(results, feature, season, folder=f"{data_folder}/subplots-{exclude_key if len(exclude_key) != 0 else 'all'}", num_missing=length)
                

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
        'schedule': "quad"
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
    }
}

nsample = 50

n_steps = 100
n_features = len(given_features)
num_seasons = 32
noise = True
train_loader, valid_loader = get_dataloader(n_steps, n_features, num_seasons, batch_size=16, missing_ratio=0.1, seed=10, is_test=False, v2=True, noise=noise)

model_csdi = CSDI_Synth(config_dict_csdi, device, target_dim=len(given_features)).to(device)
model_folder = "./saved_model_synth"
filename = f"model_csdi_synth_v3{'_noise' if noise else ''}.pth"
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)
print(f"\n\nCSDI training starts.....\n")
train(
    model_csdi,
    config_dict_csdi["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=model_folder,
    filename=f"{filename}",
    is_saits=True
)
# model_csdi.load_state_dict(torch.load(f"{model_folder}/{filename}"))
print(f"CSDI params: {get_num_params(model_csdi)}")


saits_model_file = f"{model_folder}/saits_model_synth_v3{'_noise' if noise else ''}.pkl"
saits = SAITS(n_steps=n_steps, n_features=n_features, n_layers=3, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=2500, patience=400, device=device)
X, mean, std = create_synthetic_data_v3(n_steps, num_seasons, seed=10, noise=noise)
print(f"\n\SAITS training starts.....\n")
saits.fit(X)
pickle.dump(saits, open(saits_model_file, 'wb'))


saits = pickle.load(open(saits_model_file, 'rb'))

config_dict_diffsaits = {
    'train': {
        'epochs':3000, # 3000 -> ds3
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
        'schedule': "quad"
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "mix", # noise mix
        'type': 'SAITS',
        'n_layers': 3,
        'loss_weight_p': 1,
        'loss_weight_f': 1,
        'd_time': 100,
        'n_feature': len(given_features),
        'd_model': 128,
        'd_inner': 128,
        'n_head': 8,
        'd_k': 64,
        'd_v': 64,
        'dropout': 0.1,
        'diagonal_attention_mask': False
    }
}
print(f"config: {config_dict_diffsaits}")
model_diff_saits = CSDI_Synth(config_dict_diffsaits, device, target_dim=len(given_features)).to(device)

filename = f"model_diffsaits_synth_v3_qual{'_noise' if noise else ''}_random.pth"
print(f"\n\DiffSAITS training starts.....\n")

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
    'CSDI': model_csdi,
    'SAITS': saits,
    'DiffSAITS': model_diff_saits
}
mse_folder = f"results_synth_v3_qual{'_noise' if noise else ''}_mix"
data_folder = f"results_synth_v3_qual{'_noise' if noise else ''}_mix"
lengths = [10, 50, 90]
for l in lengths:
    print(f"\nlength = {l}")
    print(f"\nBlackout:")
    evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='synth_v3', batch_size=32, length=l, noise=noise)
    evaluate_imputation_all(models=models, mse_folder=data_folder, dataset_name='synth_v3', length=l, trials=1, batch_size=1, data=True, noise=noise)

print(f"\nForecasting:")
evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='synth_v3', batch_size=32, length=(10, 80), forecasting=True, noise=noise)
    # evaluate_imputation(models, mse_folder=data_folder, length=l, forward_trial=True, trials=1, data=True)
evaluate_imputation_all(models=models, mse_folder=data_folder, forecasting=True, dataset_name='synth_v3', length=l, trials=1, batch_size=1, data=True, noise=noise)

miss_ratios = [0.1, 0.5, 0.9]
for ratio in miss_ratios:
    print(f"\nRandom Missing: ratio ({ratio})")
    evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='synth_v3', batch_size=32, missing_ratio=ratio, random_trial=True, noise=noise)
#     # evaluate_imputation(models, mse_folder=data_folder, length=l, random_trial=True, trials=1, data=True, missing_ratio=ratio)
    evaluate_imputation_all(models=models, mse_folder=data_folder, dataset_name='synth_v3', trials=1, batch_size=1, data=True, missing_ratio=ratio, random_trial=True, noise=noise)
