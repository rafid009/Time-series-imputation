import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def parse_data(sample, rate=0.3, is_test=False, length=100, include_features=None, forward_trial=-1, lte_idx=None, random_trial=False):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    if isinstance(sample, torch.Tensor):
        sample = sample.numpy()

    obs_mask = ~np.isnan(sample)
    
    if not is_test:
        shp = sample.shape
        evals = sample.reshape(-1).copy()
        indices = np.where(~np.isnan(evals))[0].tolist()
        indices = np.random.choice(indices, int(len(indices) * rate))
        values = evals.copy()
        values[indices] = np.nan
        mask = ~np.isnan(values)
        mask = mask.reshape(shp)
        gt_intact = values.reshape(shp).copy()
        obs_data = np.nan_to_num(evals, copy=True)
        obs_data = obs_data.reshape(shp)
        # obs_data_intact = evals.reshape(shp)
    elif random_trial:
        evals = sample.copy()
        values = evals.copy()
        for i in range(evals.shape[1]):
            indices = np.where(~np.isnan(evals[:, i]))[0].tolist()
            indices = np.random.choice(indices, int(len(indices) * rate))
            values[indices, i] = np.nan
        mask = ~np.isnan(values)
        gt_intact = values
        obs_data = np.nan_to_num(evals, copy=True)
    elif forward_trial != -1:
        indices = np.where(~np.isnan(sample[:, lte_idx]))[0].tolist()
        start = indices[forward_trial]
        obs_data = np.nan_to_num(sample, copy=True)
        gt_intact = sample.copy()
        gt_intact[start:, :] = np.nan
        mask = ~np.isnan(gt_intact)
    else:
        shp = sample.shape
        evals = sample.reshape(-1).copy()
        a = np.arange(sample.shape[0] - length)
        # print(f"a: {a}\nsample: {sample.shape}")
        start_idx = np.random.choice(a)
        # print(f"random choice: {start_idx}")
        end_idx = start_idx + length
        obs_data_intact = sample.copy()
        if include_features is None or len(include_features) == 0:
            obs_data_intact[start_idx:end_idx, :] = np.nan
        else:
            obs_data_intact[start_idx:end_idx, include_features] = np.nan
        mask = ~np.isnan(obs_data_intact)
        gt_intact = obs_data_intact
        obs_data = np.nan_to_num(evals, copy=True)
        obs_data = obs_data.reshape(shp)
        # obs_intact = np.nan_to_num(obs_intact, copy=True)
    # print(f"\n\nobs data 1: {obs_data}")
    return obs_data, obs_mask, mask, sample, gt_intact

class AWN_Dataset(Dataset):
    def __init__(self, filename, is_year=True, rate=0.1, test_index=32, is_test=False, length=100, seed=10, forward_trial=-1, random_trial=False) -> None:
        super().__init__()
        
        self.observed_values = []
        self.obs_data_intact = []
        self.observed_masks = []
        self.gt_masks = []
        self.gt_intact = []
        
        data = np.load(filename)
        length = 366 if is_year else 31
        self.eval_length = length
        indices = [i for i in range(len(data))]

        test_indices = [test_index]
        if not is_test:
            train_indices = []
            for i in indices:
                if i not in test_indices:
                    train_indices.append(i)
            train_indices = np.array(train_indices)
            X = data[train_indices]
        else:
            X = np.expand_dims(data[test_indices], axis=0)
        self.mean = np.nanmean(X, axis=1)
        self.std = np.nanstd(X, axis=1)
        include_features = []

        for i in range(X.shape[0]):
            obs_val, obs_mask, mask, sample, obs_intact = parse_data(X[i], rate, is_test, length, include_features=include_features, \
                                                                     forward_trial=forward_trial, random_trial=random_trial)
            self.observed_values.append(obs_val)
            self.observed_masks.append(obs_mask)
            self.gt_masks.append(mask)
            self.obs_data_intact.append(sample)
            self.gt_intact.append(obs_intact)



        self.gt_masks = torch.tensor(np.array(self.gt_masks), dtype=torch.float32)
        self.observed_values = torch.tensor(np.array(self.observed_values), dtype=torch.float32)
        self.obs_data_intact = np.array(self.obs_data_intact)
        self.gt_intact = np.array(self.gt_intact)
        self.observed_masks = torch.tensor(np.array(self.observed_masks), dtype=torch.float32)
        self.observed_values = ((self.observed_values - self.mean) / self.std) * self.observed_masks
        self.obs_data_intact = ((self.obs_data_intact - self.mean) / self.std) * self.observed_masks.numpy()
        self.gt_intact = ((self.gt_intact - self.mean) / self.std) * self.gt_masks.numpy()

        
    def __getitem__(self, index):
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "obs_data_intact": self.obs_data_intact[index],
            "timepoints": np.arange(self.eval_length),
            "gt_intact": self.gt_intact[index]
        }
        if len(self.gt_masks) == 0:
            s["gt_mask"] = None
        else:
            s["gt_mask"] = self.gt_masks[index]
        return s
    
    def __len__(self):
        return len(self.observed_values)


def get_dataloader(filename, batch_size=16, missing_ratio=0.1, is_test=False, test_index=32, is_year=True):
    # np.random.seed(seed=seed)
    train_dataset = AWN_Dataset(filename, test_index=test_index, is_year=True, rate=missing_ratio)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = AWN_Dataset(filename, test_index=test_index, rate=missing_ratio, is_year=is_year, is_test=is_test)
    if is_test:
        test_loader = DataLoader(test_dataset, batch_size=1)
    else:
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    return train_loader, test_loader


def get_testloader_awn(filename, is_year=True, n_steps=366, batch_size=16, missing_ratio=0.2, seed=10, test_index=32, length=100, forecasting=False, random_trial=False):
    np.random.seed(seed=seed)
    if forecasting:
        forward = n_steps - length
        test_dataset = AWN_Dataset(filename, is_year=is_year, test_index=test_index, rate=missing_ratio, is_test=True, length=length, forward_trial=forward)
    else:
        test_dataset = AWN_Dataset(filename, is_year=is_year, test_index=test_index, rate=missing_ratio, is_test=True, length=length, random_trial=random_trial)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader