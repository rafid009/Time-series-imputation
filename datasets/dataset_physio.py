import pickle   

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# 35 attributes which contains enough non-values
attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
              'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
              'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
              'Creatinine', 'ALP']


def extract_hour(x):
    h, _ = map(int, x.split(":"))
    return h


def parse_data(x):
    # extract the last value for each attribute
    x = x.set_index("Parameter").to_dict()["Value"]

    values = []

    for attr in attributes:
        if x.__contains__(attr):
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values


def parse_id(id_, missing_ratio=0.1, is_test=False, forecasting=False, length=-1, random_trial=False, pattern=None):
    data = pd.read_csv("./data/physio/set-a/{}.txt".format(id_))
    # set hour
    data["Time"] = data["Time"].apply(lambda x: extract_hour(x))

    # create data for 48 hours x 35 attributes
    observed_values = []
    for h in range(48):
        observed_values.append(parse_data(data[data["Time"] == h]))
    observed_values = np.array(observed_values)
    observed_masks = ~np.isnan(observed_values)

    # randomly set some percentage as ground-truth
    masks = observed_masks.reshape(-1).copy()
    if pattern is not None:
        choice = np.random.randint(low=pattern['start'], high=(pattern['start'] + pattern['num_patterns'] - 1))
        filename = f"{pattern['pattern_dir']}/pattern_{choice}.npy"
        gt_masks = np.load(filename)
        gt_masks = gt_masks * observed_masks
        eval_mask = gt_masks.reshape(-1).copy()
        gt_indices = np.where(eval_mask)[0].tolist()
        miss_indices = np.random.choice(
            gt_indices, (int)(len(gt_indices) * missing_ratio), replace=False
        )
        gt_intact = observed_values.reshape(-1).copy()
        gt_intact[miss_indices] = np.nan
    elif (not is_test) or random_trial:
        obs_indices = np.where(masks)[0].tolist()
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices] = False
        gt_masks = masks.reshape(observed_masks.shape)
        gt_intact = observed_values.reshape(-1).copy()
        gt_intact[miss_indices] = np.nan
    elif forecasting:
        time_length = observed_values.shape[0]
        start_idx = time_length - length
        gt_intact = observed_values.copy()
        gt_intact[start_idx:, :] = np.nan
        gt_masks = ~np.isnan(gt_intact)
    else:
        a = np.arange(observed_values.shape[0] - length)
        start_idx = np.random.choice(a)
        end_idx = start_idx + length
        gt_intact = observed_values.copy()
        gt_intact[start_idx:end_idx, :] = np.nan
        gt_masks = ~np.isnan(gt_intact)
    
    gt_intact = gt_intact.reshape(observed_values.shape)
    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")
    
    return observed_values, observed_masks, gt_masks, gt_intact


def get_idlist():
    patient_id = []
    for filename in os.listdir("./data/physio/set-a"):
        match = re.search("\d{6}", filename)
        if match:
            patient_id.append(match.group())
    patient_id = np.sort(patient_id)
    return patient_id


class Physio_Dataset(Dataset):
    def __init__(self, eval_length=48, use_index_list=None, missing_ratio=0.0, seed=0, is_test=False, 
                 random_trial=False, forecasting=False, length=-1, pattern=None, mean=False, std=False):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        self.gt_intacts = []
        path = (
            "./data/physio_missing" + str(missing_ratio) + "_seed" + str(seed) + ".pk"
        )
        
        mean_file = './data/physio/mean' + str(missing_ratio) + "_seed" + str(seed) + ".npy"
        std_file = './data/physio/std' + str(missing_ratio) + "_seed" + str(seed) + ".npy"
        # if os.path.isfile(path) == False:  # if datasetfile is none, create
        idlist = get_idlist()
        for id_ in idlist:
            try:
                observed_values, observed_masks, gt_masks, gt_intact = parse_id(
                    id_, missing_ratio, is_test=is_test, forecasting=forecasting, length=length, random_trial=random_trial, pattern=pattern
                )
                self.observed_values.append(observed_values)
                self.observed_masks.append(observed_masks)
                self.gt_masks.append(gt_masks)
                self.gt_intacts.append(gt_intact)
            except Exception as e:
                print(id_, e)
                continue
        self.observed_values = np.array(self.observed_values)
        self.observed_masks = np.array(self.observed_masks)
        self.gt_masks = np.array(self.gt_masks)
        self.gt_intacts = np.array(self.gt_intacts)

        if use_index_list is not None:
            self.observed_values = self.observed_values[use_index_list]
            self.observed_masks = self.observed_masks[use_index_list]
            self.gt_masks = self.gt_masks[use_index_list]
            self.gt_intacts = self.gt_intacts[use_index_list]

        # calc mean and std and normalize values
        # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
        tmp_values = self.observed_values.reshape(-1, 35)
        tmp_masks = self.observed_masks.reshape(-1, 35)
        if mean and std:
            self.mean = np.load(mean_file)
            self.std = np.load(std_file)
        else:
            mu = np.zeros(35)
            sigma = np.zeros(35)
            for k in range(35):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                mu[k] = c_data.mean()
                sigma[k] = c_data.std()
            self.mean = mu
            self.std = sigma
        self.observed_values = (
            ((self.observed_values - self.mean) / self.std) * self.observed_masks
        )

        self.use_index_list = np.arange(len(self.observed_values))
        
        if not mean and not std:
            np.save(mean_file, self.mean)
            np.save(std_file, self.std)
        # else:
        #     self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
            'gt_intact': self.gt_intacts[index]
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1, pattern=None):

    # only to obtain total length of dataset
    dataset = Physio_Dataset(missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = (int)(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    dataset = Physio_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Physio_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed, mean=True, std=True
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Physio_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed, pattern=pattern, mean=True, std=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader, test_index


def get_testloader_physio(test_indices, seed=1, batch_size=16, missing_ratio=0.1, random_trial=False, forecasting=False, length=-1, pattern=None):
    test_dataset = Physio_Dataset(
        use_index_list=test_indices, missing_ratio=missing_ratio, seed=seed, is_test=True, 
        random_trial=random_trial, forecasting=forecasting, length=length, pattern=pattern, mean=True, std=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader
