import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class AWN_Dataset(Dataset):
    def __init__(self, X, eval_length=31, type='Daily') -> None:
        super().__init__()
        self.eval_length = eval_length
        self.observed_values = torch.tensor(X, dtype=torch.float32)
        self.observed_masks = (~torch.isnan(self.observed_values)) * 1.0
    
    def __getitem__(self, index):
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s
    
    def __len__(self):
        return len(self.observed_values)


def get_dataloader(filename='data/Daily/miss_data_yy.npy', batch_size=16, val_ratio=0.2, seed=10, is_year=True, type='Daily'):
    np.random.seed(seed=seed)
    data = np.load(filename)
    length = 366 if is_year else 31
    indices = [i for i in range(len(data))]
    train_indices = np.random.choice(indices, size=int((1 - val_ratio) * len(data)), replace=False)
    train_indices
    test_indices = []
    for i in indices:
        if i not in train_indices:
            test_indices.append(i)
    test_indices = np.array(test_indices)
    train_dataset = AWN_Dataset(data[train_indices], eval_length=length, type=type)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = AWN_Dataset(data[test_indices], eval_length=length, type=type)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

