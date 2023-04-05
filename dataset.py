import os
import pickle
from IPython import embed
from sys import exit

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler

import globals
from utils import read_train, read_index_map, read_idxs, read_landmark_data_by_path


class GASLRDataset(Dataset):

    def __init__(self, data):
        self.data = data 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        sample_fpath = os.path.join(globals.DATA_RAW_DPATH, sample.path)
        X = read_landmark_data_by_path(sample_fpath)[['x','y','z']].values
        X = X.reshape(-1, globals.N_LANDMARKS, 3)
        y = sample.label

        return X, y

def my_collate_fn(samples):
    X = [s[0] for s in samples]
    y = torch.tensor([s[1] for s in samples])
    
    return X, y


def get_dataloader(
    min_n_frames,
    max_n_frames,
    missing_face,
    missing_pose,
    missing_rhand,
    missing_lhand,
    filter_by_hand,
    split,
    batch_size,
):
    # Load data
    data = read_train(os.path.join(globals.DATA_PREPROCESSED_DPATH, 'edata.csv'))
    
    # Load split idxs and filter data by split indices
    try:
        idxs = read_idxs(os.path.join(globals.DATA_PREPROCESSED_DPATH, f"{split}.txt"))
    except:
        raise Exception(f'split {split} not found, valid values are: [train, val, test]')

    # Filter by ids
    data = data.loc[idxs]

    # Filter by frames and landmark missing percentages
    query = f"n_frames >= {min_n_frames} & n_frames <= {max_n_frames} &\
        pct_missing_face <= {missing_face} &\
        pct_missing_pose <= {missing_pose} &\
        (\
            pct_missing_right_hand <= {missing_rhand} |\
            pct_missing_left_hand <= {missing_lhand}\
        )"
    data = data.query(query)

    if filter_by_hand is not None:
        query = f"has_{filter_by_hand}_hand == True"
        data = data.query(query)

    embed();
    exit()
    
    dataset = GASLRDataset(data)

    return DataLoader(
        dataset,
        # sampler=sampler,
        batch_size=batch_size if split != 'test' else 1,
        shuffle=True if split == 'train' else False,
        pin_memory=True,
        num_workers=4,
        collate_fn=my_collate_fn,
        drop_last=True if split == 'train' else False
    )