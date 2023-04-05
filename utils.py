import json
import random

import torch
import numpy as np
import pandas as pd

import globals

def read_index_map(file_path):
    """Reads the sign to predict as json file."""
    with open(file_path, "r") as f:
        result = json.load(f)
    return result    

def read_train(file_path):
    """Reads the train csv as pandas data frame."""
    return pd.read_csv(file_path).set_index(globals.CSV_INDEX)

def read_landmark_data_by_path(file_path):
    """Reads landmak data by the given file path."""
    data = pd.read_parquet(file_path)
    return data.set_index(globals.CSV_ROW_ID)

def read_landmark_data_by_id(sequence_id, train_data):
    """Reads the landmark data by the given sequence id."""
    file_path = train_data.loc[sequence_id]['path']
    return read_landmark_data_by_path(file_path)

def read_idxs(file_path):
    return np.loadtxt(file_path, dtype=int)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

# ROWS_PER_FRAME = 543
# def load_relevant_data_subset(pq_path):
#     """Load data as per the evaluation procedure."""
#     data_columns = ['x', 'y', 'z']
#     data = pd.read_parquet(pq_path, columns=data_columns)
#     n_frames = int(len(data) / ROWS_PER_FRAME)
#     data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
#     return data.astype(np.float32)

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)