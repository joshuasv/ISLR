import os

import numpy as np
import pandas as pd

import globals
from utils import read_train

def split_dataset_by_index(
    data_dpath=globals.DATA_RAW_DPATH, 
    out_dpath=globals.DATA_PREPROCESSED_DPATH,
    test_pct=.2, 
    val_pct=.15,
    seed=42
):
    # Check if file exists, if it does ask for user input
    files = os.listdir(out_dpath)
    a = 'y'
    if 'train.txt' in files and 'train.txt' in files and 'val.txt' in files:
        a = input('Train/val/test splits already exist. Do you want to proceed? [Y/n]\n')
        a = a.lower()
    if a == 'y':
        train = read_train(os.path.join(data_dpath, globals.RAW_CSV_FNAME))
        # Shuffle
        train = train.sample(frac=1, random_state=seed)
        # Get test samples
        test = train.sample(frac=test_pct, random_state=seed)
        train.drop(test.index, inplace=True)
        # Get val samples
        val = train.sample(frac=val_pct, random_state=seed)
        train.drop(val.index, inplace=True)
        
        train_idxs = train.index.values
        val_idxs = val.index.values
        test_idxs = test.index.values
        # Check that any of the indices are in any of the other splits
        assert not np.isin(train_idxs, test_idxs).any()
        assert not np.isin(train_idxs, val_idxs).any()
        assert not np.isin(test_idxs, val_idxs).any()
        # Save to disk file with idxs
        np.savetxt(os.path.join(out_dpath, "train.txt"), train_idxs)
        np.savetxt(os.path.join(out_dpath, "val.txt"), val_idxs)
        np.savetxt(os.path.join(out_dpath, "test.txt"), test_idxs)

# def extend_raw_dataset_info():
#     def myf(path):
#     parquet_df = read_landmark_data_by_path(path)
#     n_frames = len(parquet_df.frame.value_counts())
    
#     # Ignore z
#     lhand_query = parquet_df.query('type == "left_hand"')[['x', 'y']].values.reshape(n_frames, -1, 2)
#     rhand_query = parquet_df.query('type == "right_hand"')[['x', 'y']].values.reshape(n_frames, -1, 2)
#     pose_query = parquet_df.query('type == "pose"')[['x', 'y']].values.reshape(n_frames, -1, 2)
#     face_query = parquet_df.query('type == "face"')[['x', 'y']].values.reshape(n_frames, -1, 2)
    
#     lhand_nan_count = np.isnan(lhand_query).sum()
#     rhand_nan_count = np.isnan(rhand_query).sum()
#     pose_nan_count = np.isnan(pose_query).sum()
#     face_nan_count = np.isnan(face_query).sum()
    
#     lhand_query_size = lhand_query.size
#     rhand_query_size = rhand_query.size
#     pose_query_size = pose_query.size
#     face_query_size = face_query.size
    
#     has = lambda nan_count, query_size: nan_count < query_size
#     has_lhand = has(lhand_nan_count, lhand_query_size)
#     has_rhand = has(rhand_nan_count, rhand_query_size)
#     has_pose = has(pose_nan_count, pose_query_size)
#     has_face = has(face_nan_count, face_query_size)
    
#     has_complete = lambda nan_count: not nan_count
#     has_complete_lhand = has_complete(lhand_nan_count)
#     has_complete_rhand = has_complete(rhand_nan_count)
#     has_complete_pose = has_complete(pose_nan_count)
#     has_complete_face = has_complete(face_nan_count)
    
#     # np.argwhere result (n_frame, landmark_id, coord_idx(0/x,1/y)
#     out_of_frame_values = lambda query: np.argwhere((query < 0) | (query > 1))[:, 1]
#     out_of_frame_values_pct_lhand = out_of_frame_values(lhand_query).size / lhand_query_size
#     out_of_frame_values_pct_rhand = out_of_frame_values(rhand_query).size / rhand_query_size
#     # Filter lower body landmarks idxs
#     out_of_frame_values_pct_pose = np.isin(out_of_frame_values(pose_query), POSE_LOWER_BODY_LANDMARKS, invert=True).sum()
#     out_of_frame_values_pct_pose = out_of_frame_values_pct_pose / pose_query_size
#     out_of_frame_values_pct_face = out_of_frame_values(face_query).size / face_query_size

#     return [n_frames, has_lhand, has_rhand, has_pose, has_face, 
#             has_complete_lhand, has_complete_rhand, has_complete_pose,
#             has_complete_face, out_of_frame_values_pct_lhand,
#             out_of_frame_values_pct_rhand, out_of_frame_values_pct_pose, 
#             out_of_frame_values_pct_face]

#     def generate_extended_csv(train_data, name='extended_train_data'):
#         train_data[['n_frames', 'has_lhand', 'has_rhand', 'has_pose', 'has_face',
#                     'has_complete_lhand', 'has_complete_rhand', 'has_complete_pose', 
#                     'has_complete_face', 'out_of_frame_values_pct_lhand',
#                     'out_of_frame_values_pct_rhand', 'out_of_frame_values_pct_pose', 
#                     'out_of_frame_values_pct_face']] = train_data.path.swifter.apply(myf).to_list()
#         train_data.to_csv(f"{name}.csv")
        
#         return train_data