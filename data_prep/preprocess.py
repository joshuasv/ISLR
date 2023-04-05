import os
import pickle
from sys import exit
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.impute import KNNImputer

import globals
from utils import read_train, read_index_map, read_landmark_data_by_id, read_landmark_data_by_path


def get_landmarks_to_include_idxs(landmarks_to_include):
    """Add to landmark keypoints its corresponding offset to enable contiguous
    array access.

    All landmarks from left and right hand are always included. Only
    functionality for including a subset of landmarks from pose and face is 
    handled.
    
    Parameters
    ----------
    landmarks_to_include : dict
        Contains the landmarks to include in the dataset for the pose and face
        landmark groups

    Returns
    -------
    numpy.array (1d,)
        Contains all the landmarks indices that can be used to access an array
        in a contiguous fashion. 
    """
    face = landmarks_to_include['face']
    pose = landmarks_to_include['pose']

    face_idxs = []
    if face == 'ALL':
        face_idxs = range(
            globals.FACE_OFFSET, 
            globals.FACE_OFFSET + globals.FACE_N_LANDMARKS
        )
    else:
        for var in face:
            face_idxs.extend(globals.__dict__[var])
        face_idxs = [idx + globals.FACE_OFFSET for idx in face_idxs]
    pose_idxs = []
    if pose == 'ALL':
        pose_idxs = range(
            globals.POSE_OFFSET, 
            globals.POSE_OFFSET + globals.POSE_N_LANDMARKS
    )
    else:
        for var in pose:
            pose_idxs.extend(globals.__dict__[var])
        pose_idxs = [idx + globals.POSE_OFFSET for idx in pose_idxs]

    face_idxs = np.array(sorted(face_idxs))
    pose_idxs = np.array(sorted(pose_idxs))
    rhand_idxs = np.array(
        range(
            globals.RHAND_OFFSET, 
            globals.RHAND_OFFSET + globals.RHAND_N_LANDMARKS
        )
    )
    lhand_idxs = np.array(
        range(
            globals.LHAND_OFFSET,
            globals.LHAND_OFFSET + globals.LHAND_N_LANDMARKS
        )
    )

    return np.concatenate((face_idxs, lhand_idxs, pose_idxs, rhand_idxs))

def read_xyz_coords(path, max_n_frames, imputer, get_idxs_by_n_frames):
    """Extract and preprocess the X, Y and Z coordinates from a suequence stored
    in a parquet file.

    The preprocessing steps are:
        1. Filter by the landmarks of interest
        2. Replace with 0.0 all NaN values
        3. Impute missing values
        4. Repeat sequence until max_n_frames reached
    
    Parameters
    ----------
    path : str
        File path pointing to a parquet file
    max_n_frames : int
        Maximum number of frames that the output sequence must have
    imputer : sklearn.impute.KNNImputer
        Assign values to those that are missing
    get_idxs_by_n_frames : function <lambda>
            

    Returns
    -------
        xyz : numpy.ndarray (max_n_frames, landmarks_of_interest, 3)

    """
    xyz = read_landmark_data_by_path(path)
    # Get the total number of frames f
    n_frames = len(xyz.frame.unique())
    # Step 1, obtain the landmarks of interest 
    xyz = xyz.iloc[get_idxs_by_n_frames(n_frames)]
    xyz = xyz[['x', 'y', 'z']].values.reshape(n_frames, -1, 3).astype(np.float32)
    # Obtain landmark indices that have missing values
    lmks_wnan_idxs = np.argwhere(np.isnan(xyz).sum(-1).sum(0) > 0).ravel()
    # Step 2, replace NaNs with 0.0
    np.nan_to_num(xyz, copy=False, nan=0.0)
    to_impute = xyz[:, lmks_wnan_idxs].reshape(n_frames, -1)
    # Step 3, impute missing values
    imputer.fit_transform(to_impute)
    xyz[:, lmks_wnan_idxs] = to_impute.reshape(n_frames, len(lmks_wnan_idxs), 3)
    # Step 4, repeat the sequence until max_n_frames
    rest = max_n_frames - n_frames
    to_repeat = int(np.ceil(max_n_frames / n_frames)) if rest > 0 else 1
    xyz = np.repeat(xyz, to_repeat, axis=0)[:max_n_frames]

    return xyz

def read_xyz(path, padding_frames):
    xyz = read_landmark_data_by_path(os.path.join(globals.DATA_RAW_DPATH, path))
    # Get the total number of frames f
    n_frames = len(xyz.frame.unique())
    xyz = xyz[['x','y','z']].values.reshape(n_frames, -1, 3).astype(np.float32)
    # Repeat the sequence until max_n_frames
    rest = padding_frames - n_frames
    to_repeat = int(np.ceil(padding_frames / n_frames)) if rest > 0 else 1
    xyz = np.repeat(xyz, to_repeat, axis=0)[:padding_frames]

    return xyz

def preprocess(raw_data_dpath, out_root_dpath, config):
    """Preprocess the raw data to be ingested by a deep learning model.

    First the raw data is filtered by the number of frames that a sequence has.
    A sequence will be considered valid if it has a number of frames >= than 
    min_n_frames and <= max_n_frames. Then the X, Y, and Z coordinates are
    processed and extracted for each sequence.

    The sklearn.impute.KNNImputer weights neighbors by distance. Moreover, 
    instead of np.nan it treats 0.0 floats as missing values. 

    Parameters
    ----------
    raw_data_dpath : str
        Path where raw data is stored
    out_root_dpath: str
        Root path where outputs are going to be saved
    config: dict
        Configuration that determines how the preprocessing must be done

    Returns
    -------
    None
        Processed files are saved in out_root_dpath/config_name_file directory.

    """
    max_n_frames = config['max_n_frames']
    min_n_frames = config['min_n_frames']
    config_fname = config['config_fname']
    missing_face = config['missing_face']
    missing_lhand = config['missing_lhand']
    missing_rhand = config['missing_rhand']
    missing_pose = config['missing_pose']
    padding_frames = config['padding_frames']

    # Create output directory structure
    out_dpath = os.path.join(globals.DATA_PREPROCESSED_DPATH, config_fname)
    out_joint_data_fpath = os.path.join(out_dpath, 'joint_data.npy')
    out_ids_lbls_fpath = os.path.join(out_dpath, 'ids_lbls.npy')
    out_landmarks_idxs_fpath = os.path.join(out_dpath, 'landmarks_idxs.npy')
    out_data_shape_fpath = os.path.join(out_dpath, 'data_shape.pkl')

    # Check if directory exists and is not empty (not the safest, just for me)
    if os.path.exists(out_dpath) and len(os.listdir(out_dpath)) > 0:
        user_inp = input(f"{out_dpath} already exists and it is not empty. Override? [Y/n]: ")
        if user_inp.lower() == 'n': exit(0)
    else:
        os.makedirs(out_dpath, exist_ok=True)
    
    # Read the extended raw data so we have information about frame number
    train_data_csv_fpath = os.path.join(
        globals.DATA_PREPROCESSED_DPATH, globals.EXTENDED_RAW_CSV_FNAME)
    data = read_train(train_data_csv_fpath)

    # Do not process any of the test sequences
    test_idxs = np.loadtxt(os.path.join(globals.DATA_PREPROCESSED_DPATH, 'test.txt'))
    data = data[~data.index.isin(test_idxs)]

    # Filter by frames and landmark missing percentages
    query = f"n_frames >= {min_n_frames} & n_frames <= {max_n_frames} &\
        pct_missing_face <= {missing_face} &\
        pct_missing_pose <= {missing_pose} &\
        (\
            pct_missing_right_hand <= {missing_rhand} |\
            pct_missing_left_hand <= {missing_lhand}\
        )"
    data = data.query(query)

    if config['DEBUG']:
        data = data[:100]
    
    # Extract coordinates
    # tqdm.pandas(desc='Extracting coordinates')
    data_shape = (len(data), padding_frames, globals.N_LANDMARKS, 3)
    fp = np.memmap(out_joint_data_fpath, dtype=np.float32, mode='w+', shape=data_shape)

    chunk_size = 5000
    if config['DEBUG']:
            chunk_size = 10

    with tqdm(total=len(data)) as pbar:
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size]
            result_chunk = np.stack(chunk.path.apply(func=read_xyz, args=(padding_frames,)).values)
            fp[i:i+chunk_size] = result_chunk
            fp.flush()
            end = min(i + chunk_size, len(data))
            pbar.update(end - i)

    del fp
    # fp = np.stack(
    #     data.path.progress_apply(func=read_xyz, args=(padding_frames,)).values
    # )

    # # Imputer used to assign missing values
    # imputer = KNNImputer(
    #     n_neighbors=n_neighbors,
    #     copy=False,
    #     weights='distance',
    #     missing_values=0.0,
    #     keep_empty_features=True
    # )

    # # Obtain landmarks to include indices
    # lmks_in_idxs = get_landmarks_to_include_idxs(landmarks_to_include)
    # # Number of landmarks included
    # n_lmks_in = len(lmks_in_idxs)
    # # Pre compute the coordinates up to the maximum number of frames to face
    # all_lmks_in_idxs = np.concatenate(
    #     [lmks_in_idxs + (globals.N_LANDMARKS * i) for i in range(max_n_frames)]
    # )
    # # Get all the indices up to a given frame 
    # get_idxs_by_n_frames = lambda n_frames: all_lmks_in_idxs[:n_lmks_in * n_frames]

    # # Extract coordinates
    # tqdm.pandas(desc='Extracting coordinates')
    # fp = np.stack(
    #     data.path.progress_apply(
    #         func=read_xyz_coords, 
    #         args=(max_n_frames, imputer, get_idxs_by_n_frames,),
    #     ).values
    # )

    # Save labels, participant indices and sequence indices 
    ids_lbls = np.c_[
        data.index.values, data.participant_id.values, data.label.values]

    print(f"Saving data to {out_dpath}...")
    # np.save(out_joint_data_fpath, fp)
    np.save(out_ids_lbls_fpath, ids_lbls)
    # np.save(out_landmarks_idxs_fpath, lmks_in_idxs)
    with open(out_data_shape_fpath, 'wb') as fid:
        pickle.dump(data_shape, fid)

    print(f'Dataset has:', len(data), 'sequences')
