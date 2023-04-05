import torch
import numpy as np
import torch.nn as nn

from IPython import embed
from sys import exit

import globals

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
    list
        Contains all the landmarks indices that can be used to access an array
        in a contiguous fashion. 
    """
    idxs = []
    for lmark, to_include in landmarks_to_include.items():
        if to_include is None:
            continue
        else:
            offset = globals.LANDMARK_OFFSETS_MAP[lmark]
            n_landmarks = globals.N_LANDMARKS_MAP[lmark]
            for var in to_include:
                if var == 'ALL':
                    idxs.extend(list(range(offset, offset + n_landmarks)))
                else:
                    idxs.extend([v + offset for v in globals.__dict__[var]])

    return sorted(idxs)


class PreprocessLayer(nn.Module):

    def __init__(self, num_frames, landmarks_to_include, drop_z):
        super().__init__()
        self.num_frames = num_frames
        self.landmarks_idxs = get_landmarks_to_include_idxs(landmarks_to_include)
        self.drop_z = drop_z

    def forward(self, X, device):
        
        # Make elements of the list have equal frame length
        if type(X) is list:
            X = torch.stack([self._num_frames_normalization(x) for x in X])
        elif type(X) is numpy.ndarray:
            X = self._num_frames_normalization(X)
        else:
            raise ValueError(f"X is of type {type(X)}, not expected")

        if X.dim() == 3:
            X = X.unsqueeze(0)
        
        if self.drop_z:
            X = X[...,:2]
        
        # Filter by landmarks
        X = X[:, :, self.landmarks_idxs]

        # Check for NaN
        if torch.isnan(torch.min(X)):
            X = torch.nan_to_num(X)

        # X current shape, (num_seqs, num_frames, num_joints, num_coords)
        # MSG3D expects, (num_seqs, num_coords, num_frames, num_joints, num_ppl)
        X = torch.moveaxis(X, (1, 2, 3), (2, 3, 1))
        
        # Add the people dimension
        X = X.unsqueeze(-1)

        return X.to(device)

    def _num_frames_normalization(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        curr_frames = x.size(0)
        rest = self.num_frames - curr_frames
        to_repeat = int(np.ceil(self.num_frames / curr_frames)) if rest > 0 else 1
        x = torch.repeat_interleave(x, to_repeat, dim=0)[:self.num_frames]

        return x