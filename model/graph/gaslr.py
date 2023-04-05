import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))

import numpy as np

import globals
from model.graph import tools


def get_neighbors(conns_in):
    toret = []
    conns_out = {k: np.flip(v, axis=1) for k, v in conns_in.items()}
    for k, v in conns_in.items():
        toret.extend(list(map(tuple, v)) + list(map(tuple, conns_out[k])))

    # Some connections  duplicated because some mediapipe connections are 
    # defined twice, e.g. (0, 37) and (37, 0)...
    return list(set(toret))

def get_num_nodes(conns):
    toret = 0
    for _, v in conns.items():
        toret += len(np.unique(v))

    return toret

def replace_conn_vals_to_conn_idxs(conns_vals, conns_idxs):
    toret = {}
    for k, v in conns_vals.items():
        unique_vals, idxs = conns_idxs[k]
        indices = np.searchsorted(unique_vals, v)
        toret[k] = idxs[indices]

    return toret

def map_conns_to_idx(conns):
    offset = 0
    toret = {}
    for k, v in conns.items():
        conn_vals = np.unique(v)
        n_landmakrs = len(conn_vals)
        idxs = np.array(range(offset, offset + n_landmakrs))
        toret[k] = (conn_vals, idxs)
        offset += n_landmakrs

    return toret

# def get_landmarks_idxs(landmarks_to_include):
#     face = landmarks_to_include['face']
#     pose = landmarks_to_include['pose']

#     face_idxs = []
#     if face == 'ALL':
#         face_idxs = range(globals.FACE_N_LANDMARKS)
#     else:
#         for var in face:
#             face_idxs.extend(globals.__dict__[var])
#     pose_idxs = []
#     if pose == 'ALL':
#         pose_idxs = range(globals.POSE_N_LANDMARKS)
#     else:
#         for var in pose:
#             pose_idxs.extend(globals.__dict__[var])

#     face_idxs = np.array(sorted(face_idxs))
#     pose_idxs = np.array(sorted(pose_idxs))
#     rhand_idxs = np.array(range(globals.RHAND_N_LANDMARKS))
#     lhand_idxs = np.array(range(globals.LHAND_N_LANDMARKS))

#     return {'face': face_idxs, 'pose': pose_idxs, 'rhand': rhand_idxs, 'lhand': lhand_idxs}


def get_landmarks_idxs(landmarks_to_include):
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
    toret = {}
    for lmark, to_include in landmarks_to_include.items():
        if to_include is None:
            continue
        else:
            offset = globals.LANDMARK_OFFSETS_MAP[lmark]
            n_landmarks = globals.N_LANDMARKS_MAP[lmark]
            idxs = []
            for var in to_include:
                if var == 'ALL':
                    idxs.extend(list(range(n_landmarks)))
                else:
                    idxs.extend([v for v in globals.__dict__[var]])
            toret[lmark] = idxs

    return toret

    # return sorted(idxs)


def filter_conns(landmarks_to_include):
    toret = {}
    all_landmarks = get_landmarks_idxs(landmarks_to_include)
    for lname, landmarks in all_landmarks.items():
        if lname == 'face':
            conns = globals.FACE_CONNECTIONS
        elif lname == 'pose':
            conns = globals.POSE_CONNECTIONS
        elif lname == 'rhand' or lname == 'lhand':
            conns = globals.HAND_CONNECTIONS
        else:
            raise ValueError(f"lanme {lname} not an option")
        conns = np.array(conns)
        conns_to_include = np.isin(conns, landmarks).sum(1) == 2
        toret[lname] = conns[conns_to_include]

    return toret


class AdjacencyMatrix:

    def __init__(self, landmarks_to_include):
        conns_in = filter_conns(landmarks_to_include)
        num_nodes = get_num_nodes(conns_in)
        conns_in_idxs = map_conns_to_idx(conns_in)
        conns_in = replace_conn_vals_to_conn_idxs(conns_in, conns_in_idxs)

        # from IPython import embed
        # from sys import exit
        # embed(); exit();
        
        self.edges = get_neighbors(conns_in)
        self.num_nodes = num_nodes
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)


if __name__ == '__main__':
    import yaml
    import matplotlib.pyplot as plt
    with open('config/test.yaml', 'r') as fid:
        config = yaml.safe_load(fid)
    landmarks_to_include = config['landmarks_to_include']
    graph = AdjacencyMatrix(landmarks_to_include)
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(A_binary_with_I, cmap='gray')
    ax[1].imshow(A_binary, cmap='gray')
    ax[2].imshow(A, cmap='gray')
    ax[0].set_title('A_binary_with_I')
    ax[1].set_title('A_binary')
    ax[2].set_title('A')
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)
