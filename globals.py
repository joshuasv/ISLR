import os
import mediapipe as mp

root = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_ROOT_DPATH = os.path.join(root, 'data')
DATA_RAW_DPATH = os.path.join(DATA_ROOT_DPATH, 'raw')
DATA_PREPROCESSED_DPATH = os.path.join(DATA_ROOT_DPATH, 'preprocessed')
CONFIG_DPATH = os.path.join(root, 'config')
LOGS_DPATH = os.path.join(root, 'logs')

# Data
RAW_CSV_FNAME = 'train.csv'
EXTENDED_RAW_CSV_FNAME = 'edata.csv'
JSON_SIGN_TO_PRED = 'sign_to_prediction_index_map.json'
RAW_DATA_DPATH = os.path.join(DATA_RAW_DPATH, 'train_landmark_files')
CSV_INDEX = 'sequence_id'
CSV_ROW_ID = 'row_id'
VOCAB_SIZE = 250

# Landmarks
FACE_N_LANDMARKS = 468
POSE_N_LANDMARKS = 33
RHAND_N_LANDMARKS = 21
LHAND_N_LANDMARKS = 21
N_LANDMARKS = FACE_N_LANDMARKS + POSE_N_LANDMARKS + RHAND_N_LANDMARKS + LHAND_N_LANDMARKS # 543
N_LANDMARKS_MAP = {
    'face': FACE_N_LANDMARKS,
    'lhand': LHAND_N_LANDMARKS,
    'pose': POSE_N_LANDMARKS,
    'rhand': RHAND_N_LANDMARKS
}

# Contigous array access
FACE_OFFSET = 0
LHAND_OFFSET = FACE_N_LANDMARKS
POSE_OFFSET = LHAND_OFFSET + LHAND_N_LANDMARKS
RHAND_OFFSET = POSE_OFFSET + POSE_N_LANDMARKS
LANDMARK_OFFSETS_MAP = {
    'face': FACE_OFFSET,
    'lhand': LHAND_OFFSET,
    'pose': POSE_OFFSET,
    'rhand': RHAND_OFFSET
}

# Face landmarks groupings
FACE_REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 
             161, 246]
FACE_LEYE = [362, 382, 380, 381, 374, 373, 390, 249, 263, 466, 388, 387, 386,
             385, 384, 398]
FACE_LIPS = [0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
             291, 409, 270, 269, 267, 13, 82, 81, 80, 191, 78, 95, 88, 178, 87,
             14, 317, 402, 318, 324, 308, 415, 310, 311, 312]
FACE_LIPS_COMPLETE = [
        0, 11, 12, 13, 37, 72, 38, 82, 39, 73, 41, 81, 40, 74, 42, 80, 185,
        184, 183, 191, 61, 76, 62, 78, 146, 77, 96, 95, 91, 90, 89, 88,
        181, 180, 179, 178, 84, 85, 86, 87, 17, 16, 15, 14, 314, 315, 316,
        317, 405, 404, 403, 402, 321, 320, 319, 318, 375, 307, 325, 324,
        308, 292, 306, 291, 415, 407, 408, 409, 310, 272, 304, 270, 311,
        271, 303, 269, 312, 268, 302, 267]
# Pose landmarks groupings
POSE_UPPER = list(range(0, 23))
POSE_LOWER = list(range(23, 33))

# Connections
# FACE_CONNECTIONS[198] and FACE_CONNECTIONS[420] are connecting the same 
# landmarks twice. This causes the neighbors to have duplicates, need to filter
# them. Probably many more exist...
FACE_CONNECTIONS = list(mp.solutions.face_mesh.FACEMESH_TESSELATION)
HAND_CONNECTIONS = list(mp.solutions.hands.HAND_CONNECTIONS)
POSE_CONNECTIONS = list(mp.solutions.pose.POSE_CONNECTIONS)