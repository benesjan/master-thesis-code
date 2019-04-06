from multiprocessing import cpu_count

import torch


class Config:
    BATCH_SIZE = 60

    DEVICE = torch.device('cpu')

    VIDEO_PATH = 'data/videos/'
    ANNOTATIONS_PATH = 'data/CEMI-annotations-Udalosti/'

    MODEL_PATH = 'checkpoints/resnet18_110.pth'
    DB_PATH = 'out/features_merged.h5'
    DB_PATH_RAW = 'out/features.h5'
    THRESHOLD_VALS = 'out/threshold_vals.h5'

    # Constant which specifies whether to move the image selection if it crosses the image border
    # If false the frame will be ignored
    MOVE_SELECTION = True

    PROGRESS_FILE = 'out/progress.txt'

    CPU_COUNT = cpu_count()


class ConfigMeta:
    BATCH_SIZE = 60

    DEVICE = torch.device('cpu')

    HOME_DIR = '/storage/plzen1/home/benesjan'

    VIDEO_PATH = f'{HOME_DIR}/Faces/videos/'
    ANNOTATIONS_PATH = f'{HOME_DIR}/Faces/CEMI-annotations-Udalosti/'

    MODEL_PATH = f'{HOME_DIR}/spc/checkpoints/resnet18_110.pth'
    DB_PATH = f'{HOME_DIR}/features_merged.h5'
    DB_PATH_RAW = f'{HOME_DIR}/features.h5'
    THRESHOLD_VALS = f'{HOME_DIR}/threshold_vals.h5'

    MOVE_SELECTION = True

    PROGRESS_FILE = f'{HOME_DIR}/progress.txt'

    CPU_COUNT = 8
