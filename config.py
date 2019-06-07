from multiprocessing import cpu_count

import torch

from settings import META_ENV, DATASET_NAME


class Config:

    def __init__(self):
        self.BATCH_SIZE = 60

        # Constant which specifies whether to move the image selection if it crosses the image border
        # If false the frame will be ignored
        self.MOVE_SELECTION = True

        if META_ENV:
            self._set_meta()
        else:
            self._set_home()

    def _set_meta(self):
        home_dir = '/storage/plzen1/home/benesjan/spcdata'

        self.DATASET = f'/storage/plzen1/home/benesjan/datasets/{DATASET_NAME}'

        self.MODEL_PATH = f'{home_dir}/checkpoints/resnet18_110.pth'
        self.DB_PATH = f'{home_dir}/features_merged.h5'
        self.DB_PATH_RAW = f'{home_dir}/features.h5'
        self.DB_PATH_LFW = f'{home_dir}/features_lfw.h5'
        self.LABEL_MAP = f'{home_dir}/label_map.pickle'

        self.THRESHOLD_VALS = f'{home_dir}/threshold_vals.h5'
        self.THRESHOLD_VALS_LFW = f'{home_dir}/threshold_vals_lfw.h5'

        self.PROGRESS_FILE = f'{home_dir}/progress.txt'

        self.DEVICE = torch.device("cuda")
        self.CPU_COUNT = 8

        self.VIDEO_PATH = f'{home_dir}/Faces/videos/'
        self.ANNOTATIONS_PATH = f'{home_dir}/Faces/CEMI-annotations-Udalosti/'

    def _set_home(self):
        dataset_dir = '/home/honza/Data/ML/datasets'

        # Paths to datasets in standard format
        self.DATASET = f'{dataset_dir}/{DATASET_NAME}'

        self.MODEL_PATH = 'checkpoints/resnet18_110.pth'
        self.DB_PATH = 'out/features_merged.h5'
        self.DB_PATH_RAW = 'out/features.h5'
        self.DB_PATH_LFW = 'out/features_lfw.h5'
        self.LABEL_MAP = 'out/label_map.pickle'

        self.THRESHOLD_VALS = 'out/threshold_vals.h5'
        self.THRESHOLD_VALS_LFW = 'out/threshold_vals_lfw.h5'

        self.PROGRESS_FILE = 'out/progress.txt'

        self.DEVICE = torch.device('cpu')
        self.CPU_COUNT = cpu_count()

        # Path to unprocessed FAV dataset
        self.VIDEO_PATH = f'{dataset_dir}/fav_raw/videos/'
        self.ANNOTATIONS_PATH = f'{dataset_dir}/fav_raw/CEMI-annotations-Udalosti/'
