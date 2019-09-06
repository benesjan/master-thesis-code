from multiprocessing import cpu_count
import os
import torch

from settings import META_ENV, DATASET_NAME


class Config:

    def __init__(self):
        self.BATCH_SIZE = 200

        # Constant which specifies whether to move the image selection if it crosses the image border
        # If false the frame will be ignored
        self.MOVE_SELECTION = True

        if META_ENV:
            self._set_meta()
        else:
            self._set_home()

    def _set_meta(self):
        self.DEVICE = torch.device("cuda")
        self.CPU_COUNT = 8

        data_dir = '/storage/plzen1/home/benesjan/spcdata'

        self.MODEL_PATH = f'{data_dir}/checkpoints/resnet18_110.pth'

        self.DATASET = f'{data_dir}/datasets/{DATASET_NAME}'

        self.FEATURES = f'{data_dir}/features_{DATASET_NAME}.h5'

        self.THRESHOLD_VALS = f'{data_dir}/thresholds_{DATASET_NAME}.h5'

        self.VIDEO_PATH = f'{data_dir}/datasets/fav_raw/videos/'
        self.ANNOTATIONS_PATH = f'{data_dir}/datasets/fav_raw/CEMI-annotations-Udalosti/'

    def _set_home(self):
        self.DEVICE = torch.device('cpu')
        self.CPU_COUNT = cpu_count()

        dataset_dir = f'{os.environ["HOME"]}/Data/ML/datasets'

        self.MODEL_PATH = 'checkpoints/resnet18_110.pth'

        # Paths to the datasets in standard format
        self.DATASET = f'{dataset_dir}/{DATASET_NAME}'

        self.FEATURES = f'out/features_{DATASET_NAME}.h5'

        self.THRESHOLD_VALS = f'out/thresholds_{DATASET_NAME}.h5'

        # Path to the unprocessed FAV dataset
        self.VIDEO_PATH = f'{dataset_dir}/fav_raw/videos/'
        self.ANNOTATIONS_PATH = f'{dataset_dir}/fav_raw/CEMI-annotations-Udalosti/'
