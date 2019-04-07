from multiprocessing import cpu_count

import torch

from settings import META_ENV


class Config:

    def __init__(self):
        self.BATCH_SIZE = 60
        self.DEVICE = torch.device('cpu')

        # Constant which specifies whether to move the image selection if it crosses the image border
        # If false the frame will be ignored
        self.MOVE_SELECTION = True

        if META_ENV:
            self._set_meta()
        else:
            self._set_home()

    def _set_meta(self):

        home_dir = '/storage/plzen1/home/benesjan/spcdata'

        self.VIDEO_PATH = f'{home_dir}/Faces/videos/'
        self.ANNOTATIONS_PATH = f'{home_dir}/Faces/CEMI-annotations-Udalosti/'

        self.MODEL_PATH = f'{home_dir}/checkpoints/resnet18_110.pth'
        self.DB_PATH = f'{home_dir}/features_merged.h5'
        self.DB_PATH_RAW = f'{home_dir}/features.h5'
        self.LABEL_MAP = f'{home_dir}/label_map.pickle'

        self.THRESHOLD_VALS = f'{home_dir}/threshold_vals.h5'

        self.PROGRESS_FILE = f'{home_dir}/progress.txt'

        self.CPU_COUNT = 8

    def _set_home(self):

        self.VIDEO_PATH = 'data/videos/'
        self.ANNOTATIONS_PATH = 'data/CEMI-annotations-Udalosti/'

        self.MODEL_PATH = 'checkpoints/resnet18_110.pth'
        self.DB_PATH = 'out/features_merged.h5'
        self.DB_PATH_RAW = 'out/features.h5'
        self.LABEL_MAP = 'out/label_map.pickle'

        self.THRESHOLD_VALS = 'out/threshold_vals.h5'

        self.PROGRESS_FILE = 'out/progress.txt'

        self.CPU_COUNT = cpu_count()
