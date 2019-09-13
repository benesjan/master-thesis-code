import env_dependent


class Config:

    def __init__(self):
        self.BATCH_SIZE = 200
        self.CPU_COUNT = 8

        self.DEVICE = env_dependent.DEVICE

        self.MODEL_PATH = f'{env_dependent.DATA_DIR}/checkpoints/resnet18_110.pth'

        self.DATASET = f'{env_dependent.DATA_DIR}/datasets/{env_dependent.DATASET_NAME}'

        self.FEATURES = f'{env_dependent.DATA_DIR}/features_{env_dependent.DATASET_NAME}.h5'

        self.THRESHOLD_VALS = f'{env_dependent.DATA_DIR}/thresholds_{env_dependent.DATASET_NAME}.h5'

        # Path to the unprocessed dataset
        self.VIDEO_PATH = f'{env_dependent.DATA_DIR}/datasets/fav_raw/videos/'
        self.ANNOTATIONS_PATH = f'{env_dependent.DATA_DIR}/datasets/fav_raw/CEMI-annotations-Udalosti/'
