from env_dependent import DEVICE, DATA_DIR, DATASET_NAME, CPU_COUNT


class Config:

    def __init__(self):
        self.BATCH_SIZE = 200
        self.CPU_COUNT = CPU_COUNT

        self.DEVICE = DEVICE

        self.LOG_DIR = f'{DATA_DIR}/log/'

        self.MODEL_PATH = f'{DATA_DIR}/checkpoints/resnet18_110.pth'

        self.DATASET = f'{DATA_DIR}/datasets/{DATASET_NAME}'

        self.FEATURES = f'{DATA_DIR}/features_{DATASET_NAME}.h5'

        self.THRESHOLD_VALS = f'{DATA_DIR}/thresholds_{DATASET_NAME}.h5'
        self.IOU_THRESHOLD_VALS = f'{DATA_DIR}/thresholds_iou_{DATASET_NAME}.h5'

        # Path to the unprocessed dataset
        self.VIDEO_PATH = f'{DATA_DIR}/datasets/fav_raw/videos/'
        self.ANNOTATIONS_PATH = f'{DATA_DIR}/datasets/fav_raw/CEMI_Udalosti_facetracks_json/'
