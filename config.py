import torch


class Config:
    BATCH_SIZE = 60

    DEVICE = torch.device("cpu")

    VIDEO_PATH = "data/videos/"
    ANNOTATIONS_PATH = "data/CEMI-annotations-Udalosti/"

    MODEL_PATH = 'checkpoints/resnet18_110.pth'
    DB_PATH = 'out/features_merged.h5'
    DB_PATH_RAW = 'out/features.h5'
    THRESHOLD_VALS = 'out/threshold_vals.h5'

    # Constant which specifies whether to move the image selection if it crosses the image border
    # If false the frame will be ignored
    MOVE_SELECTION = True
