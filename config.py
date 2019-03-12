import torch


class Config:
    BATCH_SIZE = 60

    DEVICE = torch.device("cpu")

    VIDEO_PATH = "data/videos/"
    ANNOTATIONS_PATH = "data/CEMI-annotations-Udalosti/"

    MODEL_PATH = 'checkpoints/resnet18_110.pth'
    DB_PATH = 'out/features.h5'
