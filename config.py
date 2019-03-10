import torch


class Config:
    batch_size = 60

    device = torch.device("cpu")

    video_path = "/media/honza/My Passport/Faces/videos/"
    annotations_path = "/media/honza/My Passport/Faces/CEMI-annotations-Udalosti/"

    model_path = 'checkpoints/resnet18_110.pth'
    output_dir_path = 'out/'
