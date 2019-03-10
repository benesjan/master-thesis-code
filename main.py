import json
from os import listdir

import cv2
import torch
from torch.nn import DataParallel

from models.resnet import resnet_face18

video_path = "/media/honza/My Passport/Faces/videos/"
annotations_path = "/media/honza/My Passport/Faces/CEMI-annotations-Udalosti/"

model_path = 'checkpoints/resnet18_110.pth'


# returns an image with selected face
# video - the opened video object
# frame - the frame number
# rect - the rectangle of the face
def getFace(video, frame, rect, border=0.0):
    # set the frame position of the videofile to specific frame number
    video.set(cv2.CAP_PROP_POS_FRAMES, frame)
    # read the image from the video
    ret, im = video.read()

    return im[rect[1]:rect[3], rect[0]:rect[2], :]


if __name__ == "__main__":
    # 0) Load model
    model = resnet_face18(False)
    model = DataParallel(model)

    device = torch.device("cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 1) Get video names
    videos = listdir(video_path)

    for video_name in videos:
        # 2) Load the annotations
        with open(annotations_path + video_name + "_people.json", "r") as f:
            data = json.load(f)

        # 3) load the video
        video = cv2.VideoCapture(video_path + video_name)

        # 4) check if the video file opened successfully, if not continue with another one
        if not video.isOpened():
            print(f'The videofile {video_name} could not be opened!')
            continue

        for name in data.keys():
            # 5) Get the cropped image
            frame = data[name]['detections'][0]['frame']
            rect = data[name]['detections'][0]['rect']

            im = getFace(video, frame, rect)

            # 6) Get prediction
            print(im)

        break
