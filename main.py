import json
from os import listdir

import cv2
import h5py
import numpy as np
import torch
from torch.nn import DataParallel

from config import Config
from models.resnet import resnet_face18


# make the selection square
def get_square(rect, im_res):
    selection_height = rect[3] - rect[1]
    # make sure the selection height is even
    if selection_height & 1:
        selection_height -= 1
        rect[3] -= 1
    width_middle = (rect[0] + rect[2]) / 2
    left = int(width_middle - (selection_height / 2))
    right = int(width_middle + (selection_height / 2))
    # make sure the selection is within the image boundaries
    if left < 0:
        left = 0
        right = selection_height
    elif right > im_res[1]:
        left = im_res[1] - selection_height
        right = selection_height
    return left, rect[1], right, rect[3]


# returns an image with selected face
# video - the opened video object
# frame - the frame number
# rect - the rectangle of the face
def get_face(video, frame, rect):
    # set the frame position of the videofile to specific frame number
    video.set(cv2.CAP_PROP_POS_FRAMES, frame)
    video.set(cv2.COLOR_BGR2GRAY, frame)
    # read the image from the video
    ret, im = video.read()

    square = get_square(rect, im.shape)

    return im[square[1]:square[3], square[0]:square[2], :]


# Process image so that it can be fed to NN
def process_face(im):
    im = cv2.resize(im, (128, 128))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # The following lines are copied from arcface-pytorch project
    # Stack image and it's flipped version. Output dimensions: (128, 128, 2)
    im = np.dstack((im, np.fliplr(im)))
    # Transpose. Output dimensions: (2, 128, 128)
    im = im.transpose((2, 0, 1))
    # Add dimension. Output dimensions: (2, 1, 128, 128)
    im = im[:, np.newaxis, :, :]
    im = im.astype(np.float32, copy=False)
    # Normalize to <-1, 1>
    im -= 127.5
    im /= 127.5
    return im


def get_next_image(video, annotations):
    for name in annotations.keys():
        for detection in annotations[name]['detections']:
            # 5) Get the cropped image
            frame = detection['frame']
            rect = detection['rect']

            im = get_face(video, frame, rect)
            im = process_face(im)
            yield name, im


# Get features for 1 batch
def predict(model, images, features, device):
    data = torch.from_numpy(images)
    data = data.to(conf.DEVICE)
    output = model(data)
    output = output.data.cpu().numpy()

    fe_1 = output[::2]
    fe_2 = output[1::2]
    feature = np.hstack((fe_1, fe_2))

    if features is None:
        features = feature
    else:
        features = np.vstack((features, feature))
    return features


# get features for the whole video
def get_features(model, video, annotations, batch_size, device):
    images = None
    features = None
    names = []
    for name, image in get_next_image(video, annotations):
        # Save the name
        names.append(name)

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0:
            features = predict(model, images, features, device)
            images = None

    # Process any remaining images
    if images:
        features = predict(model, images, features, device)

    return features, np.array(names, dtype=object)


if __name__ == "__main__":
    conf = Config()

    # 1) Load model
    model = resnet_face18(False)
    model = DataParallel(model)

    model.load_state_dict(torch.load(conf.MODEL_PATH, map_location=conf.DEVICE))
    model.to(conf.DEVICE)

    model.eval()

    # 2) Get video names
    videos = listdir(conf.VIDEO_PATH)

    # 3) Open the h5 file
    with h5py.File(conf.DB_PATH, 'a') as h5f:

        for video_name in videos:
            if video_name in h5f:
                print(f'OMITTING {video_name}: features are already present in the database')
                continue

            print(f"Processing {video_name}")

            # 3) Load the annotations
            with open(conf.ANNOTATIONS_PATH + video_name + "_people.json", "r") as f:
                annotations = json.load(f)

            # 4) load the video
            video = cv2.VideoCapture(conf.VIDEO_PATH + video_name)

            # 5) check if the video file opened successfully, if not continue with another one
            if not video.isOpened():
                print(f'The videofile {video_name} could not be opened!')
                continue

            # 6) get the features
            features, names = get_features(model, video, annotations, batch_size=conf.BATCH_SIZE, device=conf.DEVICE)

            # 7) save the features and names
            h5f.create_dataset(video_name, data=features)
            h5f.create_dataset(video_name + ".names", data=names, dtype=h5py.special_dtype(vlen=str))

            print(f"Video {video_name} processed")
