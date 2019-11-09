import glob
import json
import multiprocessing as mp
from os import listdir, path
from sys import exit
from time import time

import cv2
import h5py
import numpy as np
from mtcnn import detect_faces

from config import Config
from data_processing.mtcnn_utils import get_bbox_i_by_IoU, _bb_intersection_over_union


# video - the opened video object
# frame_id - the frame number
def get_frame(video, frame_id):
    # set the frame position of the videofile to specific frame number
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    video.set(cv2.COLOR_BGR2GRAY, frame_id)
    # read the image from the video
    _, im = video.read()
    return im


def process_video(args):
    video_name, thresholds = args
    step = thresholds[1]
    print(f'Processing {video_name}')

    # Array with TP and FN as columns, threshold values as rows
    values = np.zeros((len(thresholds), 2), dtype=np.uint64)

    # Used in file name
    video_date = video_name.split('_')[2]

    # 2) Load the annotations
    with open(conf.ANNOTATIONS_PATH + video_name + '_people.json', 'r') as f:
        annotations = json.load(f)

    # 3) load the video
    video = cv2.VideoCapture(conf.VIDEO_PATH + video_name)

    # 4) check if the video file opened successfully, if not continue with another one
    if not video.isOpened():
        print(f'The videofile {video_name} could not be opened!')
        return values

    num_of_names = len(annotations.keys())
    # 5) Iterate over names
    for i, name in enumerate(annotations.keys()):
        print(f'{video_name}, {i + 1}/{num_of_names} {name}')
        try:
            # 6) Iterate over detections which belong to the name
            for detection in annotations[name]['detections']:

                frame = detection['frame']
                rect = detection['rect']

                name_formatted = name.replace(' ', '_')
                image_name = f'name={name_formatted}&video_date={video_date}&frame={frame}&region*'
                image_path = path.join(conf.DATASET, name_formatted, image_name)
                fm = glob.glob(image_path)
                if len(fm) == 1:
                    # Image already exists
                    image_path = fm[0]
                    region = image_path.split('&region=')[1][:-4].split('_')
                    bboxes = [[float(x) for x in region]]
                else:
                    # Load the frame
                    im = get_frame(video, frame)

                    # 8) Get bboxes and landmarks
                    bboxes, _ = detect_faces(im)

                bbox_i = get_bbox_i_by_IoU(bboxes, rect, threshold=0)
                if bbox_i == -1:
                    #     No intersection, FN for all the thresholds
                    values[:, 1] += 1
                else:
                    bbox = bboxes[bbox_i]
                    IoU = _bb_intersection_over_union(bbox, rect)
                    threshold_i = int(IoU / step)
                    # TP
                    values[:threshold_i + 1, 0] += 1
                    # FN
                    values[threshold_i + 1:, 1] += 1

        except Exception as e:
            print(f'An error occurred when processing image of {name}\n{e}')

    return values


def get_next(thresholds, video_path):
    # 1) Get video names
    videos = listdir(video_path)

    for video_name in videos:
        yield video_name, thresholds


if __name__ == '__main__':
    conf = Config()

    if path.exists(conf.IOU_THRESHOLD_VALS):
        print(f'{conf.IOU_THRESHOLD_VALS} already exists')
        exit(0)

    THRESHOLDS = np.arange(0, 1, 0.005)
    # Array with TP and FN as columns, threshold values as rows
    vals = np.zeros((len(THRESHOLDS), 2), dtype=np.uint64)

    pool = mp.Pool(conf.CPU_COUNT)
    start_time = time()

    for vals_x in pool.imap(process_video, get_next(THRESHOLDS, conf.VIDEO_PATH)):
        vals += vals_x

    # 1) Open the h5 file
    with h5py.File(conf.IOU_THRESHOLD_VALS, 'w') as h5t:
        h5t.create_dataset("vals", data=vals)
        h5t.create_dataset("thresholds", data=THRESHOLDS)
