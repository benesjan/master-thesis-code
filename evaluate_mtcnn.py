import json
import multiprocessing as mp
from os import listdir, path
from sys import exit

import cv2
import h5py
import numpy as np

from config import Config
from data_processing.mtcnn_utils import _bb_intersection_over_union


# video - the opened video object
# frame_id - the frame number
def get_frame(video, frame_id):
    # set the frame position of the videofile to specific frame number
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    video.set(cv2.COLOR_BGR2GRAY, frame_id)
    # read the image from the video
    _, im = video.read()
    return im


def get_file_map(dir_path):
    file_map = {}
    if not path.exists(dir_path):
        return file_map
    files = listdir(dir_path)
    for file in files:
        file_map[file.split('&region')[0]] = file
    return file_map


def process_batch(args):
    batch, thresholds = args
    step = thresholds[1]

    # Array with TP and FN as columns, threshold values as rows
    values = np.zeros((len(thresholds), 2), dtype=np.uint64)

    # 6) Iterate over detections which belong to the name
    for boxes in batch:
        if len(boxes['mtcnn']) == 0:
            #     No intersection, FN for all the thresholds
            values[:, 1] += 1
        else:
            IoU = _bb_intersection_over_union(boxes['mtcnn'], boxes['orig'])
            threshold_i = int(IoU / step)
            # TP
            values[:threshold_i + 1, 0] += 1
            # FN
            values[threshold_i + 1:, 1] += 1

    return values


def get_next(mtcnn_preds, thresholds):
    # 1) Get video names
    batch_size = 1000
    batch = []
    with open(mtcnn_preds, 'r') as mtcnn_file:
        for i, line in enumerate(mtcnn_file):
            boxes = json.loads(line)
            batch.append(boxes)
            if i % batch_size == 0:
                yield batch, thresholds
                batch = []
    yield batch, thresholds


if __name__ == '__main__':
    conf = Config()
    PARALLELIZE = True

    if path.exists(conf.IOU_THRESHOLD_VALS):
        print(f'{conf.IOU_THRESHOLD_VALS} already exists')
        exit(0)

    THRESHOLDS = np.arange(0, 1, 0.005)
    # Array with TP and FN as columns, threshold values as rows
    vals = np.zeros((len(THRESHOLDS), 2), dtype=np.uint64)

    if PARALLELIZE:
        pool = mp.Pool(conf.CPU_COUNT)

        for vals_x in pool.imap(process_batch, get_next(conf.MTCNN_PREDS, THRESHOLDS)):
            vals += vals_x
    else:
        for args in get_next(conf.MTCNN_PREDS, THRESHOLDS):
            vals += process_batch(args)

    # 1) Open the h5 file
    with h5py.File(conf.IOU_THRESHOLD_VALS, 'w') as h5t:
        h5t.create_dataset("vals", data=vals)
        h5t.create_dataset("thresholds", data=THRESHOLDS)
