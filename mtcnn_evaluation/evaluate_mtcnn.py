import json
from os import path
from sys import exit

import h5py
import numpy as np

from config import Config
from data_processing.mtcnn_utils import bb_intersection_over_union

if __name__ == '__main__':
    conf = Config()

    if path.exists(conf.IOU_THRESHOLD_VALS):
        print(f'{conf.IOU_THRESHOLD_VALS} already exists')
        exit(0)

    STEP = 0.005
    THRESHOLDS = np.arange(0, 1, STEP)
    # Array with TP and FN as columns, threshold values as rows
    vals = np.zeros((len(THRESHOLDS), 2), dtype=np.uint64)

    with open(conf.MTCNN_PREDS, 'r') as mtcnn_file:
        for line in mtcnn_file:
            boxes = json.loads(line)
            if len(boxes['mtcnn']) == 0:
                #     No intersection, FN for all the thresholds
                vals[:, 1] += 1
            else:
                IoU = bb_intersection_over_union(boxes['mtcnn'], boxes['orig'])
                threshold_i = int(IoU / STEP)
                # TP
                vals[:threshold_i + 1, 0] += 1
                # FN
                vals[threshold_i + 1:, 1] += 1

    # 1) Open the h5 file
    with h5py.File(conf.IOU_THRESHOLD_VALS, 'w') as h5t:
        h5t.create_dataset("vals", data=vals)
        h5t.create_dataset("thresholds", data=THRESHOLDS)
