from os.path import join

import h5py
import numpy as np
from matplotlib import pyplot

from config import Config, DATASET_NAME
from utils import create_dir

if __name__ == "__main__":
    conf = Config()
    target_dir = 'out'
    chosen_threshold = 0.5

    create_dir(target_dir)

    # 1) Open the h5 file
    with h5py.File(conf.IOU_THRESHOLD_VALS, 'r') as h5t:
        # TP, TN, FP, FN
        thresholds = h5t['thresholds']
        vals = h5t['vals']

        recall = np.zeros(thresholds.shape[0])
        for i, t in enumerate(thresholds):
            TP, FN = vals[i]
            # Recall
            recall[i] = TP / (TP + FN)

        fig = pyplot.figure(1)
        pyplot.grid()
        pyplot.plot(thresholds, recall)

        # Highlight chosen threshold value
        threshold_val = [chosen_threshold, recall[np.where(thresholds[:] == chosen_threshold)[0][0]]]

        pyplot.scatter(threshold_val[0], threshold_val[1], marker="x", s=300, linewidth=1.3, c='purple')
        pyplot.annotate('[%.2f, %.2f]' % (threshold_val[0], threshold_val[1]),
                        [threshold_val[0] + 0.1, threshold_val[1]])

        pyplot.legend(['Recall'])
        pyplot.xlim(thresholds[0], thresholds[-1])
        pyplot.ylim([0, 1])

        pyplot.xlabel('Threshold')
        pyplot.show()
        fig.savefig(join(target_dir, f'mtcnn_recall_{DATASET_NAME}.eps'), format='eps')
