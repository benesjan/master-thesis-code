from os.path import join

import h5py
from matplotlib import pyplot
import numpy as np

from config import Config, DATASET_NAME
from utils import create_dir


def compute_prfo(thresholds, vals):
    prf = np.zeros((thresholds.shape[0], 3))
    optimal_val = [0, 0]
    for i, t in enumerate(thresholds):
        TP, TN, FP, FN = vals[i]
        # Precision
        prf[i, 0] = TP / (TP + FP)
        # Recall
        prf[i, 1] = TP / (TP + FN)
        # F1 score
        prf[i, 2] = 2 * (prf[i, 0] * prf[i, 1]) / (prf[i, 0] + prf[i, 1])
        if prf[i, 2] > optimal_val[1]:
            optimal_val = [t, prf[i, 2]]
    return prf, optimal_val


if __name__ == "__main__":
    conf = Config()

    target_dir = 'referat/out'

    create_dir(target_dir)

    # 1) Open the h5 file
    with h5py.File(conf.THRESHOLD_VALS, 'r') as h5t:
        # TP, TN, FP, FN
        thresholds = h5t['thresholds']
        vals = h5t['vals']

        # Plot precision, recall, F1
        prf, optimal_val = compute_prfo(thresholds, vals)

        fig = pyplot.figure(1)
        pyplot.grid()
        pyplot.plot(thresholds, prf)

        pyplot.scatter(optimal_val[0], optimal_val[1], marker="x", s=300, linewidth=1.3, c='purple')
        pyplot.annotate('[%.2f, %.2f]' % (optimal_val[0], optimal_val[1]),
                        [optimal_val[0] + 0.1, optimal_val[1]])

        pyplot.legend(['Precision', 'Recall', 'F1'])
        pyplot.xlim(thresholds[0], thresholds[-1])
        # pyplot.ylim([0, 1])

        pyplot.xlabel('Threshold')
        pyplot.show()
        fig.savefig(join(target_dir, f'prft_{DATASET_NAME}.eps'), format='eps')
