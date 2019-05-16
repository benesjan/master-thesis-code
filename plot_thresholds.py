import h5py
from matplotlib import pyplot

from config import Config

if __name__ == "__main__":
    conf = Config()

    # 1) Open the h5 file
    with h5py.File(conf.THRESHOLD_VALS, 'r') as h5t:
        # TP, TN, FP, FN
        thresholds = h5t['thresholds']
        vals = h5t['vals']
        pyplot.plot(thresholds, vals)
        pyplot.legend(['TP', 'TN', 'FP', 'FN'])
        pyplot.xlabel('Threshold')
        pyplot.xlim([thresholds[0], thresholds[-1]])
        pyplot.show()
