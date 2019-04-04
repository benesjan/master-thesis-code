# 1) Compute distance matrix
# 2) Iterate through distances which represent potential threshold
# 3) Evaluate false acceptance rate and false rejection rate and plot them

import h5py
import numpy as np

from config import Config


def generate_intervals(max_val, interval_len):
    vals = list(range(0, max_val, interval_len))
    vals.append(max_val)
    return [(vals[i], vals[i + 1]) for i in range(len(vals) - 1)]


def get_next_pair(intervals):
    for i in range(len(intervals)):
        for j in range(i, len(intervals)):
            yield [intervals[i], intervals[j]]


if __name__ == "__main__":
    conf = Config()

    # 1) Open the h5 file
    with h5py.File(conf.DB_PATH, 'r') as h5f, h5py.File(conf.THRESHOLD_VALS, 'w') as h5t:
        FEATURES = h5f['features']
        NAMES = h5f['names']

        THRESHOLDS = np.arange(0, 1, 0.005)

        intervals = generate_intervals(FEATURES.shape[0], 100000)
        for pair in get_next_pair(intervals):
            print(pair)
