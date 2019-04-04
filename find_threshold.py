# 1) Compute distance matrix
# 2) Iterate through distances which represent potential threshold
# 3) Evaluate false acceptance rate and false rejection rate and plot them

import h5py
import numpy as np
import multiprocessing as mp

from sklearn.metrics.pairwise import cosine_distances

from config import Config


def generate_intervals(max_val, interval_len):
    vals = list(range(0, max_val, interval_len))
    vals.append(max_val)
    return [(vals[i], vals[i + 1]) for i in range(len(vals) - 1)]


def get_next_pair(intervals):
    for i in range(len(intervals)):
        for j in range(i, len(intervals)):
            yield (intervals[i], intervals[j])


def process_distances(interval_pair):
    dists = cosine_distances(FEATURES[interval_pair[0][0]:interval_pair[0][1]],
                             FEATURES[interval_pair[1][0]:interval_pair[1][1]])
    return interval_pair


if __name__ == "__main__":
    conf = Config()

    # 1) Open the h5 file
    with h5py.File(conf.DB_PATH, 'r') as h5f, h5py.File(conf.THRESHOLD_VALS, 'w') as h5t:
        FEATURES = h5f['features']
        NAMES = h5f['names']

        THRESHOLDS = np.arange(0, 1, 0.005)
        INTERVALS = generate_intervals(FEATURES.shape[0], 100000)

        pool = mp.Pool(mp.cpu_count())
        result = pool.map(process_distances, get_next_pair(INTERVALS))
        print(result)
