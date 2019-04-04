# 1) Compute distance matrix
# 2) Iterate through distances which represent potential threshold
# 3) Evaluate false acceptance rate and false rejection rate and plot them
from time import time

import h5py
import numpy as np
import multiprocessing as mp

from sklearn.metrics.pairwise import cosine_distances

from config import *


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
    vals = np.zeros((len(THRESHOLDS), 4), dtype=np.uint32)

    names1 = NAMES[interval_pair[0][0]:interval_pair[0][1]]
    names2 = NAMES[interval_pair[1][0]:interval_pair[1][1]]

    for i, threshold in enumerate(THRESHOLDS):
        # Iterate through upper triangular matrix
        for j in range(0, dists.shape[0]):
            for k in range(j + 1, dists.shape[1]):
                inferred_affinity = (dists[j, k] <= threshold)
                reference_affinity = (names1[j] == names2[k])

                if inferred_affinity and reference_affinity:
                    # True positive
                    vals[i, 0] += 1
                elif not inferred_affinity and not reference_affinity:
                    # True negative
                    vals[i, 1] += 1
                elif inferred_affinity and not reference_affinity:
                    # False positive +=1
                    vals[i, 2] += 1
                else:
                    # False negative
                    vals[i, 3] += 1

    return vals


if __name__ == "__main__":
    conf = ConfigMeta()

    start_time = time()

    # 1) Open the h5 file
    with h5py.File(conf.DB_PATH, 'r') as h5f, \
            h5py.File(conf.THRESHOLD_VALS, 'w') as h5t, open(conf.PROGRESS_FILE, 'w', 1) as pf:
        FEATURES = h5f['features']
        NAMES = h5f['names']

        THRESHOLDS = np.arange(0, 1, 0.005)
        INTERVALS = generate_intervals(FEATURES.shape[0], 1000)

        NUM_PAIRS = len(INTERVALS) * (len(INTERVALS) + 1) / 2

        pool = mp.Pool(mp.cpu_count())

        result = np.zeros((len(THRESHOLDS), 4), dtype=np.uint32)
        for processed_count, res_x in enumerate(pool.imap(process_distances, get_next_pair(INTERVALS)), 1):
            result += res_x

            # Printout
            processing_time = (time() - start_time) / 60
            percents_processed = processed_count / NUM_PAIRS * 100
            estimated_remaining = (processing_time / percents_processed * 100) - processing_time
            printout = f"{percents_processed}% processed in {processing_time} minutes. " \
                f"Estimated remaining time: {estimated_remaining} minutes.\n"
            pf.seek(0)
            pf.write(printout)
            pf.flush()

        h5t.create_dataset("vals", data=result)
        h5t.create_dataset("thresholds", data=THRESHOLDS)
