# 1) Compute distance matrix
# 2) Iterate through distances which represent potential threshold
# 3) Evaluate false acceptance rate and false rejection rate and plot them
from time import time

import h5py
import numpy as np
import multiprocessing as mp

from scipy.spatial import distance_matrix
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
    vals = np.zeros((len(THRESHOLDS), 4), dtype=np.uint64)

    labels1 = LABELS[interval_pair[0][0]:interval_pair[0][1]]
    labels2 = LABELS[interval_pair[1][0]:interval_pair[1][1]]

    indices = np.triu_indices_from(dists)

    dists = dists[indices]

    ref_labels = distance_matrix(labels1.reshape(-1, 1), labels2.reshape(-1, 1)) == 0
    ref_labels = ref_labels[indices]

    assert ref_labels.shape == dists.shape, "AssertionError: Dimension mismatch in process_distances"

    for i, threshold in enumerate(THRESHOLDS):
        pred_labels = dists <= threshold

        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
        vals[i, 0] += np.sum(np.logical_and(pred_labels, ref_labels))

        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        vals[i, 1] += np.sum(np.logical_and(pred_labels == False, ref_labels == False))

        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        vals[i, 2] += np.sum(np.logical_and(pred_labels, ref_labels == False))

        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        vals[i, 3] += np.sum(np.logical_and(pred_labels == False, ref_labels))

    return vals


if __name__ == "__main__":
    conf = Config()

    start_time = time()

    # 1) Open the h5 file
    with h5py.File(conf.FEATURES, 'r') as h5f, \
            h5py.File(conf.THRESHOLD_VALS, 'w') as h5t, open(conf.PROGRESS_FILE, 'w', 1) as pf:
        FEATURES = h5f['features']
        LABELS = h5f['labels']

        THRESHOLDS = np.arange(0, 2, 0.005)
        INTERVALS = generate_intervals(FEATURES.shape[0], 10000)

        NUM_PAIRS = len(INTERVALS) * (len(INTERVALS) + 1) / 2

        pool = mp.Pool(conf.CPU_COUNT)

        result = np.zeros((len(THRESHOLDS), 4), dtype=np.uint64)
        for processed_count, res_x in enumerate(pool.imap(process_distances, get_next_pair(INTERVALS)), 1):
            result += res_x

            # Printout
            processing_time = (time() - start_time) / 3600
            percents_processed = processed_count / NUM_PAIRS * 100
            estimated_remaining = (processing_time / percents_processed * 100) - processing_time
            printout = f"{percents_processed}% processed in {processing_time} hours. " \
                f"Estimated remaining time: {estimated_remaining} hours.\n"
            print(printout)
            pf.seek(0)
            pf.write(printout)
            pf.flush()

        h5t.create_dataset("vals", data=result)
        h5t.create_dataset("thresholds", data=THRESHOLDS)
