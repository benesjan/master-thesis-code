# 1) Compute distance matrix
# 2) Iterate through distances which represent potential threshold
# 3) Evaluate false acceptance rate and false rejection rate and plot them
from os import path
from time import time

import h5py
import numpy as np
import multiprocessing as mp

from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_distances

from config import Config


def _generate_intervals(max_val, interval_len):
    interval_vals = list(range(0, max_val, interval_len))
    interval_vals.append(max_val)
    return [(interval_vals[i], interval_vals[i + 1]) for i in range(len(interval_vals) - 1)]


def _get_next(intervals, features, labels, thresholds):
    for i in range(len(intervals)):
        for j in range(i, len(intervals)):
            yield (features[intervals[i][0]:intervals[i][1]],
                   features[intervals[j][0]:intervals[j][1]],
                   labels[intervals[i][0]:intervals[i][1]],
                   labels[intervals[j][0]:intervals[j][1]],
                   thresholds)


def _process_distances(args):
    # Arguments not expanded directly in the function call because of the limitations of pool.imap method
    features1, features2, labels1, labels2, thresholds = args

    dists = cosine_distances(features1, features2)
    new_vals = np.zeros((len(thresholds), 4), dtype=np.uint64)

    indices = np.triu_indices_from(dists)

    dists = dists[indices]

    ref_labels = distance_matrix(labels1.reshape(-1, 1), labels2.reshape(-1, 1)) == 0
    ref_labels = ref_labels[indices]

    assert ref_labels.shape == dists.shape, "AssertionError: Dimension mismatch in process_distances"

    for i, threshold in enumerate(thresholds):
        pred_labels = dists <= threshold

        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
        new_vals[i, 0] += np.sum(np.logical_and(pred_labels, ref_labels))

        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        new_vals[i, 1] += np.sum(np.logical_and(pred_labels == False, ref_labels == False))

        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        new_vals[i, 2] += np.sum(np.logical_and(pred_labels, ref_labels == False))

        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        new_vals[i, 3] += np.sum(np.logical_and(pred_labels == False, ref_labels))

    return new_vals


def compute_values(features, labels, thresholds, num_process=4, interval_length=4000):
    # Function which computes TP, TN, FP, FN
    intervals = _generate_intervals(features.shape[0], interval_length)
    num_pairs = len(intervals) * (len(intervals) + 1) / 2
    pool = mp.Pool(num_process)
    start_time = time()

    values = np.zeros((len(thresholds), 4), dtype=np.uint64)
    for processed_count, vals_x in enumerate(
            pool.imap(_process_distances, _get_next(intervals, features, labels, thresholds)), 1):
        values += vals_x

        # Printout
        processing_time = (time() - start_time) / 3600
        percents_processed = processed_count / num_pairs * 100
        estimated_remaining = (processing_time / percents_processed * 100) - processing_time
        print(f"{percents_processed}% processed in {processing_time} hours. "
              f"Estimated remaining time: {estimated_remaining} hours.\n")

    return values


if __name__ == "__main__":
    conf = Config()

    if path.exists(conf.THRESHOLD_VALS):
        print(f"Target file containing values already exists on path {conf.THRESHOLD_VALS}.\nEXITING")
    else:
        # 1) Open the h5 file
        with h5py.File(conf.FEATURES, 'r') as h5f, h5py.File(conf.THRESHOLD_VALS, 'w') as h5t:
            FEATURES = h5f['features']
            LABELS = h5f['labels']

            THRESHOLDS = np.arange(0, 2, 0.005)
            vals = compute_values(FEATURES, LABELS, THRESHOLDS, conf.CPU_COUNT)

            h5t.create_dataset("vals", data=vals)
            h5t.create_dataset("thresholds", data=THRESHOLDS)
