from os import path
import numpy as np
import h5py

from config import DATA_DIR, Config
from find_threshold import compute_values

if __name__ == '__main__':
    conf = Config()

    for N in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        threshold_vals = path.join(DATA_DIR, f'thresholds_fav-128_N{N}.h5')
        if path.exists(threshold_vals):
            print(f"Threshold file for N = {N} already exists on path {threshold_vals}.\nSKIPPING")
            continue

        with h5py.File(conf.FEATURES, 'r') as h5f, h5py.File(threshold_vals, 'w') as h5t:
            FEATURES = h5f['features'][::N]
            LABELS = h5f['labels'][::N]

            THRESHOLDS = np.arange(0, 2, 0.005)
            vals = compute_values(FEATURES, LABELS, THRESHOLDS, conf.CPU_COUNT)

            h5t.create_dataset("vals", data=vals)
            h5t.create_dataset("thresholds", data=THRESHOLDS)
