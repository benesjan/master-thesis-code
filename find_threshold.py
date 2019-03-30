# 1) Compute distance matrix
# 2) Iterate through distances which represent potential threshold
# 3) Evaluate false acceptance rate and false rejection rate and plot them
from time import time

import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import Config

if __name__ == "__main__":
    conf = Config()

    start_time = time()

    # 1) Open the h5 file
    with h5py.File(conf.DB_PATH, 'r') as h5f, h5py.File(conf.THRESHOLD_VALS, 'w') as h5t:
        data = h5f['features']
        names = h5f['names']
        similarities = cosine_similarity(data)
        indices = np.triu_indices(similarities.shape[0])
        # threshold_set = set(similarities[indices])

        thresholds = np.arange(-1, 1, 0.01)

        vals = np.zeros((len(thresholds), 3), dtype=np.float32)

        dst_count = len(indices[0])
        for i, threshold in enumerate(thresholds):
            vals[i, 0] = threshold
            fac = 0.
            frc = 0.

            for j in range(len(indices[0])):
                coords = (indices[0][j], indices[1][j])
                reference_affinity = (names[coords[0]] == names[coords[1]])
                inferred_affinity = (similarities[coords] <= threshold)
                if inferred_affinity and not reference_affinity:
                    fac += 1
                elif reference_affinity and not inferred_affinity:
                    frc += 1

            vals[i, 1] = fac / dst_count
            vals[i, 2] = frc / dst_count

        h5t.create_dataset("vals", data=vals)

    print(f"Processing finished in {(time() - start_time) / 60} minutes")
