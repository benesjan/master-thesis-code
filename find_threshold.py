# 1) Compute distance matrix
# 2) Iterate through distances which represent potential threshold
# 3) Evaluate false acceptance rate and false rejection rate and plot them
import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import Config

if __name__ == "__main__":
    conf = Config()

    # 1) Open the h5 file
    with h5py.File(conf.DB_PATH_RAW, 'r') as h5f:
        data = h5f['udalosti_-_11.11.2016_19-00_0.mp4'][0:1000]
        names = h5f['udalosti_-_11.11.2016_19-00_0.mp4.names'][0:1000]
        similarities = cosine_similarity(data)
        indices = np.triu_indices(similarities.shape[0])
        threshold_set = set(similarities[indices])
        values = np.zeros((len(threshold_set), 3), dtype=np.float32)

        dst_count = len(indices[0])
        for i, threshold in enumerate(threshold_set):
            values[i, 0] = threshold
            fac = 0
            frc = 0

            for j in range(len(indices[0])):
                coords = (indices[0][j], indices[1][j])
                reference_affinity = (names[coords[0]] == names[coords[1]])
                inferred_affinity = (similarities[coords] <= threshold)

                if inferred_affinity and not reference_affinity:
                    fac += 1
                elif reference_affinity and not inferred_affinity:
                    frc += 1

            values[i, 1] = fac / dst_count
            values[i, 2] = frc / dst_count
