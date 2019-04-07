import pickle

import h5py
import numpy as np

from config import Config

if __name__ == "__main__":
    conf = Config()

    # 3) Open the h5 file
    with h5py.File(conf.DB_PATH_RAW, 'r') as h5f_raw, \
            h5py.File(conf.DB_PATH, 'w') as h5f, open(conf.LABEL_MAP, 'wb') as pf:

        feature_keys = [x for x in h5f_raw.keys() if x.endswith("mp4")]

        features = np.concatenate([h5f_raw[x] for x in feature_keys])
        h5f.create_dataset("features", data=features)

        labels = np.zeros(features.shape[0], dtype=np.int16) - 1

        label_map = {}
        i = 0
        max_label_val = 0
        for key in feature_keys:
            name_key = key + '.names'
            for name_label in h5f_raw[name_key]:
                if name_label not in label_map:
                    max_label_val += 1
                    label_map[name_label] = max_label_val
                labels[i] = label_map[name_label]
                i += 1

        assert -1 not in labels, "AssertionError: not all label positions assigned"

        h5f.create_dataset("labels", data=labels)
        pickle.dump(label_map, pf, protocol=pickle.HIGHEST_PROTOCOL)
