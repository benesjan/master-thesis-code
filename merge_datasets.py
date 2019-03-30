import h5py
import numpy as np

from config import Config

if __name__ == "__main__":
    conf = Config()

    # 3) Open the h5 file
    with h5py.File(conf.DB_PATH_RAW, 'r') as h5f_raw, h5py.File(conf.DB_PATH, 'w') as h5f:
        for dataset_name in h5f_raw.keys():
            features = np.concatenate([h5f_raw[x] for x in h5f_raw.keys() if x.endswith("mp4")])
            names = np.concatenate([h5f_raw[x] for x in h5f_raw.keys() if x.endswith("names")])

            h5f.create_dataset("features", data=features)
            h5f.create_dataset("names", data=names, dtype=h5py.special_dtype(vlen=str))
