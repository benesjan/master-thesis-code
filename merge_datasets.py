import h5py

from config import Config

if __name__ == "__main__":
    conf = Config()

    # 3) Open the h5 file
    with h5py.File(conf.DB_PATH_RAW, 'r') as h5f_raw, h5py.File(conf.DB_PATH, 'w') as h5f:
        pass