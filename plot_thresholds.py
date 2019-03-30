import h5py
from matplotlib import pyplot

from config import Config

if __name__ == "__main__":
    conf = Config()

    # 1) Open the h5 file
    with h5py.File(conf.THRESHOLD_VALS, 'r') as h5t:
        vals = h5t['vals']
        pyplot.plot(h5t['vals'][:, 0], h5t['vals'][:, 1:3])
        pyplot.show()
