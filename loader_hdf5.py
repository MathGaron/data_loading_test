import numpy as np
import h5py

from loader_base import LoaderBase
import os
import h5py


class LoaderHdf5(LoaderBase):
    def __init__(self, root, data_transform=[], target_transform=[]):
        super(LoaderHdf5, self).__init__(root, data_transform, target_transform)

    def index_dataset(self, dir):
        # get all name without depth
        data_file = h5py.File(os.path.join(dir, "data.hdf5"), 'r')
        data_keys = list(data_file.keys())
        data_file.close()
        data_keys = [x for x in data_keys if x[-1] != "d"]
        return data_keys

    def from_index(self, index):
        data_file = h5py.File(os.path.join(self.root, "data.hdf5"), 'r')
        data_key = self.indexes[index]
        # load a sample
        rgb = data_file[data_key][:]
        depth = data_file[data_key+"d"][:]

        data_file.close()

        # convert to float32 (usually necessary for deep learning models
        rgb = rgb.astype(np.float32)
        depth = depth.astype(np.float32)

        # None as a target for this demo code
        target = 1
        return (rgb, depth), [target]

    #def __del__(self):
    #    self.data_file.close()
