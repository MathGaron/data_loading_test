import numpy as np
from PIL import Image

from loader_base import LoaderBase
import os


class LoaderNumpy(LoaderBase):
    def __init__(self, root, data_transform=[], target_transform=[]):
        super(LoaderNumpy, self).__init__(root, data_transform, target_transform)

    def index_dataset(self, dir):
        # get all png files, without depth
        files = [x for x in os.listdir(dir) if ".npy" in x]
        return files

    def from_index(self, index):
        file = self.indexes[index]
        # load a sample
        frame = np.load(os.path.join(self.root, file))
        depth = self.numpy_uint8_to_int16(frame[:, :, 3:])
        rgb = frame[:, :, 0:3]

        # convert to float32 (usually necessary for deep learning models
        rgb = rgb.astype(np.float32)
        depth = depth.astype(np.float32)

        # None as a target for this demo code
        target = 1
        return (rgb, depth), [target]

    @staticmethod
    def numpy_uint8_to_int16(depth8):
        x, y, c = depth8.shape
        out = np.ndarray((x, y), dtype=np.int16)
        out[:, :] = depth8[:, :, 0]
        out = np.left_shift(out, 8)
        out[:, :] += depth8[:, :, 1]
        return out
