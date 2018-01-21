import numpy as np
from PIL import Image

from loader_base import LoaderBase
import os


class LoaderPng(LoaderBase):
    def __init__(self, root, data_transform=[], target_transform=[]):
        super(LoaderPng, self).__init__(root, data_transform, target_transform)

    def index_dataset(self, dir):
        # get all png files, without depth
        files = [x.split(".")[0] for x in os.listdir(dir) if ".png" in x and "d.png" not in x]
        return files

    def from_index(self, index):
        file = self.indexes[index]
        # load a sample
        rgb = np.array(Image.open(os.path.join(self.root, file + ".png")))
        depth = np.array(Image.open(os.path.join(self.root, file + "d.png"))).astype(np.uint16)

        # convert to float32 (usually necessary for deep learning models
        rgb = rgb.astype(np.float32)
        depth = depth.astype(np.float32)

        # None as a target for this demo code
        target = 1
        return (rgb, depth), [target]
