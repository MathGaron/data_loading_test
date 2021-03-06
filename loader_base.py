from torch.utils.data.dataset import Dataset
from abc import ABCMeta, abstractmethod


class LoaderBase(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self, root, data_transforms=[], target_transforms=[]):
        self.indexes = self.index_dataset(root)
        self.root = root
        self.data_transforms = data_transforms
        self.target_transforms = target_transforms

    @abstractmethod
    def index_dataset(self, dir):
        """
        Should return a list of samples (index)
        :param path:
        :param class_to_idx:
        :return:
        """
        pass

    @abstractmethod
    def from_index(self, index):
        """
        should return a tuple : (data, target)
        :param index:
        :return:
        """
        pass

    def __getitem__(self, index):
        data, target = self.from_index(index)
        if not isinstance(data, list):
            data = [data]

        # apply data augmentation
        for i, transform in enumerate(self.data_transforms):
            if transform is not None:
                data[i] = transform(data[i])
        for i, transform in enumerate(self.target_transforms):
            if transform is not None:
                target[i] = transform(target[i])
        return data, target

    def __len__(self):
        return len(self.indexes)
