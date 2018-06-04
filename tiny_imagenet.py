import os
import os.path
import torch.utils.data as data
import torchvision.datasets as datasets


class TinyImageNet(data.Dataset):
    """
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `valid`.
    subset: string
        `train`, `test`, `valid`
    """

    def __init__(self, root, subset='train', transform=None):
        self.root = os.path.expanduser(root)
        self.subset = subset
        self.transform = transform
        self.subset_dir = os.path.join(root, self.subset)
        self.subset = datasets.ImageFolder(self.subset_dir, self.transform)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data"""

if __name__ == '__main__':
    tiny_train = TinyImageNet('./dataset', subset='train')
    print(tiny_train)