import os
import os.path
import glob
import torch.utils.data as data
from PIL import Image


class TinyImageNet(data.Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `valid` subdirectories.
    subset: string
        Indicating which subset(split) to return as a data set.
        Valid option: [`train`, `test`, `valid`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    """
    def __init__(self, root, subset='train', transform=None):
        self.root = os.path.expanduser(root)
        self.subset = subset
        self.transform = transform
        self.subset_dir = os.path.join(root, self.subset)
        self.image_paths = sorted(glob.iglob(self.subset_dir + '/**/*.JPEG', recursive=True))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img) if self.transform else img

        return 


if __name__ == '__main__':

    tiny_train = TinyImageNet('./dataset', subset='train')
    print(len(tiny_train))
    print(tiny_train.__getitem__(99999))



