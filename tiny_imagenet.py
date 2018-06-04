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
        return len(self.subset.samples)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return 'I dont know'


if __name__ == '__main__':


    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])



    tiny_test = TinyImageNet('./dataset', subset='test', transform=transform)
    print(tiny_test.__len__())

    import torch
    loader = torch.utils.data.DataLoader(tiny_test, batch_size=128,
                                         shuffle=True, num_workers=4)
    dataiter = iter(loader)
    images, labels = dataiter.next()