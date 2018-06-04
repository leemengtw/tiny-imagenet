import os
import os.path
import glob
import torch.utils.data as data
from PIL import Image

EXTENSION = '.JPEG'
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = 'wnids.txt'
VAL_ANNOTATION_FILE = 'val_annotations.txt'

class TinyImageNet(data.Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    subset: string
        Indicating which subset(split) to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    """
    def __init__(self, root, subset='train', transform=None):
        self.root = os.path.expanduser(root)
        self.subset = subset
        self.transform = transform
        self.subset_dir = os.path.join(root, self.subset)
        self.image_paths = sorted(glob.iglob(self.subset_dir + '/**/*' + EXTENSION, recursive=True))
        self.labels = {} # fname - label number mapping

        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.replace('\n', '') for _, text in enumerate(fp)])
            self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.subset == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels['_'.join((label_text, str(cnt))) + EXTENSION] = i
        elif self.subset == 'val':
            with open(os.path.join(self.subset_dir, VAL_ANNOTATION_FILE), 'r') as fp:
                for _, line in enumerate(fp):
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]
        img = Image.open(file_path)
        img = self.transform(img) if self.transform else img

        if self.subset == 'test':
            return img
        else:
            file_name = file_path.split('/')[-1]
            print(file_name)
            return img, self.labels[file_name]


if __name__ == '__main__':

    tiny_train = TinyImageNet('./dataset', subset='train')
    print(len(tiny_train))
    print(tiny_train.__getitem__(99999))
    for fname, number in tiny_train.labels.items():
        if number == 192:
            print(fname, number)

    tiny_train = TinyImageNet('./dataset', subset='val')
    print(tiny_train.__getitem__(99))



