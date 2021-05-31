import numpy as np
import os
import os.path
from PIL import Image


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fromlist(image_list, label_list):
    with open(image_list) as f:
        image_index = f.readlines()
    label_arr = np.loadtxt(label_list)

    return image_index, label_arr


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


class Imagelists_VISDA(object):
    def __init__(self, image_list, label_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False):
        imgs, labels = make_dataset_fromlist(image_list, label_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index][:-1])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


class Imagelists_VISDA_twice(Imagelists_VISDA):

    def __init__(self, image_list, label_list, root="./data/multi/", transform=None):
        super(Imagelists_VISDA_twice, self).__init__(image_list, label_list, root=root, transform=transform)
