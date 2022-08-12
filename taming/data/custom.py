import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
import glob
from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from PIL import Image
import torch.nn.functional as F
import torch

class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class SmallDatasetSeg(Dataset):
    # Used for Bedroom-28, FFHQ-34, Cat-15, Horse-21, CelebA-19, ADE-Bedroom-30
    # This dataset loads an image (.png or .jpg) and annotation (.npy)
    # Resolution is fixed to 256 x 256.
    # For more information, refer to https://github.com/yandex-research/ddpm-segmentation.
    def __init__(self, dir, num_classes, mul_length = 1):
        super().__init__()
        im_names = glob.glob(os.path.join(dir, "*.png")) + glob.glob(os.path.join(dir, "*.jpg")) * mul_length
        im_names.sort()
        seg_names = [im_name.replace(".png", ".npy").replace(".jpg", ".npy") for im_name in im_names]

        self.im_names = im_names
        self.seg_names = seg_names
        self.num_classes = num_classes
    def __getitem__(self, i):
        im_name = self.im_names[i]
        seg_name = self.seg_names[i]
        assert os.path.exists(im_name)
        assert os.path.exists(seg_name)

        img = Image.open(im_name)
        img = np.array(img)
        img = img / 127.5 - 1.0

        seg = torch.tensor(np.load(seg_name), dtype=torch.long)
        seg = F.one_hot(seg, num_classes= self.num_classes).type(torch.float)
        seg = seg.permute(2, 0, 1)

        return {'image': img, 'seg': seg}

    def __len__(self):
        return len(self.im_names)
        
