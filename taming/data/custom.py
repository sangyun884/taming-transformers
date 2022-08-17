import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
import glob
from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms

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
        im_names_original = glob.glob(os.path.join(dir, "*.png")) + glob.glob(os.path.join(dir, "*.jpg"))
        im_names = im_names_original * mul_length
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
        
class CelebAwithPatches(Dataset):
    def __init__(self, size, patch_size, indices_dir, im_dir, patch_num):
        # size: image size
        # patch_size: the size of patches that will be extracted from the image
        # indices_dir: the directory that contains the indices of the images
        # im_dir: the directory that contains the images. The images will be loaded from im_dir/indices_dir/
        # patch_num: the number of patches that will be extracted from the image

        super().__init__()
        with open(indices_dir, "r") as f:
            self.fnames = f.read().splitlines()
        self.fnames.sort()
        self.dir_dset = im_dir
        self.size = size
        self.patch_size = patch_size
        self.patch_num = patch_num
        assert patch_size < size, f"Patch size should be smaller than image size, but got {patch_size} and {size}"

        # self.rand_perspective = transforms.RandomPerspective(distortion_scale=0.4, p=0.5)
    
    def __getitem__(self, i):
        fname = self.fnames[i]
        img = Image.open(os.path.join(self.dir_dset, fname))
        img = img.resize((self.size, self.size))
        patches = self.__extract_patches_from_image(img)
        img = np.array(img)
        img = img / 127.5 - 1.0
        # print(f"self_patches: {patches}")
        return {'image': img, 'self_patches': patches}
    def __len__(self):
        return len(self.fnames)
    def __extract_patches_from_image(self, img, aug = True):
        # TODO: extract non-overlapping patches from the image
        # Hint: use torch.nn.functional.grid_sample

        img = TF.to_tensor(img)
        size_max = min(self.patch_size * 1.3, self.size - 1)
        size_min = self.patch_size * 0.7
        patches = []
        for i in range(self.patch_num):
            width = np.random.randint(size_min, size_max)
            height = np.random.randint(size_min, size_max)
            x = np.random.randint(0, self.size - width)
            y = np.random.randint(0, self.size - height)
            patch = img[:, y:y+height, x:x+width]
            patch = F.interpolate(patch.unsqueeze(0), size=(self.patch_size, self.patch_size), mode='bicubic')
            if aug:
                patch = self.__patch_augmentation(patch)
            patches.append(patch)
        patches = torch.cat(patches, dim=0)
        return patches
    def __patch_augmentation(self, patch):
        p = 0.5
        # Geometric augmentation
        if np.random.rand() > p:
            patch = TF.hflip(patch)
        if np.random.rand() > p:
            patch = TF.vflip(patch)
        # patch = self.rand_perspective(patch)
        # if np.random.rand() > p:
        #     angle = np.random.randint(-10, 10)
        #     patch = TF.rotate(patch, angle=angle)
        # Color augmentation
        # if np.random.rand() > p:
        #     patch = TF.adjust_brightness(patch, np.random.randint(0.95, 1.05))
        # if np.random.rand() > p:
        #     patch = TF.adjust_contrast(patch, np.random.randint(0.95, 1.05))
        # if np.random.rand() > p:
        #     patch = TF.adjust_saturation(patch, np.random.randint(0.95, 1.05))
        
        return patch
        
    
class CelebAwithPatchesVal(CelebAwithPatches):
    # Same as CelebAwithPatches, but now we mix the patches from an image with patches from another image (sample_patches)
    def __init__(self, size, patch_size, indices_dir, im_dir, patch_num, sample_patch_indice):
        # sample_patch_indice: .txt file that contains the absolute paths to the sample patches
        super().__init__(size, patch_size, indices_dir, im_dir, patch_num)
        with open(sample_patch_indice, "r") as f:
            self.sample_patch_dirs = f.read().splitlines()
        if len(self.sample_patch_dirs) > patch_num:
            raise ValueError(f"{len(self.sample_patch_dirs)} is larger than {patch_num}")
        sample_patch_list = []
        for patch_dir in self.sample_patch_dirs:
            patch = Image.open(patch_dir)
            patch = TF.to_tensor(patch)
            patch = F.interpolate(patch.unsqueeze(0), size=(self.patch_size, self.patch_size), mode='bicubic').squeeze(0)
            sample_patch_list.append(patch)
        self.sample_patch_list = sample_patch_list
    def __getitem__(self, i):
        out = super().__getitem__(i)
        patches = out['self_patches']
        for i, sample_patch in enumerate(self.sample_patch_list):
            patches[i] = sample_patch
        out["self_patches"] = patches
        return out