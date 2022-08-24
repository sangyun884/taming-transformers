import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
import glob
from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
import h5py
import cv2
from torchvision.utils import save_image
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
    def __extract_patches_from_image(self, img,):
        img = TF.to_tensor(img)
        patches = []
        _, h, w = img.shape
        x, y = 0, 0
        delta_x = w // 3
        delta_y = h // 3
        while x + delta_x < w:
            while y + delta_y < h:
                patch = img[:, y:y+delta_y, x:x+delta_x]
                patch = F.interpolate(patch.unsqueeze(0), size=self.patch_size, mode='bicubic')
                # patch = self.__patch_augmentation(patch)
                patches.append(patch)
                y += delta_y
            x += delta_x
            y = 0
        np.random.shuffle(patches)
        patches = patches[:self.patch_num]
        patches = torch.cat(patches, dim=0)
        return patches
        
    def __extract_patches_from_image_crop(self, img, aug = True):
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

class ArtBenchWithPatches(Dataset):
    def __init__(self, patch_size, im_dir):
        size = 256
        # patch_size: the size of patches that will be extracted from the image
        # im_dir: the directory that contains images.

        super().__init__()
        self.im_list = []
        for root, dirs, files in os.walk(im_dir):
            for f in files:
                if f.endswith(".jpg") or f.endswith(".png"):
                    self.im_list.append(os.path.join(root, f))
        self.size = size
        self.patch_size = patch_size
        assert patch_size < size, f"Patch size should be smaller than image size, but got {patch_size} and {size}"

        # self.rand_perspective = transforms.RandomPerspective(distortion_scale=0.4, p=0.5)
    
    def __getitem__(self, i):
        img = Image.open(self.im_list[i])
        assert img.size == (self.size, self.size), f"{img.size} is not equal to {self.size}"
        patches = self.__extract_patches_from_image(img)
        img = np.array(img)
        img = img / 127.5 - 1.0
        patches = (patches - 0.5) / 0.5
        return {'image': img, 'self_patches': patches}
    def __len__(self):
        return len(self.im_list)
    def __extract_patches_from_image(self, img,):
        patch_num = np.random.randint(1, 16)
        img = TF.to_tensor(img)
        patches = []
        _, h, w = img.shape
        x, y = 0, 0
        delta_x_list = self._divide_range_into_intervals(start = x, end = w, num = 4)
        delta_y_list = self._divide_range_into_intervals(start = y, end = h, num = 4)
        for delta_x in delta_x_list: 
            for delta_y in delta_y_list:
                patch = img[:, y:y+delta_y, x:x+delta_x]
                patch = F.interpolate(patch.unsqueeze(0), size=self.patch_size, mode='bicubic')
                patch = self._patch_augmentation(patch)
                patches.append(patch)
                y += delta_y
            x += delta_x
            y = 0
        np.random.shuffle(patches)
        patches = patches[:patch_num]
        patches = torch.cat(patches, dim=0)
        return patches
    def _divide_range_into_intervals(self, start, end, num):
        """
        Divide the range [start, end] into num intervals.
        Return: a list of intervals.
        """


        delta_min = (end - start) / num / 2
        delta_list = []
        for i in range(num - 1):
            delta_max = (end - start) / (num - i + 1) * 2
            delta = np.random.randint(delta_min, delta_max)
            delta_list.append(delta)
            start += delta
        delta_list.append(end - start)
        return delta_list
    def _patch_augmentation(self, patch):
        p = 0.5
        if np.random.rand() > p:
            patch = TF.rotate(patch, angle=np.random.randint(-10, 10))        
        return patch
  

class NYUv2Depth(Dataset):
    f = None
    def __init__(self, dir, size = 256, train=True):
        # dir: path to .mat file
        super().__init__()
        if NYUv2Depth.f is None:
            print(f"Loading {dir}")
            NYUv2Depth.f = h5py.File(dir) 
        self.images = NYUv2Depth.f['images']
        self.depths = NYUv2Depth.f['depths']
        self.size = size
        if train:
            self.images = self.images[:1200]
            self.depths = self.depths[:1200]
        else:
            self.images = self.images[1200:]
            self.depths = self.depths[1200:]

    def __getitem__(self, i):
        # read i-th image. original format is [3 x 640 x 480], uint8
        img = self.images[i]

        # reshape
        img_ = np.empty([480, 640, 3])
        img_[:,:,0] = img[0,:,:].T
        img_[:,:,1] = img[1,:,:].T
        img_[:,:,2] = img[2,:,:].T

        # imshow
        img = img_.astype('float32') / 127.5 - 1.0


        depth = torch.tensor(self.depths[i].astype(np.float32).T).unsqueeze(0)

        # Random crop into square
        h, w = depth.shape[1], depth.shape[2]
        if h > w:
            crop_size = w
            top = np.random.randint(0, h - crop_size)
            left = 0
        else:
            crop_size = h
            top = 0
            left = np.random.randint(0, w - crop_size)
        depth_cropped = depth[:, top:top+crop_size, left:left+crop_size]
        img_cropped = img[top:top+crop_size, left:left+crop_size, :]
        
        # Resize
        depth_resized = F.interpolate(depth_cropped.unsqueeze(0), size=(self.size, self.size), mode='bilinear').squeeze(0)
        img_resized = cv2.resize(img_cropped, (self.size, self.size))

        depth_resized = (depth_resized - 3) / 3
        
        return {'image': img_resized, 'seg': depth_resized}
    def __len__(self):
        return len(self.images)