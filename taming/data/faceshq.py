import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
from PIL import Image
from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
import torch
import torch.nn.functional as F
import glob

class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex


class CelebAHQTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        
        root = "/home/sangyunlee/datasets/CelebAMask-HQ/CelebAMask-HQ/256/train"
        # with open("data/celebahqvalidation.txt", "r") as f:
        #     relpaths = f.read().splitlines()
        # paths = [os.path.join(root, relpath) for relpath in relpaths]
        paths = glob.glob(os.path.join(root, "*.jpg")) + glob.glob(os.path.join(root, "*.png"))
        
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class CelebAHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "/home/sangyunlee/datasets/CelebAMask-HQ/CelebAMask-HQ/256/test"
        # with open("data/celebahqvalidation.txt", "r") as f:
        #     relpaths = f.read().splitlines()
        # paths = [os.path.join(root, relpath) for relpath in relpaths]
        paths = glob.glob(os.path.join(root, "*.jpg")) + glob.glob(os.path.join(root, "*.png"))
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys

class CelebAHQMask(Dataset):
    def __init__(self, size, indices_dir, im_dir, seg_dir):
        super().__init__()
        self.size = size
        self.dir = indices_dir
        self.im_dir = im_dir
        self.seg_dir = seg_dir
        self.im_names = []
        with open(indices_dir, "r") as f:
            self.im_names = f.read().splitlines()
        # if len(self.im_names) < 100:
        #     self.im_names = self.im_names * 10
        self.im_names = self.im_names * 10
        self.label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
    def __getitem__(self, i):
        im_name = self.im_names[i]
        num = im_name.split('.')[0]
        img = Image.open(os.path.join(self.im_dir, im_name))
        img = img.resize((self.size, self.size), Image.BILINEAR)
        img = np.array(img)
        img = img / 127.5 - 1.0

        seg = torch.zeros(img.shape[0], img.shape[1])
        for idx, c in enumerate(self.label_list):
            c_name = f"{num}_{c}.png"
            if not os.path.exists(os.path.join(self.seg_dir, c_name)):
                continue
            c_img = Image.open(os.path.join(self.seg_dir, c_name)).convert('L').resize((self.size, self.size), Image.NEAREST)
            c_np = np.array(c_img)
            seg[c_np != 0] = idx + 1
        seg = F.one_hot(seg.type(torch.long), num_classes=len(self.label_list) + 1).float()
        seg = seg.permute(2, 0, 1)

        return {'image': img, 'seg': seg}
    def __len__(self):
        return len(self.im_names)
class CelebAHQImg(Dataset):
    def __init__(self, size, im_dir):
        super().__init__()
        self.size = size
        self.im_dir = im_dir
        self.im_names  = glob.glob(os.path.join(im_dir, "*.jpg")) + glob.glob(os.path.join(im_dir, "*.png"))

    def __getitem__(self, i):
        im_name = self.im_names[i]
        img = Image.open(os.path.join(self.im_dir, im_name))
        img = img.resize((self.size, self.size), Image.BILINEAR)
        img = np.array(img)
        img = img / 127.5 - 1.0
        return {'image': img}
    def __len__(self):
        return len(self.im_names)

class CelebAHQwithVAE(Dataset):
    def __init__(self, size, im_dir, recon_dir):
        super().__init__()
        self.size = size
        self.im_dir = im_dir
        self.recon_dir = recon_dir
        self.im_names  = glob.glob(os.path.join(im_dir, "*.jpg")) + glob.glob(os.path.join(im_dir, "*.png"))

    def __getitem__(self, i):
        im_name = self.im_names[i].split('/')[-1]
        img = Image.open(os.path.join(self.im_dir, im_name))
        img = img.resize((self.size, self.size), Image.BILINEAR)
        img = np.array(img)
        img = img / 127.5 - 1.0

        recon = Image.open(os.path.join(self.recon_dir, im_name))
        recon = recon.resize((self.size//4, self.size//4), Image.BILINEAR)
        recon = np.array(recon)
        recon = recon / 127.5 - 1.0
        return {'image': img, 'LR_image': recon}
    def __len__(self):
        return len(self.im_names)
 

class FFHQTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FFHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FacesHQTrain(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQTrain(size=size, keys=keys)
        d2 = FFHQTrain(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex


class FacesHQValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQValidation(size=size, keys=keys)
        d2 = FFHQValidation(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex
