import random
import torchvision.transforms.functional as F
from PIL import Image


class RescalePIL(object):
    r"""
    Rescales images considering the training patch size and the size of the input image.
    """
    def __init__(self, patch_size):
        self.patch_size = (patch_size, patch_size)

    def __call__(self, sample):
        img_hr, scale = sample['HR'], sample['scale']
        sample['LR'] = img_hr.resize(self.patch_size, resample=Image.BICUBIC)
        return sample


class ToTensor(object):
    r"""
    Converts a sample of uint8 numpy ndarrays with range [0, 255]
    to float32 torch tensors with range [0., 1.], it also transposes
    the images from shape (H, W, C) to (C, H, W).
    """
    def __call__(self, sample):
        img_hr, img_lr = sample['HR'], sample['LR']
        sample['target size'] = img_hr.size[1], img_hr.size[0]
        sample['HR'] = F.to_tensor(img_hr)
        sample['LR'] = F.to_tensor(img_lr)

        return sample


class _RandomTransform(object):
    r"""
    Applies a random transform to the dataset sample.
    Args:
        transform (function): PIL transpose transformation function
        p (float): probability to apply the random transformation
    """
    def __init__(self, transform, p):
        self.p = p
        self.transform = transform

    def __call__(self, sample):
        if random.random() < self.p:
            img_hr, img_lr = sample['HR'], sample.get('LR', None)
            sample['HR'] = img_hr.transpose(self.transform)
            if img_lr is not None:
                sample['LR'] = img_lr.transpose(self.transform)
        
        return sample


class RandomVerticalFlip(_RandomTransform):
    def __init__(self, p=0.5):
        super().__init__(Image.FLIP_TOP_BOTTOM, p)


class RandomHorizontalFlip(_RandomTransform):
    def __init__(self, p=0.5):
        super().__init__(Image.FLIP_LEFT_RIGHT, p)


class RandomRot90(_RandomTransform):
    def __init__(self, p=0.5):
        super().__init__(Image.ROTATE_90, p)


class RandomPatches(object):
    r"""
    Transformation for generate random HR patches.
    """
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, sample):
        image, scale = sample['HR'], sample['scale']
        h, w = image.shape[:2]

        h_patch, w_patch = round(scale * self.patch_size), round(scale * self.patch_size)

        top, left = random.randint(0, h - h_patch), random.randint(0, w - w_patch)
        bottom, right = top + h_patch, left + w_patch

        image_patch = image[top: bottom, left: right, ...]
        sample['HR'] = Image.fromarray(image_patch)

        return sample
