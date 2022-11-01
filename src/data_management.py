import os
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, Sampler
from skimage.util import view_as_windows
from PIL import Image, ImageFile

from utils import convert_rgb_to_ycbcr

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(path, luma_only=False, patch_size=None, step_size=None):
    r"""
    Loads a single image and converts it in patches if it is necessary.
    :param path: str
        The path of the image.
    :param luma_only: bool
        If luma_only=True, then the image is converted to YCbCr colo space
        and only uses the Y channel.
    :param patch_size: tuple
        The patch size.
    :param step_size: tuple
        The step size at which extraction shall be performed.
    :return: list, numpy ndarray
        if patch_size is not None, then a liste of patches is returned,
        otherwise an array is returned.
    """
    image = np.array(Image.open(path).convert('RGB'))
    if luma_only:
        image = convert_rgb_to_ycbcr(image)[..., 0]
        image = np.expand_dims(image, -1)

    if patch_size is not None:
        return list(create_patches(image, patch_size, step_size))

    return image


def create_patches(image, patch_size, step):
    r"""
    Splits the image sample image into patches.
    :param image: numpy ndarray
        The image to split into patches.
    :param patch_size: tuple
        The patch size.
    :param step: tuple
        The step size at which extraction shall be performed.
    :return: numpy ndarray
        The image split image with shape (n_patches, height, width, channels).
    """
    image = view_as_windows(image, patch_size, step)
    h, w = image.shape[:2]
    return np.reshape(image, (h * w, patch_size[0], patch_size[1], patch_size[2]))


def load_dataset(dataset_folders, limit=None, verbose=True):
    r"""
    Load HR images
    :param dataset_folders: list, tuple
        Path to dataset folders.
    :param limit: int
        Maximum number of images to load (for debugging or reduce validation time).
    :param verbose: bool
        Show progress bar
    :return:
    """
    image_paths, images = [], []
    for folder in dataset_folders:
        image_paths += list(map(lambda file: os.path.join(folder, file), os.listdir(folder)))

    random.shuffle(image_paths)
    if limit:
        image_paths = image_paths[:limit]
    if verbose:
        image_paths = tqdm(image_paths, total=len(image_paths), ncols=50)
    for path in image_paths:
        images.append(load_image(path))

    return images


class MultiScaleDataSampler(Sampler):
    r"""
    Dataset sampler to train the model in sub-epochs,
    it considers the scale factor of the super-resolution images.

    Args:
        data_source (torch Dataset): training dataset.
        num_samples (int): number of samples per epoch (sub-epoch).
        batch_size (int): batch size, necessary value to generate the scale factor for the whole batch
        scale_range (list): scale factors of HR images
        scale_range (list): scale factors of HR images
    """
    def __init__(self, data_source, num_samples=None, batch_size=None, scale_range=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        if num_samples is not None and batch_size is not None:
            self._num_samples = (num_samples // batch_size) * batch_size
        self.scale_range = scale_range
        self.perm = []
        self.scale_perm = []

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)

        if self._num_samples is not None:
            # Generate idx to create batches.
            self.perm, self.scale_perm = [], []
            while len(self.perm) < self._num_samples:
                perm = np.random.permutation(n).astype('int32').tolist()
                self.perm.extend(perm)

            # Generate the same scale factor for the whole batch.
            if len(self.perm) != len(self.scale_perm):
                batches = int(np.ceil(len(self.perm) / self.batch_size))

                if isinstance(self.scale_range, (list, tuple)):     # Multi-scale model
                    batch_scale_factors = np.random.uniform(self.scale_range[0], self.scale_range[1], size=batches)
                else:                                               # Single-scale model
                    batch_scale_factors = self.scale_range * np.ones(batches)

                batch_scale_factors = np.round(batch_scale_factors, decimals=1).tolist()
                batch_scale_factors = np.repeat(batch_scale_factors, repeats=self.batch_size).tolist()
                self.scale_perm.extend(batch_scale_factors)

                # Keep the same length as the index list
                self.scale_perm = self.scale_perm[:len(self.perm)]

            # Keep _num_samples index and scale factors for iterator and update the rest.
            idx = self.perm[:self._num_samples]
            scales = self.scale_perm[:self._num_samples]

            self.perm = self.perm[self._num_samples:]
            self.scale_perm = self.scale_perm[self._num_samples:]
        else:
            idx = np.random.permutation(n).astype('int32').tolist()
            batches = int(np.ceil(n / self.batch_size))
            batch_scale_factors = np.random.choice(self.scale_range, size=batches, replace=True).tolist()
            scales = np.repeat(batch_scale_factors, repeats=self.batch_size).tolist()

        return iter(zip(idx, scales))

    def __len__(self):
        return self.num_samples


class SuperResolutionTrainingDataset(Dataset):
    r"""
    Super-resolution dataset for arbitrary scale.
    Args:
        dataset (list): List of training images.
        transforms (torchvision transform): Transform for data augmentation and tensor conversion.
    """
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transform = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx, scale = idx
        img_hr = self.dataset[idx]
        sample = {'HR': img_hr, 'scale': scale}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class SuperResolutionValidationDataset(Dataset):
    r"""
    Super-resolution dataset with fixed LR and HR images.
    Args:
        dataset (dict): Dictionary with HR, LR and scale_factor lists.
        transforms (torchvision transform): Transform for tensor conversion.
    """
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transform = transforms

    def __len__(self):
        return len(self.dataset['LR'])

    def __getitem__(self, idx):
        idx_hr = idx % len(self.dataset['HR'])
        sample = {
            'LR': Image.fromarray(self.dataset['LR'][idx]),
            'HR': Image.fromarray(self.dataset['HR'][idx_hr]),
            'scale': self.dataset['scale'][idx]
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
