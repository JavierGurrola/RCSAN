import random
import torch
import numpy as np
from collections import OrderedDict


def correct_model_dict(state_dict):
    r"""
    Rename the model components if it was trained using multiple GPUs.
    :param state_dict: dict
        Torch model state dict
    :return: dict
        Renamed torch model state dict
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value

    return new_state_dict


def mod_crop(img, mod):
    r"""
    Crops img to fit the int scale factor
    :param img: ndarray
        Image to crop.
    :param mod: int
        Divisor used to crop the image.
    :return: ndarray
        Cropped image.
    """
    size = img.shape[:2]
    size = size - np.mod(size, mod)
    img = img[:size[0], :size[1], ...]

    return img


def set_seed(seed=1):
    r"""
    Set seeds for all random number generators.
    :param seed: int
        Seed value for random number generators.
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_ensemble(image, target_size, normalize=True):
    r"""
    Generate images using flips and 90° rotation for ensemble estimation.
    :param image: ndarray
        Image to transform.
    :param target_size: tuple, list
        Target height and width.
    :param normalize: bool
        Convert image from [0, 255] range to [0, 1].
    :return:
    """
    if normalize:
        image = image / 255.
    # Flipped versions.
    ensemble = [image, np.fliplr(image), np.flipud(image), np.flipud(np.fliplr(image))]
    # Flipped versions of rotated image.
    img_rot = np.rot90(image)
    img_rot = [img_rot, np.fliplr(img_rot), np.flipud(img_rot), np.flipud(np.fliplr(img_rot))]
    ensemble.extend(img_rot)

    for i, image in enumerate(ensemble):
        image = np.transpose(image.copy(), (2, 0, 1))               # Channels-first transposition.
        image = torch.from_numpy(np.expand_dims(image, 0)).float()  # Expand dims to create batch dimension.
        ensemble[i] = image

    target_sizes = [target_size] * 4 + [[target_size[1], target_size[0]]] * 4

    return ensemble, target_sizes


def split_ensemble(ensemble, return_single=False):
    r"""
    Reconstruct the original image using estimations in the image ensemble.
    :param ensemble: list
        List with the estimated ndarray images.
    :param return_single: bool
        Return also the first image of the ensemble as single image estimation.
    :return: ndarray, tuple
        If return_single=False returns the ensemble estimation only, else returns ensemble and single image estimation.
    """
    for i, image in enumerate(ensemble):
        image = image.squeeze()
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
        ensemble[i] = image

    # Vertical and Horizontal Flips
    image = ensemble[0] + np.fliplr(ensemble[1]) + np.flipud(ensemble[2]) + np.fliplr(np.flipud(ensemble[3]))

    # 90º Rotation, Vertical and Horizontal Flips
    image = image + np.rot90(ensemble[4], k=3) + np.rot90(np.fliplr(ensemble[5]), k=3)
    image = image + np.rot90(np.flipud(ensemble[6]), k=3) + np.rot90(np.fliplr(np.flipud(ensemble[7])), k=3)
    image = np.clip(image / 8., 0., 1.)

    if return_single:
        return image, np.clip(ensemble[0], 0., 1.)
    return image


def predict_ensemble(model, ensemble, target_sizes, device):
    r"""
    Predicts each image in the ensemble.
    :param model: torch Module
        Model used to estimate the HR images.
    :param ensemble: list
        ndarray images to estimate.
    :param target_sizes: list
        (height, width) target sizes.
    :param device: torch device
        Device used for the estimation (CPU/GPU).
    :return: list
        Estimated ndarray images.
    """
    pred_ensemble = []
    for x, t in zip(ensemble, target_sizes):
        x = x.to(device)
        t = torch.tensor(t)
        with torch.no_grad():
            pred = model(x, target_size=[t]).detach().cpu().numpy().astype('float32')
            torch.cuda.empty_cache()
            pred_ensemble.append(pred)

    return pred_ensemble


def convert_rgb_to_ycbcr(img):
    r"""
    Convert ndarray image from RGB space color to YCbCr space color.
    :param img: ndarray
        Image to convert.
    :return:
    """
    if type(img) == np.ndarray:
        y = 16. + (65.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (65.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))
