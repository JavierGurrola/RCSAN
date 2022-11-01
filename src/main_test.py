import yaml
import torch
import numpy as np
import os

from skimage import io
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from PIL import Image

from model import Net
from utils import correct_model_dict, build_ensemble, split_ensemble, convert_rgb_to_ycbcr, mod_crop, predict_ensemble


def get_image_metrics(image_true, image_test, data_range=1.):
    r"""
    Get the performance metrics.
    :param image_true: ndarray
        Ground-truth image.
    :param image_test: ndarray
        Predicted image.
    :param data_range: float
        Range of values of the reference image.
    :return:
    """
    psnr = peak_signal_noise_ratio(image_true, image_test, data_range=data_range)
    ssim = structural_similarity(image_true, image_test, data_range=data_range, multichannel=image_true.ndim == 3,
                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

    return psnr, ssim


def predict(model, dataset, device, results_path, shave_border=True, scale=2):
    r"""
    Predict the images HR images from dataset.
    :param model: torch Module
        Pretrained model used to generate the HR images.
    :param dataset: dict
        Dataset with the HR (ground-truth) images and LR images.
    :param device: torch device
        Device used during the inference (CPU/GPU).
    :param results_path: str
        Path where the predicted images will be stored.
    :param shave_border: bool
        Method to remove border artifacts.
    :param scale: int
        Scale factor used in the prediction.
    :return: dic
        Dictionary with the average of evaluation metrics.
    """
    n_images = len(dataset['HR'])
    images_pred, images_pred_ens = [], []
    psnr_list, ssim_list, psnr_ens_list, ssim_ens_list = [], [], [], []

    for i in range(n_images):
        image_hr, image_lr = dataset['HR'][i], dataset['LR'][i]

        if image_hr.ndim == 3:
            y_hr = convert_rgb_to_ycbcr(image_hr)[..., 0]
        else:
            y_hr = image_hr.copy()
            image_lr = np.repeat(np.expand_dims(image_lr, -1), 3, -1)

        ensemble, target_sizes = build_ensemble(image_lr, image_hr.shape[:2], normalize=True)
        pred = predict_ensemble(model, ensemble, target_sizes, device)
        pred_ens, pred = split_ensemble(pred, return_single=True)

        # Evaluate PSNR and SSIM in shaved image
        if shave_border:
            pred_ens_eval, pred_eval = pred_ens[scale:-scale, scale:-scale], pred[scale:-scale, scale:-scale]
            y_hr_eval = y_hr[scale:-scale, scale:-scale]
        else:
            pred_ens_eval, pred_eval = pred_ens.copy(), pred.copy()
            y_hr_eval = y_hr.copy()

        if image_hr.ndim == 3:
            pred_ens_eval = convert_rgb_to_ycbcr(255 * pred_ens_eval)[..., 0]
            pred_eval = convert_rgb_to_ycbcr(255 * pred_eval)[..., 0]
        else:
            pred_ens_eval = 255 * np.mean(pred_ens_eval, axis=-1)
            pred_eval = 255 * np.mean(pred_eval, axis=-1)

        psnr, ssim = get_image_metrics(y_hr_eval, pred_eval, 255)
        psnr_ens, ssim_ens = get_image_metrics(y_hr_eval, pred_ens_eval, 255)

        psnr_list.append(psnr)
        psnr_ens_list.append(psnr_ens)
        ssim_list.append(ssim)
        ssim_ens_list.append(ssim_ens)

        message = 'Image:{} - PSNR:{:.4f} - SSIM:{:.4f} - ens PSNR:{:.4f} - ens SSIM:{:.4f}'
        print(message.format(i + 1, psnr, ssim, psnr_ens, ssim_ens))

        if results_path is not None:
            pred_ens, pred = np.clip(np.round(255 * pred_ens), 0., 255.), np.clip(np.round(255 * pred), 0., 255.)
            pred_ens, pred = pred_ens.astype('uint8'), pred.astype('uint8')
            images_pred_ens.append(pred_ens)
            images_pred.append(pred)

    if results_path is not None:
        os.makedirs(results_path, exist_ok=True)
        for i in range(n_images):
            pred_ens, pred = np.squeeze(images_pred_ens[i]), np.squeeze(images_pred[i])
            name = os.path.join(results_path, '{}_{:.4f}_{:.4f}.png'.format(i + 1, psnr_list[i], ssim_list[i]))
            io.imsave(name, pred)

            name = os.path.join(results_path, '{}_{:.4f}_{:.4f}_ens.png'.format(i + 1, psnr_ens_list[i], ssim_ens_list[i]))
            io.imsave(name, pred_ens)

    return {
        'PSNR': np.mean(psnr_list), 'SSIM': np.mean(ssim_list),
        'ens PSNR': np.mean(psnr_ens_list), 'ens SSIM': np.mean(ssim_ens_list),
    }


if __name__ == '__main__':
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    model_params, test_params = config['model'], config['test']
    model_path = os.path.join(test_params['model path'], 'model_{}.pth')
    device = torch.device(test_params['device'])

    datasets = test_params['test datasets']
    scale_factors = test_params['scale factors']
    model = None

    for scale_factor in scale_factors:
        print('Scale factor: ', scale_factor)
        if model is not None:
            del model

        if test_params['multi scale']:
            state_dict = torch.load(model_path.format('all'), map_location=device)
        else:
            state_dict = torch.load(model_path.format('x' + str(scale_factor)), map_location=device)
        state_dict = correct_model_dict(state_dict)
        model = Net(**model_params)
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()

        for dataset in datasets:
            print('Dataset: ', dataset)
            dataset_path = os.path.join(test_params['dataset path'], dataset)
            img_files = sorted(os.listdir(dataset_path))
            hr_images, lr_images = [], []

            for img_file in img_files:
                img_file = os.path.join(dataset_path, img_file)
                hr_image = np.array(Image.open(img_file))
                hr_image = Image.fromarray(mod_crop(hr_image, scale_factor))
                w, h = hr_image.size
                w, h = w // scale_factor, h // scale_factor
                lr_image = hr_image.resize((w, h), resample=Image.BICUBIC)

                hr_images.append(np.array(hr_image))
                lr_images.append(np.array(lr_image))

            dataset_images = {'HR': hr_images, 'LR': lr_images}
            results_path = os.path.join(test_params['results path'], dataset, 'X' + str(scale_factor)) if test_params['save images'] else None
            metrics = predict(model, dataset_images, device, results_path, shave_border=True, scale=scale_factor)
            message = 'TOTAL: - PSNR:{:.4f} - SSIM:{:.4f} - ens PSNR:{:.4f} - ens SSIM:{:.4f}\n'

            print(message.format(metrics['PSNR'], metrics['SSIM'], metrics['ens PSNR'], metrics['ens SSIM']))
