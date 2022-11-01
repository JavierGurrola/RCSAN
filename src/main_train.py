import yaml
import torch
import numpy as np
from PIL import Image
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from ptflops import get_model_complexity_info

from model import Net
from train import fit_model
from data_management import SuperResolutionTrainingDataset, SuperResolutionValidationDataset, MultiScaleDataSampler, load_dataset, create_patches
from transforms import RandomRot90, ToTensor, RandomVerticalFlip, RandomHorizontalFlip, RescalePIL, RandomPatches
from utils import set_seed
from metrics import CharbonnierLoss


def main(model_params, train_params, val_params, seed=111):
    r"""
    Prepares the training of the model.
    :param model_params: dict
        Parameters for build the model.
    :param train_params: dict
        Training parameters.
    :param val_params: dict
        Validation parameters.
    :param seed: int
        Seed for random number generators.
    :return:
    """
    set_seed(seed)
    device = torch.device(train_params['device'])
    model = Net(**model_params).to(device)
    scale_range = train_params['scale range']

    param_group = []
    for name, param in model.named_parameters():
        p = {'params': param, 'weight_decay': 0. if 'act' in name else train_params['weight decay']}
        param_group.append(p)

    model.eval()
    if train_params['verbose']:
        with torch.no_grad():
            sample_size = (3, 64, 64)
            macs, params = get_model_complexity_info(model, sample_size)
            message = "Model summary:\n{:<30}  {:<8}\n{:<30}  {:<8}"
            print(message.format('Computational complexity: ', macs, 'Number of parameters: ', params))

    print("Using device: {}".format(device))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print('Using multiple GPUs')

    train_images = load_dataset(train_params['dataset path'], 50, train_params['verbose'])
    val_images = load_dataset(val_params['dataset path'], 5, train_params['verbose'])

    # Create validation patches.
    patch_size = (val_params['patch size'], val_params['patch size'], 3)
    for i, image in enumerate(val_images):
        patches = create_patches(image, patch_size, patch_size)
        val_images[i] = list(patches)
    hr_val_patches, lr_val_patches, scale_factors = sum(val_images, []), [], []
    size_val_patches = (len(hr_val_patches) // val_params['batch size']) * val_params['batch size']
    hr_val_patches = hr_val_patches[:size_val_patches]
    for scale_factor in val_params['scale factors']:
        for hr_patch in hr_val_patches:
            lr_patch = Image.fromarray(hr_patch)
            width, height = lr_patch.size
            lr_patch = lr_patch.resize((width // scale_factor, height // scale_factor), resample=Image.BICUBIC)
            lr_val_patches.append(np.array(lr_patch))
            scale_factors.append(scale_factor)

    val_dataset = {'HR': hr_val_patches, 'LR': lr_val_patches, 'scale': scale_factors}

    train_transforms = transforms.Compose([
        RandomPatches(train_params['patch size']),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRot90(),
        RescalePIL(train_params['patch size']),
        ToTensor()
    ])

    training_dataset = SuperResolutionTrainingDataset(train_images, train_transforms)
    validation_dataset = SuperResolutionValidationDataset(val_dataset, ToTensor())

    training_sampler = MultiScaleDataSampler(training_dataset, train_params['samples per epoch'],
                                             train_params['batch size'], scale_range=scale_range)

    print('Training samples: {}\nValidation samples: {}'.format(len(training_dataset), len(validation_dataset)))
    if isinstance(training_sampler.scale_range, (list, tuple)):
        print('Multi-Scale model.\nMin scale factor:{} - Max scale factor:{}'.format(
            training_sampler.scale_range[0], training_sampler.scale_range[1]
        ))
    else:
        print('Single-Scale model:{}'.format(training_sampler.scale_range))
        train_params['checkpoint path'] += ('_' + str(training_sampler.scale_range))

    data_loaders = {
        'train': DataLoader(
            training_dataset, batch_size=train_params['batch size'],
            num_workers=train_params['workers'], sampler=training_sampler
        ),
        'val': DataLoader(
            validation_dataset, batch_size=val_params['batch size'], shuffle=False, num_workers=train_params['workers']
        )
    }

    # Optimization:
    criterion = CharbonnierLoss()
    optimizer = optim.AdamW(param_group, lr=train_params['learning rate'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, train_params['step decay'], train_params['gamma decay'])

    # Train the model
    fit_model(model, data_loaders, model_params['channels'], criterion, optimizer, lr_scheduler, device,
              train_params['epochs'], val_params['frequency'], train_params['checkpoint path'], 'model',
              train_params['verbose'])


if __name__ == '__main__':
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
        model_params, train_params, val_params = config['model'], config['train'], config['val']

    main(model_params, train_params, val_params)
