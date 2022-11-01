import csv
import os
import torch
import time
import numpy as np
from tqdm import tqdm
from metrics import PSNR, SSIM


class EpochLogger:
    r"""
    Keeps a log of metrics in the current epoch.
    """
    def __init__(self):
        self.log = {'train loss': 0, 'train psnr': 0, 'train ssim': 0, 'val loss': 0, 'val psnr': 0, 'val ssim': 0}

    def update_log(self, metrics, phase):
        for key, value in metrics.items():
            self.log[''.join([phase, ' ', key])] += value

    def get_log(self, num_samples, phase):
        log = {
            phase + ' loss': self.log[phase + ' loss'] / num_samples[phase],
            phase + ' psnr': self.log[phase + ' psnr'] / num_samples[phase],
            phase + ' ssim': self.log[phase + ' ssim'] / num_samples[phase]
        }
        return log


class FileLogger(object):
    r"""
    Keeps a log of the whole training and validation process.
    The results are recorded in a CSV files.
    Args:
        file_path (string): path of the csv file.
        header (list): header of the csv file.
    """
    def __init__(self, file_path, header):
        self.file_path = file_path
        with open(self.file_path, 'w') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(header)

    def __call__(self, log):
        # Format log file:
        log[1] = '{:.5e}'.format(log[1])    # Learning rate
        # Train loss, PSNR, SSIM:
        log[2], log[3], log[4] = '{:.5e}'.format(log[2]), '{:.5f}'.format(log[3]), '{:.5f}'.format(log[4])
        # Val loss, PSNR, SSIM:
        log[5], log[6], log[7] = '{:.5e}'.format(log[5]), '{:.5f}'.format(log[6]), '{:.5f}'.format(log[7])

        with open(self.file_path, 'a') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(log)


def fit_model(model, data_loaders, channels, criterion, optimizer, scheduler, device, num_epochs, val_freq,
              checkpoint_dir, model_name, verbose=True):
    """
    Training of the denoiser model.
    :param model: torch Module
        Model to fit.
    :param data_loaders: dict
        Dictionary with torch DataLoaders with training and validation datasets.
    :param channels: int
        Number of image channels.
    :param criterion: torch Module
        Loss function.
    :param optimizer: torch Optimizer
        Optimizer algorithm.
    :param scheduler: torch lr_scheduler
        Learning rate scheduler.
    :param device:  torch device
        Device used during train (CPU/GPU).
    :param num_epochs: int
        Number of training epochs.
    :param val_freq: int
        Interval of the validation process.
    :param checkpoint_dir: str
        Path to create and store training log file and model checkpoints.
    :param model_name: str
        Base name of the model saved in checkpoint_dir.
    :param verbose: bool
        If true, displays progress bar. If false, only print results at the end of the epoch.
    :return:
    """
    psnr = PSNR(data_range=1., reduction='none', eps=0)
    ssim = SSIM(channels, data_range=1., size_average=False)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logfile_path = os.path.join(checkpoint_dir,  ''.join([model_name, '_logfile.csv']))
    model_path = os.path.join(checkpoint_dir, ''.join([model_name, '-{:04d}-{:.4e}-{:.4f}-{:.4f}{}.pth']))
    description_format = ' - Loss:{:.5e} - PSNR:{:.5f} - SSIM:{:.5f}'
    header = ['epoch', 'lr', 'train loss', 'train psnr', 'train ssim', 'val loss', 'val psnr', 'val ssim']
    file_logger = FileLogger(logfile_path, header)
    best_model_path, best_loss, best_psnr, best_ssim = '', np.inf, -np.inf, -np.inf
    since = time.time()

    for epoch in range(1, num_epochs + 1):
        learning_rate = optimizer.param_groups[0]['lr']
        epoch_logger = EpochLogger()
        num_samples = {'train': 0, 'val': 0}
        epoch_log = dict()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            elif phase == 'val' and epoch % val_freq == 0:
                model.eval()
            else:
                break

            if verbose:
                if phase == 'train':
                    print('\nEpoch: {}/{} - lr: {:.4e}'.format(epoch, num_epochs, learning_rate))
                description = phase + description_format
                iterator = tqdm(enumerate(data_loaders[phase], 1), total=len(data_loaders[phase]), ncols=100)
                iterator.set_description(description.format(0, 0, 0))
            else:
                iterator = enumerate(data_loaders[phase], 1)

            for step, sample in iterator:
                lr, hr = sample['LR'], sample['HR']
                lr, hr = lr.to(device), hr.to(device)
                target_size = sample['target size']

                model.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_hr = model(lr, target_size)
                    loss = criterion(outputs_hr, hr)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                num_samples[phase] += lr.size()[0]
                metrics = {
                    'loss': loss.item() * lr.size()[0],
                    'psnr': psnr(outputs_hr, hr).sum().item(),
                    'ssim': ssim(outputs_hr, hr).sum().item(),
                }
                epoch_logger.update_log(metrics, phase)
                log = epoch_logger.get_log(num_samples, phase)

                if verbose:
                    iterator.set_description(description.format(
                        log[phase + ' loss'],
                        log[phase + ' psnr'],
                        log[phase + ' ssim']
                    ))

            # Save the model improves if any validation metric is improved.
            if phase == 'val':
                save_model = False
                if log['val psnr'] > best_psnr:
                    best_psnr = log['val psnr']
                    save_model = True
                if log['val loss'] < best_loss:
                    best_loss = log['val loss']
                    save_model = True
                if log['val ssim'] > best_ssim:
                    best_ssim = log['val ssim']
                    save_model = True

                if save_model:
                    best_model_path = model_path.format(epoch, log['val loss'], log['val psnr'], log['val ssim'], '-improve')
                    torch.save(model.state_dict(), best_model_path)

            elif scheduler is not None:
                scheduler.step()

            epoch_log = {**epoch_log, **log}

        epoch_data = [
            epoch, learning_rate, epoch_log['train loss'], epoch_log['train psnr'], epoch_log['train ssim'],
            epoch_log.get('val loss', 0), epoch_log.get('val psnr', 0), epoch_log.get('val ssim', 0)
        ]
        file_logger(epoch_data)

    # Save final model
    model_path = model_path.format(num_epochs + 1, log['val loss'], log['val psnr'], log['val ssim'], '-final')
    torch.save(model.state_dict(), model_path)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best PSNR: {:4f}'.format(best_psnr))
