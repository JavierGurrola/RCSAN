import torch
from pytorch_msssim import SSIM as _SSIM


class PSNR(object):
    r"""
    Peak signal-to-noise ratio
    Args:
        data_range (float): Range of the input images.
        reduction (str): Type of reduction applied to the batch.
        eps (float): Epsilon value for numerical stability.
    """
    def __init__(self, data_range, reduction='none', eps=1e-8):
        self.data_range = data_range
        self.reduction = reduction
        self.eps = eps

    def __call__(self, outputs, targets):
        with torch.set_grad_enabled(False):
            mse = torch.mean((outputs - targets) ** 2., dim=(1, 2, 3))
            psnr = 10. * torch.log10((self.data_range ** 2.) / (mse + self.eps))

            if self.reduction == 'mean':
                return psnr.mean()
            if self.reduction == 'sum':
                return psnr.sum()

            return psnr


class SSIM(object):
    r"""
    Structural similarity index measure.
    Args:
        channels (int):  Number of input channels.
        data_range (float): Range of the input images.
        size_average (bool): If true, the ssim value of all images will be averaged in a scalar
    """
    def __init__(self, channels, data_range, size_average=True):
        self.data_range = data_range
        self.ssim_module = _SSIM(data_range=data_range, size_average=size_average, channel=channels)

    def __call__(self, outputs, targets):
        with torch.set_grad_enabled(False):
            return self.ssim_module(outputs, targets)


class CharbonnierLoss(torch.nn.Module):
    r"""
    Charbonnier loss function.
    Args:
        eps (float): epsilon parameter of the Carbonnier loss function.
    """
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, outputs, targets):
        diff = outputs - targets
        error = torch.sqrt(diff ** 2. + self.eps)
        loss = torch.mean(error)

        return loss
