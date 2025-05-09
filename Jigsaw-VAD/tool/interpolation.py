import torch
import torch.nn.functional as F


def duplicate_interpolate(score, sample_step, **kwargs):
    """Duplicate interpolation."""
    # kernel_size = sample_step + (sample_step+1) % 2
    # pad = kernel_size//2
    # score = F.pad(score, (0, 0, 0, 0, pad, pad), mode='constant', value=1)
    # return -F.max_pool3d(-score, (kernel_size, 1, 1), stride=(1, 1, 1), padding=0)
    kernel_size = sample_step
    pad_front = (kernel_size - 1) // 2
    pad_back = (kernel_size - 1) - pad_front
    score = F.pad(score, (0, 0, 0, 0, pad_front, pad_back), mode='constant', value=1)
    return -F.max_pool3d(-score, (kernel_size, 1, 1), stride=(1, 1, 1), padding=0)


def interpolate1plus2D(score, sample_step, dim_interp=5, **kwargs):
    """1+2D interpolation."""
    assert dim_interp % 2 == 1, "Kernel size must be odd for this operation"
    score = linear_interpolate(score, sample_step, **kwargs)

    kernel_size = dim_interp
    pad = kernel_size // 2
    score = F.pad(score, (pad, pad, pad, pad, 0, 0), mode='constant', value=1)
    return -F.avg_pool3d(-score, (1, kernel_size, kernel_size), stride=(1, 1, 1), padding=0)


def linear_interpolate(score, sample_step, **kwargs):
    """Linear interpolation using 1D convolution."""
    kernel_size = 2*sample_step - 1
    pad_front = (kernel_size - 1) // 2
    pad_back = (kernel_size - 1) - pad_front

    # Create linear weights for interpolation
    weights = torch.linspace(1, -1, kernel_size, device=score.device)
    weights = 1 - torch.abs(weights)
    weights = weights.view(1, 1, kernel_size, 1, 1)

    # Pad and apply convolution
    score = F.pad(score, (0, 0, 0, 0, pad_front, pad_back), mode='constant', value=1)
    return F.conv3d(score-1, weights, stride=(1, 1, 1), padding=0)+1


def moving_avg_interpolate(score, sample_step, **kwargs):
    """Simple moving average using 1D convolution."""
    kernel_size = sample_step
    pad_front = (kernel_size - 1) // 2
    pad_back = (kernel_size - 1) - pad_front

    # Create uniform weights
    weights = torch.ones(1, 1, kernel_size, 1, 1, device=score.device) / kernel_size

    # Pad and apply convolution
    score = F.pad(score, (0, 0, 0, 0, pad_front, pad_back), mode='constant', value=1)
    return F.conv3d(score, weights, stride=(1, 1, 1), padding=0)


def no_interpolate(score, sample_step, **kwargs):
    """No interpolation, leave missing frames as 1."""
    return score


# Dictionary mapping method names to their functions
INTERPOLATION_METHODS = {
    'nearest': duplicate_interpolate,
    'linear': linear_interpolate,
    'moving_avg': moving_avg_interpolate,
    'none': no_interpolate,
    '1+2D': interpolate1plus2D
}
