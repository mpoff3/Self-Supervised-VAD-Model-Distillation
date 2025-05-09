import torch
import torch.nn.functional as F

def max_pool_interpolate(score, sample_step, **kwargs):
    """Max pooling interpolation."""
    kernel_size = sample_step
    pad_front = (kernel_size - 1) // 2
    pad_back = (kernel_size - 1) - pad_front
    score = F.pad(score, (0, 0, 0, 0, pad_front, pad_back), mode='constant', value=1)
    return -F.max_pool3d(-score, (kernel_size, 1, 1), stride=(1, 1, 1), padding=0)

def linear_interpolate(score, sample_step, **kwargs):
    """Linear interpolation using 1D convolution."""
    kernel_size = sample_step
    pad_front = (kernel_size - 1) // 2
    pad_back = (kernel_size - 1) - pad_front
    
    # Create linear weights for interpolation
    weights = torch.linspace(1, 0, kernel_size, device=score.device)
    weights = weights / weights.sum()
    weights = weights.view(1, 1, kernel_size, 1, 1)
    
    # Pad and apply convolution
    score = F.pad(score, (0, 0, 0, 0, pad_front, pad_back), mode='constant', value=1)
    return F.conv3d(score, weights, stride=(1, 1, 1), padding=0)

def gaussian_interpolate(score, sample_step, sigma=0.5, **kwargs):
    """Gaussian weighted interpolation using 1D convolution."""
    kernel_size = sample_step
    pad_front = (kernel_size - 1) // 2
    pad_back = (kernel_size - 1) - pad_front
    
    # Create Gaussian weights
    x = torch.linspace(-2, 2, kernel_size, device=score.device)
    weights = torch.exp(-0.5 * (x / sigma) ** 2)
    weights = weights / weights.sum()
    weights = weights.view(1, 1, kernel_size, 1, 1)
    
    # Pad and apply convolution
    score = F.pad(score, (0, 0, 0, 0, pad_front, pad_back), mode='constant', value=1)
    return F.conv3d(score, weights, stride=(1, 1, 1), padding=0)

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

def nearest_interpolate(score, sample_step, **kwargs):
    """Nearest neighbor using max pooling with kernel size based on sample_step."""
    kernel_size = sample_step
    pad_front = (kernel_size - 1) // 2
    pad_back = (kernel_size - 1) - pad_front
    
    # Create mask for non-1 values
    mask = (score != 1).float()
    
    # Use max pooling to propagate nearest non-1 values
    score = F.pad(score, (0, 0, 0, 0, pad_front, pad_back), mode='constant', value=1)
    mask = F.pad(mask, (0, 0, 0, 0, pad_front, pad_back), mode='constant', value=0)
    
    # Use max pooling to find nearest non-1 values
    return F.max_pool3d(score * mask, (kernel_size, 1, 1), stride=(1, 1, 1), padding=0)

def no_interpolate(score, sample_step, **kwargs):
    """No interpolation, leave missing frames as 1."""
    return score

# Dictionary mapping method names to their functions
INTERPOLATION_METHODS = {
    'max_pool': max_pool_interpolate,
    'linear': linear_interpolate,
    'gaussian': gaussian_interpolate,
    'moving_avg': moving_avg_interpolate,
    'nearest': nearest_interpolate,
    'none': no_interpolate
} 