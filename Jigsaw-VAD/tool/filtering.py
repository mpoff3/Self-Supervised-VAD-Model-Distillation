import torch
import torch.nn.functional as F

def mean_filter(score, dim=5, **kwargs):
    """3D mean filter."""
    p3d = (dim // 2, dim // 2, dim // 2, dim // 2, dim // 2, dim // 2)
    score = F.pad(score, p3d, mode='constant', value=1)
    return F.avg_pool3d(score, kernel_size=dim, stride=1, padding=0)

def gaussian_filter(score, dim=5, sigma=1.0, **kwargs):
    """3D Gaussian filter."""
    # Create 3D Gaussian kernel
    x = torch.linspace(-2, 2, dim, device=score.device)
    y = torch.linspace(-2, 2, dim, device=score.device)
    z = torch.linspace(-2, 2, dim, device=score.device)
    x, y, z = torch.meshgrid(x, y, z)
    kernel = torch.exp(-0.5 * ((x/sigma)**2 + (y/sigma)**2 + (z/sigma)**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, dim, dim, dim)
    
    # Pad and apply convolution
    p3d = (dim // 2, dim // 2, dim // 2, dim // 2, dim // 2, dim // 2)
    score = F.pad(score, p3d, mode='constant', value=1)
    return F.conv3d(score, kernel, stride=1, padding=0)

def median_filter(score, dim=5, **kwargs):
    """3D median filter using max pooling of negative values."""
    p3d = (dim // 2, dim // 2, dim // 2, dim // 2, dim // 2, dim // 2)
    score = F.pad(score, p3d, mode='constant', value=1)
    # Approximate median using max pooling of negative values
    return -F.max_pool3d(-score, kernel_size=dim, stride=1, padding=0)

def bilateral_filter(score, dim=5, sigma_space=1.0, sigma_intensity=0.1, **kwargs):
    """3D bilateral filter (approximated using separable convolutions)."""
    # Create spatial Gaussian kernel
    x = torch.linspace(-2, 2, dim, device=score.device)
    spatial_kernel = torch.exp(-0.5 * (x / sigma_space) ** 2)
    spatial_kernel = spatial_kernel / spatial_kernel.sum()
    spatial_kernel = spatial_kernel.view(1, 1, dim, 1, 1)
    
    # Pad
    p3d = (dim // 2, dim // 2, dim // 2, dim // 2, dim // 2, dim // 2)
    score = F.pad(score, p3d, mode='constant', value=1)
    
    # Apply spatial filtering
    filtered = F.conv3d(score, spatial_kernel, stride=1, padding=0)
    
    # Apply intensity-based weighting
    diff = (score - filtered).abs()
    intensity_weight = torch.exp(-0.5 * (diff / sigma_intensity) ** 2)
    
    return filtered * intensity_weight + score * (1 - intensity_weight)

def no_filter(score, dim=5, **kwargs):
    """No filtering, return score as is."""
    return score

# Dictionary mapping method names to their functions
FILTER_METHODS = {
    'mean': mean_filter,
    'gaussian': gaussian_filter,
    'median': median_filter,
    'bilateral': bilateral_filter,
    'none': no_filter
} 