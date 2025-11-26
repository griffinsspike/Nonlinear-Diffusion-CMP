import numpy as np
from scipy.ndimage import gaussian_filter

def compute_gradients(image):
    """
    Computes central difference gradients for x and y directions.
    Returns: (grad_x, grad_y)
    """
    # Pad image to handle boundaries
    padded = np.pad(image, 1, mode='edge')
    
    # Central difference: (u[x+1] - u[x-1]) / 2
    grad_x = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.0
    grad_y = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.0
    
    return grad_x, grad_y

def gaussian_smooth(image, sigma):
    """Applies Gaussian smoothing if sigma > 0."""
    if sigma > 0:
        return gaussian_filter(image, sigma=sigma)
    return image.copy()