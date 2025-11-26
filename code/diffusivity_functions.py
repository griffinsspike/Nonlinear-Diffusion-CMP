import numpy as np

def pm_type1(gradient_magnitude, lambda_param):
    """
    Perona-Malik Type 1 Diffusivity.
    g(|x|) = exp(-|x|^2 / lambda^2)
    """
    return np.exp(-(gradient_magnitude ** 2) / (lambda_param ** 2))

def pm_type2(gradient_magnitude, lambda_param):
    """
    Perona-Malik Type 2 Diffusivity.
    g(|x|) = 1 / (1 + |x|^2 / lambda^2)
    """
    return 1.0 / (1.0 + (gradient_magnitude ** 2) / (lambda_param ** 2))

def charbonnier(gradient_magnitude, lambda_param):
    """
    Charbonnier Diffusivity.
    g(|x|) = 1 / sqrt(1 + |x|^2 / lambda^2)
    """
    return 1.0 / np.sqrt(1.0 + (gradient_magnitude ** 2) / (lambda_param ** 2))

def get_diffusivity(name):
    """Selects the correct function based on string input."""
    mapping = {
        'pm1': pm_type1,
        'pm2': pm_type2,
        'charbonnier': charbonnier
    }
    if name not in mapping:
        raise ValueError(f"Invalid type: {name}. Options: pm1, pm2, charbonnier")
    return mapping[name]