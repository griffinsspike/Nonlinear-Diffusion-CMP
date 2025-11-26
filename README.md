# Nonlinear Diffusion - Perona-Malik Model
## CMP 717 - Practical Assignment 

### 📋 Overview
This project implements **Nonlinear Diffusion Filtering** techniques to create a scale-space representation of images. The primary objective is to smooth images and reduce noise while preserving important features such as edges using Partial Differential Equations (PDEs).

The implementation focuses on the **Perona-Malik (PM)** model and extends to color images using coupled gradient diffusion.

---

### 📂 Directory Structure
The project adheres to the strict directory structure required for the assignment submission.

**Note:** Input images should be placed in the root directory (parent of `code/`).

```text
muhammed-emin-erdag-PA1/
├── README.md                   # Project documentation
├── ornekColor.png              # Input image (Required for demo)
├── code/
│   ├── nonlinear_diffusion.py  # Main entry point and class logic
│   ├── diffusivity_functions.py# Mathematical definitions (PM1, PM2, Charbonnier)
│   ├── analysis.py             # Plotting and statistics tools
│   └── utils.py                # Helper functions (Gradients, Gaussian smoothing)
└── html/                       # Output directory
    ├── results/                # Processed output images (comparison_result.png)
    └── plots/                  # Statistical graphs (statistics.png)

```

### 🚀 Features 
#### Problem 1.1: Perona-Malik Model
Three distinct diffusivity functions are implemented:
*	**PM Type 1:** Favors high-contrast edges ($g(x) = e^{-|x|^2/\lambda^2}$).
*	**PM Type 2:** Better for wide regions ($g(x) = 1 / (1 + |x|^2/\lambda^2)$).
*	**Charbonnier:** Robust, convex diffusion profile ($g(x) = 1 / \sqrt{1 + |x|^2/\lambda^2}$).

#### Problem 1.3: Color Image Support
The solution implements **Vector-Valued Diffusion** for RGB images. Instead of treating channels independently, it couples them using a joint gradient magnitude to prevent false colors:$$ \theta = \sqrt{\sum |\nabla u_k|^2} $$

### 💻 Installation & Usage
#### 1. Prerequisites
The project requires Python 3.x and the following libraries:
```
pip install numpy opencv-python matplotlib scipy 
```

#### 2. Running the Code
The main script is designed to be run from inside the code directory. It looks for the input image ornekColor.png in the parent directory.

```
cd code
python nonlinear_diffusion.py
```	

### ⚙️ Configuration
The default parameters in main are currently tuned for strong smoothing on high-resolution mosaic images:
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Method** | `pm1` | Perona-Malik Type 1 |
| **Lambda ($\lambda$)** | `150.0` | High contrast threshold (preserves only very strong edges) |
| **Sigma ($\sigma$)** | `1.0` | Gaussian smoothing scale before gradient calculation |
| **Time Step ($dt$)** | `0.3` | Integration step size |
| **Iterations** | `60` | Total number of diffusion steps |

To change these values, modify the NonlinearDiffusion initialization block in ```code/nonlinear_diffusion.py```	:

``` 
solver = NonlinearDiffusion(
    lambda_param=150.0, 
    sigma=1.0, 
    dt=0.3, 
    num_iterations=60, 
    method='pm1'
)
```

### 📊 Outputs
After execution, the results are automatically saved to the ```html/ ```directory:

*	```html/results/processed_image.png```: The final smoothed image.

*	```html/plots/statistics.png```: Graphs showing the evolution of Mean Intensity, Variance, and Gradient Magnitude.

*	```comparison_result.png```: Side-by-side comparison of Original vs. Filtered (not saved in the html directory).

### 📚 References
*  **Perona, P., & Malik, J. (1990).** "Scale space and edge detection using anisotropic diffusion". IEEE Transactions on Pattern Analysis and Machine Intelligence.

* **Charbonnier, P., et al. (1994).** "Two deterministic half-quadratic regularization algorithms for computed imaging". IEEE ICIP.