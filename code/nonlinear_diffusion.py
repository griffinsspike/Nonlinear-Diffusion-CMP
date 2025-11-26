import numpy as np
import cv2
import os

# PDF yapısına uygun yerel importlar
from diffusivity_functions import get_diffusivity
from utils import compute_gradients, gaussian_smooth
from analysis import plot_comparison, plot_statistics

class NonlinearDiffusion:
    """
    Main class implementing the Perona-Malik diffusion model.
    """
    def __init__(self, lambda_param=10.0, sigma=1.0, dt=0.25, num_iterations=50, method='pm1'):
        self.lambda_param = lambda_param
        self.sigma = sigma
        self.dt = dt
        self.num_iterations = num_iterations
        self.diff_func = get_diffusivity(method)
        self.method_name = method

    def run_grayscale(self, image):
        """Applies diffusion to a grayscale image."""
        img_float = image.astype(np.float64)
        history = {'mean': [], 'variance': [], 'grad_mag': []}
        
        print(f"Running Grayscale Diffusion ({self.method_name})...")
        
        for i in range(self.num_iterations):
            # 1. Smooth
            smoothed = gaussian_smooth(img_float, self.sigma)
            
            # 2. Gradients
            gx, gy = compute_gradients(smoothed)
            grad_mag = np.sqrt(gx**2 + gy**2)
            
            # 3. Diffusivity
            g = self.diff_func(grad_mag, self.lambda_param)
            
            # 4. Divergence
            gx_orig, gy_orig = compute_gradients(img_float)
            flux_x = g * gx_orig
            flux_y = g * gy_orig
            
            div_x, _ = compute_gradients(flux_x)
            _, div_y = compute_gradients(flux_y)
            
            # 5. Update
            img_float += self.dt * (div_x + div_y)
            
            # Stats
            history['mean'].append(np.mean(img_float))
            history['variance'].append(np.var(img_float))
            history['grad_mag'].append(np.sum(grad_mag))
            
        return np.clip(img_float, 0, 255).astype(np.uint8), history

    def run_color(self, image):
        """
        Applies Coupled Vector-Valued Diffusion for RGB images.
        Uses Joint Gradient Magnitude: g(sum(|grad(uk)|))
        """
        img_float = image.astype(np.float64)
        channels = [img_float[:, :, i] for i in range(3)]
        history = {'mean': [], 'variance': [], 'grad_mag': []}
        
        print(f"Running Color Diffusion ({self.method_name})...")

        for i in range(self.num_iterations):
            # 1. Joint Gradient Magnitude Calculation
            total_grad_sq = np.zeros_like(channels[0])
            
            for ch in channels:
                smoothed = gaussian_smooth(ch, self.sigma)
                gx, gy = compute_gradients(smoothed)
                total_grad_sq += (gx**2 + gy**2)
            
            joint_grad_mag = np.sqrt(total_grad_sq)
            
            # 2. Shared Diffusivity
            g = self.diff_func(joint_grad_mag, self.lambda_param)
            
            # 3. Update Channels
            new_channels = []
            for ch in channels:
                gx, gy = compute_gradients(ch)
                div_x, _ = compute_gradients(g * gx)
                _, div_y = compute_gradients(g * gy)
                
                updated_ch = ch + self.dt * (div_x + div_y)
                new_channels.append(updated_ch)
            
            channels = new_channels
            
            # Stats (using average)
            history['mean'].append(np.mean(np.stack(channels)))
            history['variance'].append(np.var(np.stack(channels)))
            history['grad_mag'].append(np.sum(joint_grad_mag))

        result = np.stack(channels, axis=2)
        return np.clip(result, 0, 255).astype(np.uint8), history

# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================
if __name__ == "__main__":
    # Setup directories
    os.makedirs("../html/results", exist_ok=True)
    os.makedirs("../html/plots", exist_ok=True)
    
    print("--- CMP 717 Assignment 4: Nonlinear Diffusion ---")
    
    # Load Image (Adjust path as needed)
    # Using relative path assuming code is run from 'code/' directory
    img_path = '../ornekColor.png' 
    
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}. Creating synthetic image.")
        image = np.random.randint(0, 255, (256, 256)).astype(np.uint8)
        image[50:150, 50:150] = 200
        is_color = False
    else:
        image = cv2.imread(img_path)
        is_color = len(image.shape) == 3

    # Initialize Solver
    solver = NonlinearDiffusion(
        lambda_param=150.0, 
        sigma=1.0, 
        dt=0.3, 
        num_iterations=60, 
        method='pm1'
    )

    # Run Diffusion
    if is_color:
        result, stats = solver.run_color(image)
    else:
        # Convert to grayscale if loaded as color but meant to be gray
        if len(image.shape) == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result, stats = solver.run_grayscale(image)

    # Visualizations
    plot_comparison(image, result, title="PM1 Result")
    plot_statistics(stats)
    
    # Save Results
    stats_path = "../html/plots/statistics.png"
    cv2.imwrite("../html/results/processed_image.png", result)
    print("Process complete. Results saved to ../html/results/")
    print(f"Statistics plot saved to {stats_path}")  