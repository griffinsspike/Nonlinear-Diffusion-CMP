import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_comparison(original, result, title="Result", save_path=None):
    """Plots Original vs Processed Image."""
    plt.figure(figsize=(10, 5))
    
    # Handle color conversion for matplotlib (BGR to RGB)
    if len(original.shape) == 3:
        orig_view = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        res_view = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        cmap = None
    else:
        orig_view, res_view = original, result
        cmap = 'gray'

    plt.subplot(1, 2, 1)
    plt.imshow(orig_view, cmap=cmap)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(res_view, cmap=cmap)
    plt.title(f"Filtered ({title})")
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_statistics(history, save_path=None):
    """Plots Mean Intensity, Variance and Gradient Magnitude over iterations."""
    iterations = range(len(history['mean']))
    
    plt.figure(figsize=(15, 4))
    
    # Mean
    plt.subplot(1, 3, 1)
    plt.plot(iterations, history['mean'], 'b-')
    plt.title("Mean Intensity")
    plt.xlabel("Iteration")
    plt.grid(True, alpha=0.3)
    
    # Variance
    plt.subplot(1, 3, 2)
    plt.plot(iterations, history['variance'], 'r-')
    plt.title("Intensity Variance")
    plt.xlabel("Iteration")
    plt.grid(True, alpha=0.3)
    
    # Gradient Magnitude
    plt.subplot(1, 3, 3)
    plt.plot(iterations, history['grad_mag'], 'g-')
    plt.title("Total Gradient Magnitude")
    plt.xlabel("Iteration")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()