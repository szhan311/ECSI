import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from PIL import Image

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

# Use make_grid to create a grid of images
def show_img_grid(sample, nrow=8, ncol=8):
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.contiguous()
    
    # Ensure the number of images matches the grid dimensions
    total_images = nrow * ncol
    sample = sample[:total_images]  # Adjust sample size to fit the grid

    # Create the grid with specified number of rows
    grid = vutils.make_grid(sample, nrow=nrow, padding=2, normalize=False)

    # Convert the grid to a numpy array
    grid_np = grid.cpu().numpy()

    # Transpose the numpy array from (C, H, W) to (H, W, C) for displaying
    grid_np = np.transpose(grid_np, (1, 2, 0))

    # Display the grid of images
    plt.figure(figsize=(ncol, nrow), dpi=600)  # Adjust size based on grid dimensions
    plt.imshow(grid_np)
    plt.axis('off')  # Turn off the axis numbers and ticks
    plt.show()

# Use make_grid to create a grid of images
def show_img_grid_s(sample):
      sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
      sample = sample.contiguous()
      n = sample.shape[1]
      sample_flat = sample.view(-1, sample.shape[-3], sample.shape[-2], sample.shape[-1])
      grid = vutils.make_grid(sample_flat, nrow=n, padding=2, normalize=False)

      # Convert the grid to a numpy array
      grid_np = grid.cpu().numpy()

      # Transpose the numpy array from (C, H, W) to (H, W, C) for displaying
      grid_np = np.transpose(grid_np, (1, 2, 0))

      # Display the grid of images
      plt.figure(figsize=(5, 5), dpi=300)  # Adjust size to your liking
      plt.imshow(grid_np)
      plt.axis('off')  # Turn off the axis numbers and ticks
      plt.show()

def save_img_grid(sample, save_path, nrow=8, ncol=8):
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.contiguous()
    
    # Ensure the number of images matches the grid dimensions
    total_images = nrow * ncol
    sample = sample[:total_images]  # Adjust sample size to fit the grid

    # Create the grid with specified number of rows
    grid = vutils.make_grid(sample, nrow=nrow, padding=2, normalize=False)

    # Convert the grid to a numpy array
    grid_np = grid.cpu().numpy()

    # Transpose the numpy array from (C, H, W) to (H, W, C) for saving
    grid_np = np.transpose(grid_np, (1, 2, 0))

    # Save the grid of images
    Image.fromarray(grid_np).save(save_path)
    
def save_img_grid_s(sample, save_path, row_labels=None):
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.contiguous()
    n = sample.shape[1]
    sample_flat = sample.view(-1, sample.shape[-3], sample.shape[-2], sample.shape[-1])
    grid = vutils.make_grid(sample_flat, nrow=n, padding=2, normalize=False)

    # Convert the grid to a numpy array
    grid_np = grid.cpu().numpy()

    # Transpose the numpy array from (C, H, W) to (H, W, C) for displaying
    grid_np = np.transpose(grid_np, (1, 2, 0))

    # Generate column labels
    col_labels = ["x0"] + [""] * (n - 3) + ["x1"]+ ["ref img"]

    # Plot the grid of images with labels
    fig, ax = plt.subplots(dpi=600)
    ax.imshow(grid_np)
    ax.axis('off')

    # Add row and column labels if provided
    if row_labels:
        for idx, row_label in enumerate(row_labels):
            ax.text(grid_np.shape[1] + 10, idx * (grid_np.shape[0] // len(row_labels)) + 0.5 * (grid_np.shape[0] // len(row_labels)), 
                    row_label, va='center', ha='left', rotation=0, fontsize=12)

    for idx, col_label in enumerate(col_labels):
        ax.text(idx * (grid_np.shape[1] // len(col_labels)) + 0.5 * (grid_np.shape[1] // len(col_labels)), -10, 
                col_label, va='bottom', ha='center', rotation=0, fontsize=12)

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()