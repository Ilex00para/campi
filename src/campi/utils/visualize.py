import numpy as np
import matplotlib.pyplot as plt

def plot_image_channels(image: np.ndarray, titles: list = None, max_channels: int = 6):
    """
    Plot up to `max_channels` individual channels from a multi-channel image.

    Parameters:
    - image (np.ndarray): Input image of shape (2592, 4608, channels)
    - titles (list of str): Optional list of titles for each subplot
    - max_channels (int): Maximum number of channels allowed (default is 6)

    Raises:
    - ValueError: if input shape is invalid or number of channels exceeds max_channels
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if image.ndim != 3:
        raise ValueError("Input array must be 3D (H, W, Channels).")
    
    height, width, channels = image.shape
    # if height != 2592 or width != 4608:
    #     raise ValueError(f"Expected image shape (2592, 4608, channels), got {image.shape}")
    if channels > max_channels:
        raise ValueError(f"Too many channels ({channels}). Maximum allowed is {max_channels}.")

    if titles and len(titles) != channels:
        raise ValueError("Length of titles must match number of channels.")

    fig, axes = plt.subplots(1, channels, figsize=(5 * channels, 5))
    
    # Ensure axes is iterable even if only one channel
    if channels == 1:
        axes = [axes]
    
    for i in range(channels):
        ax = axes[i]
        ax.imshow(image[:, :, i])
        if titles:
            ax.set_title(titles[i], fontsize=12)
        else:
            ax.set_title(f"Channel {i}", fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('data/pictures/image_channels.png', dpi=300)