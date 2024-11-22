import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import functional as F
import random

def get_sample_image(index=None):
    """Get a sample image from MNIST dataset"""
    dataset = datasets.MNIST('data', train=True, download=True, 
                           transform=transforms.ToTensor())
    
    if index is None:
        index = random.randint(0, len(dataset)-1)
    
    image, label = dataset[index]
    return image, label, index

def show_augmentations(image, label):
    """Apply and display various augmentations on a single image"""
    # Define augmentations
    augmentations = {
        'Original': lambda x: x,
        'Rotation (30°)': lambda x: F.rotate(x, 30),
        'Rotation (45°)': lambda x: F.rotate(x, 45),
        'Horizontal Flip': F.hflip,
        'Vertical Flip': F.vflip,
        'Affine': lambda x: F.affine(x, angle=15, translate=(0.1, 0.1), 
                                   scale=0.9, shear=10),
        'Brightness': lambda x: F.adjust_brightness(x, 1.5),
        'Contrast': lambda x: F.adjust_contrast(x, 1.5),
        'Gaussian Blur': lambda x: F.gaussian_blur(x, kernel_size=[3, 3], 
                                                 sigma=[0.5, 0.5])
    }

    # Create subplot grid
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(f'MNIST Digit: {label}', fontsize=16)

    for idx, (name, aug_func) in enumerate(augmentations.items(), 1):
        ax = fig.add_subplot(3, 3, idx)
        img_aug = aug_func(image)
        
        # Convert tensor to numpy array for plotting
        if isinstance(img_aug, torch.Tensor):
            img_aug = img_aug.squeeze().numpy()
        
        ax.imshow(img_aug, cmap='gray')
        ax.set_title(name)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Get a random sample
    image, label, index = get_sample_image()
    print(f"Showing augmentations for MNIST digit {label} (index: {index})")
    
    # Show augmentations
    show_augmentations(image, label)

    # Allow user to try different samples
    while True:
        user_input = input("\nEnter an index (0-59999) to see another sample, or 'q' to quit: ")
        if user_input.lower() == 'q':
            break
        try:
            index = int(user_input)
            if 0 <= index <= 59999:
                image, label, _ = get_sample_image(index)
                print(f"Showing augmentations for MNIST digit {label}")
                show_augmentations(image, label)
            else:
                print("Please enter a valid index between 0 and 59999")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

if __name__ == "__main__":
    main() 