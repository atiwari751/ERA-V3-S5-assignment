import torch
import pytest
from torchvision import datasets, transforms
from model.network import SimpleCNN
import torch.nn.utils.prune as prune
import os
import glob
from train import train
from utils.visualize_augmentations import get_sample_image, show_augmentations
from torchvision.transforms import functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = SimpleCNN()
    num_params = count_parameters(model)
    assert num_params < 25000, f"Model has {num_params} parameters, should be less than 25000"
    print(f"\nModel has {num_params} parameters")

def test_input_output_shape():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # Check if model exists, if not train it
    model_files = glob.glob('models/*.pth')
    if not model_files:
        train()
        model_files = glob.glob('models/*.pth')
    
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))
    
    # Test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 95, f"Accuracy is {accuracy}%, should be > 95%"
    print(f"\nModel achieved {accuracy:.2f}% accuracy on test set")

def test_augmentations():
    """Test the image augmentation functionality"""
    print("\nTesting Image Augmentations:")
    print("----------------------------")
    
    # Get a sample image
    image, label, _ = get_sample_image(0)  # Use first image for reproducibility
    print(f"Testing with MNIST digit: {label}")
    
    # Define expected augmentations
    augmentations = {
        'Original': lambda x: x,
        'Rotation': lambda x: F.rotate(x, 30),
        'Horizontal Flip': F.hflip,
        'Vertical Flip': F.vflip,
        'Affine': lambda x: F.affine(x, angle=15, translate=(0.1, 0.1), scale=0.9, shear=10),
        'Brightness': lambda x: F.adjust_brightness(x, 1.5),
        'Contrast': lambda x: F.adjust_contrast(x, 1.5),
        'Gaussian Blur': lambda x: F.gaussian_blur(x, kernel_size=[3, 3], sigma=[0.5, 0.5]),
        'Random Perspective': lambda x: F.perspective(x, 
            startpoints=[[0, 0], [1, 0], [1, 1], [0, 1]],
            endpoints=[[0.1, 0.1], [0.9, 0], [0.9, 0.9], [0.1, 0.9]])
    }
    
    # Test each augmentation
    successful_augmentations = 0
    failed_augmentations = []
    
    print("\nTesting individual augmentations:")
    for name, aug_func in augmentations.items():
        try:
            # Apply augmentation
            aug_image = aug_func(image)
            
            # Verify augmented image
            assert isinstance(aug_image, torch.Tensor), f"Augmentation {name} failed to return a tensor"
            assert aug_image.shape == image.shape, f"Augmentation {name} changed image shape"
            
            if name != 'Original':
                # Check if augmentation made changes (compare with original)
                assert not torch.allclose(aug_image, image), f"Augmentation {name} didn't modify the image"
                diff_percentage = torch.mean(torch.abs(aug_image - image)) * 100
                print(f"✓ {name:<20} - Success (Pixel difference: {diff_percentage:.2f}%)")
            else:
                print(f"✓ {name:<20} - Original image preserved")
            
            successful_augmentations += 1
            
        except Exception as e:
            failed_augmentations.append(f"{name}: {str(e)}")
            print(f"✗ {name:<20} - Failed: {str(e)}")
    
    # Report summary
    print("\nAugmentation Summary:")
    print(f"Total augmentations tested: {len(augmentations)}")
    print(f"Successful augmentations:   {successful_augmentations}")
    print(f"Failed augmentations:       {len(failed_augmentations)}")
    
    if failed_augmentations:
        print("\nFailed augmentations details:")
        for failure in failed_augmentations:
            print(f"- {failure}")
    
    assert successful_augmentations == len(augmentations), \
        f"Only {successful_augmentations} augmentations succeeded out of {len(augmentations)}"
    assert not failed_augmentations, f"Some augmentations failed: {failed_augmentations}"

def test_model_layer_shapes():
    """Test if model layers have correct shapes and configurations"""
    model = SimpleCNN()
    
    # Test convolution layers
    assert model.conv1.in_channels == 1, "First conv layer should have 1 input channel"
    assert model.conv1.out_channels == 4, "First conv layer should have 4 output channels"
    assert model.conv2.in_channels == 4, "Second conv layer should have 4 input channels"
    assert model.conv2.out_channels == 8, "Second conv layer should have 8 output channels"
    
    # Test batch normalization layers
    assert model.bn1.num_features == 4, "First BatchNorm should have 4 features"
    assert model.bn2.num_features == 8, "Second BatchNorm should have 8 features"
    
    # Test dropout rate
    assert model.dropout.p == 0.25, "Dropout rate should be 0.25"
    
    print("\nModel architecture verification:")
    print(f"Conv1: {model.conv1.in_channels} → {model.conv1.out_channels}")
    print(f"Conv2: {model.conv2.in_channels} → {model.conv2.out_channels}")
    print(f"FC1: {8 * 7 * 7} → 32")
    print(f"FC2: 32 → 10")

def test_model_memory_usage():
    """Test if model fits within memory constraints"""
    import sys
    
    model = SimpleCNN()
    
    # Get model size in MB
    model_size = sum(param.nelement() * param.element_size() 
                    for param in model.parameters()) / (1024 * 1024)
    
    # Get size of one batch of data
    batch_size = 64
    sample_input = torch.randn(batch_size, 1, 28, 28)
    input_size = (sample_input.nelement() * sample_input.element_size()) / (1024 * 1024)
    
    total_size = model_size + input_size
    
    assert total_size < 100, f"Model + batch uses {total_size:.2f}MB, should be < 100MB"
    print(f"\nMemory Usage Analysis:")
    print(f"Model size: {model_size:.2f}MB")
    print(f"Batch size: {input_size:.2f}MB")
    print(f"Total size: {total_size:.2f}MB")