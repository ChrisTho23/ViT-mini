import pytest
import torch
import logging
import os

from src.data import load_cifar10, Patchify
from src.utils import display_image, Depatchify

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

@pytest.fixture(scope="class")
def target_directory(tmpdir_factory):
    """Fixture to create a temporary directory for dataset storage."""
    temp_dir = tmpdir_factory.mktemp("data")
    return temp_dir

@pytest.fixture(scope="class")
def cifar10_data(target_directory):
    """Fixture to load the CIFAR-10 dataset once for all tests."""
    train_loader, test_loader = load_cifar10(
        target_directory=str(target_directory), 
        download=True, 
        batch_size=64
    )
    return train_loader, test_loader

@pytest.mark.usefixtures("target_directory")
class TestLoadCIFAR10:
    """Test class for load_cifar10 functionality."""

    def test_load_cifar10(self, cifar10_data):
        """Test the load_cifar10 function with download=True."""
        train_loader, test_loader = cifar10_data

        # Check if DataLoader objects are returned
        assert isinstance(train_loader, torch.utils.data.DataLoader), "Expected train_loader to be a DataLoader instance"
        assert isinstance(test_loader, torch.utils.data.DataLoader), "Expected test_loader to be a DataLoader instance"

        # Check if DataLoader has data
        train_data_iter = iter(train_loader)
        test_data_iter = iter(test_loader)
        train_data, train_labels = next(train_data_iter)
        test_data, test_labels = next(test_data_iter)

        # Validate the number of images in the train and test sets
        total_train_images = len(train_loader.dataset)
        total_test_images = len(test_loader.dataset)
        assert total_train_images == 50000, f"Expected 50000 training images, but got {total_train_images}"
        assert total_test_images == 10000, f"Expected 10000 test images, but got {total_test_images}"

        # Validate the shape of the images and labels
        assert train_data.shape[1:] == (3, 32, 32), f"Expected train images of shape (3, 32, 32), but got {train_data.shape[1:]}"
        assert test_data.shape[1:] == (3, 32, 32), f"Expected test images of shape (3, 32, 32), but got {test_data.shape[1:]}"
        assert len(train_labels) == 64, f"Expected 64 train labels, but got {len(train_labels)}"
        assert len(test_labels) == 64, f"Expected 64 test labels, but got {len(test_labels)}"

        logging.info(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
        logging.info(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

    def test_load_cifar10_no_download(self, target_directory):
        """Test the load_cifar10 function with download=False."""
        # Check if dataset files exist
        train_dir = os.path.join(target_directory, 'cifar-10-batches-py')
        assert os.path.exists(train_dir), "Expected CIFAR-10 training directory to exist"

        # Load the dataset without downloading
        train_loader, test_loader = load_cifar10(
            target_directory=str(target_directory), 
            download=False, 
            batch_size=64
        )

        # Check if DataLoader objects are returned
        assert isinstance(train_loader, torch.utils.data.DataLoader), "Expected train_loader to be a DataLoader instance"
        assert isinstance(test_loader, torch.utils.data.DataLoader), "Expected test_loader to be a DataLoader instance"

        # Check if DataLoader has data
        train_data_iter = iter(train_loader)
        test_data_iter = iter(test_loader)
        train_data, train_labels = next(train_data_iter)
        test_data, test_labels = next(test_data_iter)

        # Validate the number of images in the train and test sets
        total_train_images = len(train_loader.dataset)
        total_test_images = len(test_loader.dataset)
        assert total_train_images == 50000, f"Expected 50000 training images, but got {total_train_images}"
        assert total_test_images == 10000, f"Expected 10000 test images, but got {total_test_images}"

        # Validate the shape of the images and labels
        assert train_data.shape[1:] == (3, 32, 32), f"Expected train images of shape (3, 32, 32), but got {train_data.shape[1:]}"
        assert test_data.shape[1:] == (3, 32, 32), f"Expected test images of shape (3, 32, 32), but got {test_data.shape[1:]}"
        assert len(train_labels) == 64, f"Expected 64 train labels, but got {len(train_labels)}"
        assert len(test_labels) == 64, f"Expected 64 test labels, but got {len(test_labels)}"

        logging.info(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
        logging.info(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

    def test_reassemble_patches(self, cifar10_data):
        """Test reassembling patches back to images."""
        train_loader, _ = cifar10_data

        # Get a single batch (one image)
        train_data_iter = iter(train_loader)
        image, _ = next(train_data_iter)

        # Define patch size
        patch_size = 4

        # Patchify the image
        patchify = Patchify(patch_size=patch_size)
        patches = patchify(image)

        # Reassemble the image
        depatchify = Depatchify(patch_size=patch_size, image_size=(32, 32))
        reassembled_image = depatchify(patches)

        # Display the original and reassembled images
        #display_image(image[0], title="Original Image")
        #display_image(reassembled_image[0], title="Reassembled Image")

        # Check if the reassembled image is close to the original
        assert torch.allclose(image, reassembled_image, atol=1e-6), "Reassembled image is not close to the original image"

        logging.info("Original and reassembled images are close.")