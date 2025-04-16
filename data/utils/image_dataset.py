from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    A dataset class for loading images from a directory.
    
    This dataset loads all image files from the specified directory and optionally
    applies transformations to them. It's designed to work with PyTorch's DataLoader
    for efficient batch processing of images.
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset with images from the specified directory.
        
        Args:
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []

        # Collect all image paths
        for path in self.root_dir.iterdir():
            if path.is_file():
                self.image_paths.append(path)

    def __len__(self):
        """
        Return the number of images in the dataset.
        
        Returns:
            int: The total number of images
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get an image by index.
        
        Args:
            idx (int): Index of the image to retrieve
            
        Returns:
            torch.Tensor or PIL.Image: The image at the specified index,
                transformed if a transform is specified
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB

        if self.transform:
            image = self.transform(image)

        return image
