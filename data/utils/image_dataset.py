from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
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
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        
        if self.transform:
            image = self.transform(image)
            
        return image