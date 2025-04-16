from dataclasses import dataclass
from typing import Optional

import pyrallis
import torchvision

from data.preprocessing.utils.collection_utils import (
    construct_head_from_boxes,
    form_masks_and_embeddings,
    get_filtered_indexes,
)
from data.utils.image_dataset import ImageDataset


@dataclass
class TrainConfig:
    """
    Configuration for dataset preparation process.
    
    Attributes:
        common_data_folder: Path to the folder with original LEGO minifigure images
        synthetic_data_folder: Path to the folder with synthetic/augmented LEGO images
        all_data_folder: Output directory for processed head images
        resize_param_common: Optional resize parameter for common dataset images
        resize_param_synthetic: Optional resize parameter for synthetic dataset images
        filter_percentile: Percentile threshold for filtering face embeddings
        padding_param_common: Padding to add around detected faces in common dataset
        padding_param_syntetic: Padding to add around detected faces in synthetic dataset
        device: Device to use for face detection models (cuda or cpu)
    """
    common_data_folder: str = "./lego_minifigs"
    synthetic_data_folder: str = "./lego_professions"
    all_data_folder: str = "./collected_heads"
    resize_param_common: Optional[int] = None
    resize_param_synthetic: Optional[int] = 512
    filter_percentile: float = 0.75
    padding_param_common: int = 20
    padding_param_syntetic: int = 50
    device: str = "cuda"


@pyrallis.wrap()
def main(config: TrainConfig):
    """
    Main function to prepare dataset for StyleGAN training.
    
    This function:
    1. Loads common and synthetic LEGO image datasets
    2. Detects faces and extracts embeddings from both datasets
    3. Filters the datasets based on face embedding similarity
    4. Extracts and saves the head regions with appropriate padding
    
    Args:
        config: Configuration object with dataset parameters
    """
    print("Beginning of collect dataset for tuning StyleGAN")

    # Load datasets with tensor transformation
    common_dataset = ImageDataset(
        config.common_data_folder,
        transform=lambda x: torchvision.transforms.ToTensor()(x),
    )
    synthetic_dataset = ImageDataset(
        config.synthetic_data_folder,
        transform=lambda x: torchvision.transforms.ToTensor()(x),
    )

    # Detect faces and extract embeddings from common dataset
    common_boxes, common_embeddings = form_masks_and_embeddings(
        config, common_dataset, config.resize_param_common, "common"
    )

    # Detect faces and extract embeddings from synthetic dataset
    synthetic_boxes, synthetic_embeddings = form_masks_and_embeddings(
        config, synthetic_dataset, config.resize_param_synthetic, "synthetic"
    )

    # Filter datasets based on embedding similarity
    common_indicies, synthetic_indicies = get_filtered_indexes(
        common_embeddings, synthetic_embeddings, percentile=config.filter_percentile
    )

    # Extract and save head regions from common dataset
    construct_head_from_boxes(
        common_dataset,
        common_boxes,
        common_indicies,
        config.padding_param_common,
        config.all_data_folder,
        "common",
    )

    # Extract and save head regions from synthetic dataset
    construct_head_from_boxes(
        synthetic_dataset,
        synthetic_boxes,
        synthetic_indicies,
        config.padding_param_syntetic,
        config.all_data_folder,
        "synthetic",
    )


if __name__ == "__main__":
    main()
