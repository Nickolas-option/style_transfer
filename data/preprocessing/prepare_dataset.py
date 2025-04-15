
from dataclasses import dataclass
from typing import Optional

import pyrallis
import torchvision

from data.preprocessing.utils.collection_utils import (
    construct_head_from_boxes, form_masks_and_embeddings, get_filtered_indexes)
from data.utils.image_dataset import ImageDataset


@dataclass
class TrainConfig:
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
    print("Beginning of collect dataset for tuning StyleGAN")

    common_dataset = ImageDataset(config.common_data_folder, transform=lambda x: torchvision.transforms.ToTensor()(x))
    synthetic_dataset = ImageDataset(config.synthetic_data_folder, transform=lambda x: torchvision.transforms.ToTensor()(x))

    common_boxes, common_embeddings = form_masks_and_embeddings(
        config,
        common_dataset,
        config.resize_param_common,
        "common"
    )

    synthetic_boxes, synthetic_embeddings = form_masks_and_embeddings(
        config,
        synthetic_dataset,
        config.resize_param_synthetic,
        "synthetic"
    )

    common_indicies, synthetic_indicies = get_filtered_indexes(
        common_embeddings,
        synthetic_embeddings,
        percentile = config.filter_percentile
    )

    construct_head_from_boxes(
        common_dataset,
        common_boxes,
        common_indicies,
        config.padding_param_common,
        config.all_data_folder,
        "common"
    )

    construct_head_from_boxes(
        synthetic_dataset,
        synthetic_boxes,
        synthetic_indicies,
        config.padding_param_syntetic,
        config.all_data_folder,
        "synthetic"
    )

if __name__ == '__main__':
    main()