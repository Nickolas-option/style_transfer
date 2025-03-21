
from torch.utils.data import Dataset
import torchvision
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.transforms import ToTensor
from PIL import Image
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional
import pyrallis
from tqdm import trange, tqdm

from data.utils.image_dataset import ImageDataset
from data.preprocessing.utils.face_detector import FaceDetector
from data.preprocessing.utils.collection_utils import get_filtered_indexes, form_masks_and_embeddings, construct_head_from_boxes

@dataclass
class TrainConfig:
    common_data_folder: str = "./lego_minifigs"
    synthetic_data_folder: str = "./lego_professions"
    resize_param_common: Optional[int] = None
    resize_param_synthetic: Optional[int] = None
    filter_percentile: float = 0.75
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
        config.resize_param_common,
        config.common_data_folder,
        "common"
    )

    construct_head_from_boxes(
        synthetic_dataset,
        synthetic_boxes,
        synthetic_indicies,
        config.resize_param_synthetic,
        config.synthetic_data_folder,
        "synthetic"
    )

if __name__ == '__main__':
    main()