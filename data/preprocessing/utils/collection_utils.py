import os

import numpy as np
from torchvision.utils import save_image
from tqdm import trange

from data.preprocessing.utils.face_detector import FaceDetector


def form_masks_and_embeddings(config, dataset, resize_param, dataset_label):
    """
    Detect faces in a dataset and extract their coordinates and embeddings.
    
    Arguments:
        config: Configuration object containing device information
        dataset: The dataset containing images to process
        resize_param: Parameter for resizing images during face detection
        dataset_label: Label for the dataset (used for progress bar)
        
    Returns:
        result_boxes: Dictionary mapping image indices to face bounding boxes
        result_embeddings: List of tuples containing (index, embedding) pairs
    """
    curr_face_detector = FaceDetector(config.device, resize_param)
    result_embeddings = []
    result_boxes = {}

    for k in trange(len(dataset), desc=f"Recognize faces on {dataset_label} dataset"):
        recognized_items = curr_face_detector.get_coords_and_embeds(dataset[k])
        if recognized_items is not None:
            result_embeddings.append((k, recognized_items[1]))
            result_boxes[k] = recognized_items[0]

    return result_boxes, result_embeddings


def get_filtered_indexes(common_data_pairs, syntetic_data_pairs, percentile=0.9):
    """
    Filter augmented and non-augmented data based on their face embeddings.
    
    This function normalizes face embeddings and filters out outliers based on
    the vector norm, keeping only the specified percentile of samples.
    
    Arguments:
        common_data_pairs: List of (index, embedding) pairs for original data
        syntetic_data_pairs: List of (index, embedding) pairs for synthetic/augmented data
        percentile: Fraction of data to keep (default: 0.9)
        
    Returns:
        Tuple of (filtered_common_indexes, filtered_synthetic_indexes)
    """
    common_data_indexes, common_data_vectors = zip(*common_data_pairs)
    syntetic_data_indexes, syntetic_data_vectors = zip(*syntetic_data_pairs)

    common_data_indexes = np.array(common_data_indexes)
    syntetic_data_indexes = np.array(syntetic_data_indexes)

    common_data_vectors = np.array(common_data_vectors)
    syntetic_data_vectors = np.array(syntetic_data_vectors)

    face_detected_indexes = np.hstack((common_data_indexes, syntetic_data_indexes))

    face_detected_embeddings = np.vstack(
        (np.array(common_data_vectors), np.array(syntetic_data_vectors))
    )

    face_detected_embeddings -= np.mean(face_detected_embeddings, axis=0)[None, :]
    face_detected_norms = np.linalg.norm(face_detected_embeddings, axis=1)

    final_filter_indexes = np.argsort(face_detected_norms)[
        : int(percentile * face_detected_indexes.shape[0])
    ]

    fpart = final_filter_indexes[final_filter_indexes < common_data_indexes.shape[0]]
    spart = final_filter_indexes[final_filter_indexes >= common_data_indexes.shape[0]]

    return face_detected_indexes[fpart], face_detected_indexes[spart]


def construct_head_from_boxes(
    dataset, boxes, indicies, padding_param, dataset_save_dir, dataset_label
):
    """
    Crop face regions from images based on bounding boxes and save them.
    
    This function extracts face regions from images using the provided bounding boxes,
    applies padding, and saves the cropped faces as individual image files.
    
    Arguments:
        dataset: The dataset containing images
        boxes: Dictionary mapping image indices to face bounding boxes
        indicies: List of indices to process
        padding_param: Amount of padding to add around the face region
        dataset_save_dir: Directory to save the cropped face images
        dataset_label: Label to use in the saved filenames
    """
    if not os.path.exists(dataset_save_dir):
        os.makedirs(dataset_save_dir)

    for index in indicies:
        coords = boxes[index]
        coords[coords < 0] = 0

        x, y, x1, y1 = coords[0], coords[1], coords[2], coords[3]

        if x > x1:
            x, x1 = x1, x
        if y > y1:
            y, y1 = y1, y

        image = dataset[index]

        x1 = min(x1 + padding_param * 2, image.shape[2] - 1)
        y1 = min(y1 + padding_param * 2, image.shape[1] - 1)

        image = image[:, int(y) : int(y1), int(x) : int(x1)]

        save_path = os.path.join(dataset_save_dir, f"{dataset_label}_{index}.png")
        save_image(image, save_path)
