# LEGO Style Transfer with JoJoGAN

## Overview

This repository contains an end-to-end pipeline for fine-tuning a StyleGAN-based model using JoJoGAN for LEGO-style image transformation. The project includes data preprocessing, synthetic data generation, and training scripts.

## Features

Face Detection & Preprocessing: Utilizes FaceNet-PyTorch for facial detection and alignment.

Synthetic Data Generation: Uses yandex-art API to generate LEGO-style face images.

JoJoGAN for Style Transfer: Implements JoJoGAN [arXiv:2112.11641](https://arxiv.org/abs/2112.11641) for fine-tuning StyleGAN with minimal training data.

Evaluation Metrics: Supports FID, LPIPS, and Inception Score evaluation. **(TBA)**


## Dataset Preparation

We used both real and synthetic data to enhance model performance. Synthetic images (~150) were generated using yandex-art.

Face Detection: Removing incorrectly formatted or obscured faces using MTCNN.

Embedding Filtering: Sorting detected faces by similarity to an average face embedding to ensure high-quality samples.



# 🏗️ Project Structure  

This repository is structured to efficiently handle **style transfer** using **StyleGAN** and **JoJoGAN**, incorporating data preprocessing, synthetic data generation, and training pipelines.  

## 📂 Directory Overview  

  ```plaintext
style_transfer/
│
├── .git/                     # Git repository metadata
├── .gitignore                # Specifies intentionally untracked files that Git should ignore
├── .pylintrc                 # Pylint configuration file
├── .ruff_cache/              # Cache directory for Ruff linter/formatter
├── .venv/                    # Virtual environment directory
├── data/
│   ├── __init__.py
│   ├── notebooks/
│   │   └── collected_data_eda.ipynb # (Assuming this file exists based on original structure)
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── prepare_dataset.py # Script to prepare the dataset
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── collection_utils.py # Utilities for data collection
│   │       └── face_detector.py    # Face detection utility
│   ├── scripts/
│   │   ├── readme.md               # README for data scripts
│   │   ├── scrapping_lego_script.py # LEGO-style image scraping script
│   │   └── synth_gen.py          # Synthetic data generation script
│   └── utils/
│       ├── __init__.py
│       └── image_dataset.py      # Custom image dataset class
│
├── README.md                 # Project README file
├── requirements.txt          # Python package dependencies
├── scripts/
│   └── stylegan/
│       └── training/
│           └── train.py          # Main StyleGAN training script
│
└── test_input/               # Directory for test inputs
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Nickolas-option/style_transfer
    cd style_transfer
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

The main training script is located at `scripts/stylegan/training/train.py`.

1.  **Ensure the virtual environment is active:**
    ```bash
    source .venv/bin/activate
    ```

2.  **Run the training script:**
    ```bash
    python scripts/stylegan/training/train.py
    ```
    The script uses `pyrallis` for configuration, defined in the `TrainConfig` dataclass within the script. You can override default parameters via command-line arguments if needed (e.g., `python scripts/stylegan/training/train.py --num_iter 1000`). Refer to the `TrainConfig` class for available options.
