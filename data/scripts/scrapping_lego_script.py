import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests
from pyrallis import field, parse


@dataclass
class Config:
    base_url: str = field(default="")
    theme_prefixes: list = field(default=["cty", "agt", "cas", "pi", "que", "twn", "trn", "pln", "soc", "nba", "chef", "hol", "oct", "par", "wc", "firec", "hgh", "ovr", "zip", "but", "air", "wtr"])
    output_folder: str = field(default="lego_minifigs")
    start_num: int = field(default=1)
    end_num_dict: dict = field(default_factory=lambda: {"cty": 2800, "default": 999})


def download_image(base_url, theme_prefix, num, output_folder):
    """
    Download an image from a given URL and save it to the specified output folder.

    Parameters:
    base_url (str): The base URL of the image.
    theme_prefix (str): The theme prefix of the image file name.
    num (int): The number to be used in the file name.
    output_folder (str): The folder path where the image will be saved.
    """
    img_name = f"{theme_prefix}{num:04d}-bl.webp?v=6" if theme_prefix == "cty" else f"{theme_prefix}{num:03d}-bl.webp?v=6"
    img_url = f"{base_url}/{img_name}"
    img_path = os.path.join(output_folder, img_name.split('?')[0])  # Remove query parameters from filename

    if os.path.exists(img_path):
        print(f"Skipping (already exists): {img_name}")
        return

    try:
        # Send a request to the image URL
        response = requests.get(img_url, stream=True, timeout=10)

        if response.status_code == 200:
            with open(img_path, 'wb') as img_file:
                for chunk in response.iter_content(1024):
                    img_file.write(chunk)
            print(f"Downloaded: {img_name}")
        elif response.status_code == 404:
            print(f"Image not found: {img_name} (Skipping...)")
        else:
            print(f"Failed to download {img_name}. Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {img_name}: {e}")


def download_images_in_parallel(base_url, theme_prefix, start_num, end_num, output_folder, max_workers=32):
    """
    Download multiple images in parallel from a given URL range and save them to the specified output folder.

    Parameters:
    base_url (str): The base URL of the images.
    theme_prefix (str): The theme prefix of the image file names.
    start_num (int): The starting number in the range of numbers to be used in the file names.
    end_num (int): The ending number in the range of numbers to be used in the file names.
    output_folder (str): The folder path where the images will be saved.
    max_workers (int): The maximum number of threads to use for downloading images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_image, base_url, theme_prefix, num, output_folder)
            for num in range(start_num, end_num + 1)
        ]

        for future in as_completed(futures):
            future.result() 


def main(cfg: Config):
    if not os.path.exists(cfg.output_folder):
        os.makedirs(cfg.output_folder)

    for theme_prefix in cfg.theme_prefixes:
        end_num = cfg.end_num_dict.get(theme_prefix, cfg.end_num_dict['default'])
        download_images_in_parallel(cfg.base_url, theme_prefix, cfg.start_num, end_num, cfg.output_folder)

if __name__ == "__main__":
    cfg = parse(Config)
    main(cfg)