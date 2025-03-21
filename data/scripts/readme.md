# Lego Figure Image Generator

This script generates Lego figure images with specified professions and emotions using the Yandex Cloud ML SDK.

## Prerequisites

1. **Python 3.7+**: Ensure you have Python 3.7 or later installed.
2. **Yandex Cloud Account**: You need a Yandex Cloud account with the necessary permissions and API access.
3. **Environment Variables**: Set the following environment variables:
   - `YANDEX_FOLDER_ID`: Your Yandex Cloud folder ID.
   - `YANDEX_AUTH_TOKEN`: Your Yandex Cloud authentication token.

## Installation

1. **Install Dependencies**:
   ```bash
   pip install yandex-cloud-ml-sdk pyrallis
   ```

## Usage

1. **Run the Script**:
   You can run the script with default parameters or specify custom parameters using command-line arguments.

   - **Default Parameters** (1 image, 3 concurrent requests, output folder `./lego_professions`):
     ```bash
     python synthetic_gen.py
     ```

   - **Custom Parameters**:
     ```bash
     python synthetic_gen.py --num_images 5 --concurrent_requests 2 --output_folder ./my_lego_images
     ```

## Command-Line Arguments

- `--num_images`: Number of images to generate (default: 1).
- `--concurrent_requests`: Maximum number of concurrent API requests (default: 3).
- `--output_folder`: Directory where generated images will be saved (default: `./lego_professions`).

## Example

To generate 3 images with 2 concurrent requests and save them in the `./my_lego_images` folder, run:
```bash
python synthetic_gen.py --num_images 3 --concurrent_requests 2 --output_folder ./my_lego_images
```

## Notes

- Ensure that the `YANDEX_FOLDER_ID` and `YANDEX_AUTH_TOKEN` environment variables are set correctly.
- The script will create the output folder if it does not exist.
- The script will skip generating images if they already exist in the output folder.
