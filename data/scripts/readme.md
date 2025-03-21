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
     python synth_gen.py
     ```

   - **Custom Parameters**:
     ```bash
     python synth_gen.py --num_images 5 --concurrent_requests 2 --output_folder ./my_lego_images
     ```

## Command-Line Arguments

- `--num_images`: Number of images to generate (default: 1).
- `--concurrent_requests`: Maximum number of concurrent API requests (default: 3).
- `--output_folder`: Directory where generated images will be saved (default: `./lego_professions`).

## Example

To generate 3 images with 2 concurrent requests and save them in the `./my_lego_images` folder, run:
```bash
python synth_gen.py --num_images 3 --concurrent_requests 2 --output_folder ./my_lego_images
```

## Notes

- Ensure that the `YANDEX_FOLDER_ID` and `YANDEX_AUTH_TOKEN` environment variables are set correctly.
- The script will create the output folder if it does not exist.
- The script will skip generating images if they already exist in the output folder.

--------

# LEGO Minifig Scraper

This script scrapes LEGO minifig images from a given base URL and saves them to a specified output folder. It uses parallel processing to efficiently download images.

## Requirements

- Python 3.6 or later
- `requests` library
- `pyrallis` library
- `concurrent-futures` module (comes with Python)

You can install the required libraries using pip:

```bash
pip install requests pyrallis
```

## Usage

1. **Clone the Repository (if not already done):**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Configure the Script:**

   The script uses a configuration class `Config` which can be modified programmatically or via command line arguments. The configuration options are as follows:
   - `base_url`: The base URL of the images.
   - `theme_prefixes`: A list of theme prefixes for the images.
   - `output_folder`: The folder where the images will be saved.
   - `start_num`: The starting number for the image filenames.
   - `end_num_dict`: A dictionary specifying the end number for each theme prefix.

3. **Run the Script:**

   You can run the script directly from the command line. Here is an example command:

   ```bash
   python scrapping_lego_script.py --base_url="https://example.com/images" --output_folder="lego_minifigs"
   ```

   You can also specify other parameters as needed. For example:

   ```bash
   python scrapping_lego_script.py --base_url="https://example.com/images" --theme_prefixes="cty,agt" --output_folder="lego_minifigs" --start_num=1 --end_num_dict='{"cty": 2800, "agt": 1500}'
   ```

4. **Output:**

   The images will be saved in the specified `output_folder` with filenames based on the theme prefix and number.

## Notes

- The script skips images that already exist in the output folder.
- Images not found (HTTP 404) are skipped with a message.
- Other errors during the download process are printed to the console.

