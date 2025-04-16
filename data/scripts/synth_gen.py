#!/usr/bin/env python3
"""
Lego Figure Image Generator

This script generates Lego figure images with specified professions and emotions
using the Yandex Cloud ML SDK. It allows for concurrent generation of multiple images
and handles API rate limiting through a semaphore.

Usage:
    python synth_gen.py [--num_images N] [--concurrent_requests N] [--output_folder PATH]

Environment Variables:
    YANDEX_FOLDER_ID: Your Yandex Cloud folder ID
    YANDEX_AUTH_TOKEN: Your Yandex Cloud authentication token
"""
import asyncio
import os
import pathlib
import random
import time
from dataclasses import dataclass, field

import pyrallis
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML


@dataclass
class Config:
    """Configuration for generating Lego figure images"""

    num_images: int = field(
        default=1, metadata={"help": "Number of images to generate"}
    )
    concurrent_requests: int = field(
        default=3, metadata={"help": "Maximum number of concurrent requests"}
    )
    output_folder: str = field(
        default="./lego_professions3", metadata={"help": "Output folder for images"}
    )
    folder_id: str = field(
        default=os.getenv("YANDEX_FOLDER_ID"),
        metadata={"help": "Yandex Cloud folder ID"},
    )
    auth_token: str = field(
        default=os.getenv("YANDEX_AUTH_TOKEN"),
        metadata={"help": "Authentication token"},
    )


# List of professions to randomly select from when generating images
professions = [
    "механик",
    "программист",
    "повар",
    "врач",
    "учитель",
    "строитель",
    "пожарный",
    "полицейский",
    "пилот",
    "стюардесса",
    "бухгалтер",
    "юрист",
    "адвокат",
    "судья",
    "журналист",
    "фотограф",
    "художник",
    "музыкант",
    "актер",
    "режиссер",
    "писатель",
    "ученый",
    "инженер",
    "архитектор",
    "дизайнер",
    "парикмахер",
    "визажист",
    "фермер",
    "садовник",
    "ветеринар",
    "зоолог",
    "биолог",
    "химик",
    "физик",
    "астроном",
    "археолог",
    "историк",
    "психолог",
    "социолог",
    "экономист",
    "маркетолог",
    "менеджер",
    "предприниматель",
    "водитель",
    "таксист",
    "курьер",
    "продавец",
    "библиотекарь",
]

# List of emotions to randomly select from when generating images
emotions = ["happy", "sad", "angry", "surprised", "fearful"]


async def generate_image(
    model, profession: str, emotion: str, output_folder: pathlib.Path
) -> None:
    """
    Generate a Lego figure image with specified profession and emotion.

    Args:
        model: The image generation model instance
        profession: The profession of the Lego figure
        emotion: The emotion to be displayed by the Lego figure
        output_folder: Directory where the generated image will be saved
    """
    message = f"Lego figure {profession} {emotion} unobstructed face UltraHD"
    seed_value = random.randint(1, 2**32)
    model_configured = model.configure(seed=seed_value)

    output_path = output_folder / f"lego_person_{profession}_{emotion}.jpeg"

    if output_path.exists():
        print(f"Image for {profession} (emotion: {emotion}) already exists, skipping")
        return

    try:
        operation = model_configured.run_deferred(message)
        start_time = time.time()
        timeout = 300  # 5 minutes timeout

        while True:
            if time.time() - start_time > timeout:
                print(f"Timeout for {profession} (emotion: {emotion}), cancelling")
                return

            try:
                result = operation.wait(timeout=1)
                break
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(2)

        output_path.write_bytes(result.image_bytes)
        print(f"Generated image for {profession} (emotion: {emotion})")
    except Exception as e:
        print(f"Failed to generate image for {profession} (emotion: {emotion}): {e}")


async def main(
    num_images: int = 3,
    concurrent_requests: int = 3,
    output_folder: str = "./lego_professions",
    folder_id: str = "",
    auth_token: str = "",
) -> None:
    """
    Main function to generate multiple Lego figure images concurrently.

    Args:
        num_images: Number of images to generate
        concurrent_requests: Maximum number of concurrent API requests
        output_folder: Directory where generated images will be saved
        folder_id: Yandex Cloud folder ID
        auth_token: Authentication token for Yandex Cloud
    """
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    sdk = YCloudML(folder_id=folder_id, auth=auth_token)
    model = sdk.models.image_generation("yandex-art")

    tasks = []
    for _ in range(num_images):
        profession = random.choice(professions)
        emotion = random.choice(emotions)
        tasks.append(generate_image(model, profession, emotion, output_folder))

    semaphore = asyncio.Semaphore(concurrent_requests)

    async def bounded_generate(coro):
        """
        Wrapper function to limit concurrent execution using a semaphore.
        
        Args:
            coro: The coroutine to execute with the semaphore
            
        Returns:
            The result of the coroutine execution
        """
        async with semaphore:
            return await coro

    bounded_tasks = [bounded_generate(task) for task in tasks]

    await asyncio.gather(*bounded_tasks)
    print(f"Completed image generation! Check your images in {output_folder}")


def run():
    """
    Entry point function that parses configuration and runs the main async function.
    Loads environment variables from .env file if present.
    """
    load_dotenv()
    cfg = pyrallis.parse(config_class=Config)
    asyncio.run(
        main(
            num_images=cfg.num_images,
            concurrent_requests=cfg.concurrent_requests,
            output_folder=cfg.output_folder,
            folder_id=cfg.folder_id,
            auth_token=cfg.auth_token,
        )
    )


if __name__ == "__main__":
    run()
