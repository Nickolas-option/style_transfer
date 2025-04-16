"""
StyleGAN fine-tuning with JoJoGAN for style transfer.

This module implements the JoJoGAN training pipeline, which fine-tunes
a pre-trained StyleGAN model to perform style transfer with minimal data.
It includes configuration, data processing, training, and evaluation.
"""

import os
import sys
import random
from copy import deepcopy
from dataclasses import dataclass
from PIL import Image

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import lpips
import wandb
import pyrallis
from tqdm import tqdm

# Add JoJoGAN to path and import modules
sys.path.append("./JoJoGAN")
try:
    from JoJoGAN.model import Generator, Discriminator
    from JoJoGAN.util import align_face
    from JoJoGAN.e4e_projection import projection as e4e_projection
except ImportError:
    # Fallback to direct import if package structure is different
    from model import Generator, Discriminator
    from util import align_face
    from e4e_projection import projection as e4e_projection


@dataclass
class TrainConfig:
    """Configuration for StyleGAN fine-tuning with JoJoGAN.
    
    This dataclass defines all parameters needed for the training process,
    including paths, hyperparameters, and logging settings.
    """
    dataset_dir: str = "./collected_heads"  # Directory containing target style images
    device: str = "cuda"  # Device to run training on (cuda or cpu)
    stylegan_ckpt: str = "models/stylegan2-ffhq-config-f.pt"  # Path to StyleGAN checkpoint
    test_img_dir: str = "test_input"  # Directory containing test images for evaluation
    alpha: float = 0.0  # Controls stylization strength (1-alpha)
    preserve_color: bool = False  # Whether to preserve original colors
    num_iter: int = 5000  # Number of training iterations
    log_interval: int = 50  # Interval for logging metrics and images
    use_wandb: bool = True  # Whether to use Weights & Biases for logging
    latent_dim: int = 512  # Dimension of the latent space
    lr: float = 2e-3  # Learning rate for optimizer
    wandb_project: str = "JoJoGAN"  # W&B project name
    wandb_group: str = "test_jojogan"  # W&B group name
    wandb_name: str = "logging"  # W&B run name
    seed: int = 52  # Random seed for reproducibility


@pyrallis.wrap()
def main(config: TrainConfig):
    """Main training function for JoJoGAN style transfer.
    
    This function implements the JoJoGAN training pipeline, which fine-tunes
    a pre-trained StyleGAN model to perform style transfer with minimal data.
    
    Args:
        config: Configuration parameters for the training process
    """
    # ----------------------- Reproducibility -----------------------
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ----------------------- Load Generator -----------------------
    original_generator = Generator(1024, config.latent_dim, 8, 2).to(config.device)
    ckpt = torch.load(config.stylegan_ckpt, map_location="cpu")
    original_generator.load_state_dict(ckpt["g_ema"], strict=False)
    original_generator.mean_latent(10000)

    generator = deepcopy(original_generator)

    # ----------------------- Image Preprocessing -----------------------
    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    names = [
        os.path.join(config.dataset_dir, f) for f in os.listdir(config.dataset_dir)
    ]
    targets, latents = [], []

    # Process target style images and their latent codes
    for i, name in enumerate(tqdm(names, desc="Processing style images")):
        if name.endswith(".pt") or i >= 5:
            continue

        img = Image.open(name).convert("RGB")
        latent_path = f"{'.'.join(name.split('.')[:-1])}.pt"

        if not os.path.exists(latent_path):
            latent = e4e_projection(img, latent_path, config.device)
        else:
            latent = torch.load(latent_path)["latent"]

        targets.append(transform(img).to(config.device))
        latents.append(latent.to(config.device))

    targets = torch.stack(targets)
    latents = torch.stack(latents)

    # ----------------------- Load Discriminator -----------------------
    discriminator = Discriminator(1024, 2).eval().to(config.device)
    discriminator.load_state_dict(ckpt["d"], strict=False)

    # ----------------------- Inversion of Input Image -----------------------
    test_img_paths = [
        os.path.join(config.test_img_dir, f)
        for f in os.listdir(config.test_img_dir)
        if f.endswith((".jpg", ".png"))
    ]

    # ----------------------- Initialize wandb -----------------------
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            group=config.wandb_group,
            name=config.wandb_name,
            config=config,
        )

    # ----------------------- Initialize LPIPS -----------------------
    lpips_metric = lpips.LPIPS(net="alex").to(config.device)

    # ----------------------- Training Setup -----------------------
    generator = deepcopy(original_generator)
    g_optim = optim.Adam(generator.parameters(), lr=config.lr, betas=(0, 0.99))
    
    # Define which layers to swap for style transfer
    id_swap = (
        [9, 11, 15, 16, 17]
        if config.preserve_color
        else list(range(7, generator.n_latent))
    )

    # ----------------------- Metrics Setup -----------------------
    fid = FrechetInceptionDistance(feature=2048).to(config.device)
    inception = InceptionScore().to(config.device)

    # ----------------------- Training Loop -----------------------
    for step in tqdm(range(config.num_iter), desc="Training progress"):
        idx = np.random.choice(latents.shape[0], 2)
        sample_targets = targets[idx]
        sample_latents = latents[idx]

        # Generate mean latent vector for style mixing
        mean_w = generator.get_latent(
            torch.randn([2, config.latent_dim]).to(config.device)
        )
        mean_w = mean_w.unsqueeze(1).repeat(1, generator.n_latent, 1)

        # Mix latents for style transfer
        in_latent = sample_latents.clone()
        in_latent[:, id_swap] = (1 - config.alpha) * sample_latents[
            :, id_swap
        ] + config.alpha * mean_w[:, id_swap]

        # Generate images with the current generator
        img = generator(in_latent, input_is_latent=True)

        # Extract features using the discriminator
        with torch.no_grad():
            real_feat = discriminator(sample_targets)
        fake_feat = discriminator(img)

        # Calculate feature matching loss
        loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]) / len(
            fake_feat
        )
        print(f"Step {step} - Loss: {loss.item()}")

        if config.use_wandb:
            wandb.log({"loss": loss.item()}, step=step)

            if step % config.log_interval == 0:
                # Log stylization for multiple test images
                generator.eval()

                # Initialize a variable to accumulate LPIPS scores
                total_lpips_score = 0
                num_images = len(test_img_paths)

                # Process each test image
                for test_img_path in test_img_paths:
                    if test_img_path.endswith(".pt"):
                        continue
                    aligned_face = align_face(test_img_path)
                    latent_path = f"{'.'.join(test_img_path.split('.')[:-1])}.pt"
                    my_w = e4e_projection(
                        aligned_face, latent_path, config.device
                    ).unsqueeze(0)

                    # Generate stylized image
                    my_sample = generator(my_w, input_is_latent=True)

                    # Log stylized image
                    img_pil = transforms.ToPILImage()(
                        my_sample[0].cpu().clamp(-1, 1) * 0.5 + 0.5
                    )
                    wandb.log(
                        {
                            "Stylized Output": [
                                wandb.Image(
                                    img_pil, caption=f"Stylized {test_img_path}"
                                )
                            ]
                        },
                        step=step,
                    )

                    # Compute LPIPS score for the current test image
                    target_img = Image.open(test_img_path).convert("RGB")
                    target_img = transform(target_img).unsqueeze(0).to(config.device)
                    lpips_score = lpips_metric(target_img, my_sample).item()
                    total_lpips_score += lpips_score

                # Compute and log the mean LPIPS score for all test images
                mean_lpips_score = total_lpips_score / num_images
                wandb.log({"Mean LPIPS": mean_lpips_score}, step=step)

                # Log metrics
                fid.update((img * 255).to(torch.uint8), real=False)
                fid.update((sample_targets * 255).to(torch.uint8), real=True)

                inception.update((img * 255).to(torch.uint8))

                fid_score = fid.compute().item()
                inception_score = inception.compute()[0].item()

                wandb.log(
                    {
                        "FID": fid_score,
                        "Inception Score": inception_score,
                    },
                    step=step,
                )

        # Update generator weights
        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

    # Save the final model
    if config.use_wandb:
        torch.save(generator.state_dict(), "finetuned_generator.pt")
        wandb.save("finetuned_generator.pt")
        wandb.finish()


if __name__ == "__main__":
    main()
