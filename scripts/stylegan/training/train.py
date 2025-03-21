import torchvision
from PIL import Image
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional
import pyrallis
from tqdm import trange, tqdm
import torch
import torchvision.transforms as transforms
import torchvision.utils as utils

import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
import wandb


from copy import deepcopy

import sys
sys.path.append("./JoJoGAN")
from model import *
from util import align_face
from e4e_projection import projection as e4e_projection

from data.utils.image_dataset import ImageDataset
from data.preprocessing.utils.face_detector import FaceDetector
from data.preprocessing.utils.collection_utils import get_filtered_indexes, form_masks_and_embeddings, construct_head_from_boxes

@dataclass
class TrainConfig:
    dataset_dir: str = "./collected_heads"
    device: str = "cuda"

@pyrallis.wrap()
def main(config: TrainConfig):

    latent_dim = 512
    
    # Load original generator
    original_generator = Generator(1024, latent_dim, 8, 2).to(config.device)
    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    original_generator.load_state_dict(ckpt["g_ema"], strict=False)
    mean_latent = original_generator.mean_latent(10000)

    # to be finetuned generator
    generator = deepcopy(original_generator)

    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    names = [os.path.join(config.dataset_dir, file) for file in os.listdir(config.dataset_dir)]
    
    targets = []
    latents = []

    count = 0
    
    for name in tqdm(names):
        if name.endswith('.pt'):
            continue
        # crop and align the face
        style_aligned = Image.open(name).convert('RGB')

        count += 1
        if count > 100:
            break

        print(name)
    
        # GAN invert
        style_code_path = f"{'.'.join(name.split('.')[:-1])}.pt"
        if not os.path.exists(style_code_path):
            latent = e4e_projection(style_aligned, style_code_path, config.device)
        else:
            latent = torch.load(style_code_path)['latent']
    
        targets.append(transform(style_aligned).to(config.device))
        latents.append(latent.to(config.device))
    
    targets = torch.stack(targets, 0)
    latents = torch.stack(latents, 0)

    alpha =  1.0 #@param {type:"slider", min:0, max:1, step:0.1}
    alpha = 1-alpha
    
    #@markdown Tries to preserve color of original image by limiting family of allowable transformations. Set to false if you want to transfer color from reference image. This also leads to heavier stylization
    preserve_color = False #@param{type:"boolean"}
    #@markdown Number of finetuning steps. Different style reference may require different iterations. Try 200~500 iterations.
    num_iter = 5000 #@param {type:"number"}
    #@markdown Log training on wandb and interval for image logging
    use_wandb = True #@param {type:"boolean"}
    log_interval = 50 #@param {type:"number"}
    
    filename = 'image.jpg' #@param {type:"string"}
    filepath = f'test_input/{filename}'
    
    name = ".".join(filepath.split('.')[:-1])+'.pt'
    
    # aligns and crops face
    aligned_face = align_face(filepath)
    
    my_w = e4e_projection(aligned_face, name, config.device).unsqueeze(0)

    target_im = Image.open(filepath)
    
    if use_wandb:
        wandb.init(project="JoJoGAN")
        config.num_iter = num_iter
        config.preserve_color = preserve_color
        wandb.log(
        {"Style reference": [wandb.Image(target_im)]},
        step=0)
    
    # load discriminator for perceptual loss
    discriminator = Discriminator(1024, 2).eval().to(config.device)
    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    discriminator.load_state_dict(ckpt["d"], strict=False)
    
    # reset generator
    del generator
    generator = deepcopy(original_generator)
    
    g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))
    
    # Which layers to swap for generating a family of plausible real images -> fake image
    if preserve_color:
        id_swap = [9,11,15,16,17]
    else:
        id_swap = list(range(7, generator.n_latent))
    
    for idx in tqdm(range(num_iter)):
        index = np.random.choice(latent.shape[0], 2)
        sample_targets = targets[index]
        sample_latents = latents[index]

        mean_w = generator.get_latent(torch.randn([sample_latents.size(0), latent_dim]).to(config.device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
        in_latent = sample_latents.clone()
        in_latent[:, id_swap] = alpha*sample_latents[:, id_swap] + (1-alpha)*mean_w[:, id_swap]
    
        img = generator(in_latent, input_is_latent=True)
    
        with torch.no_grad():
            real_feat = discriminator(sample_targets)
        fake_feat = discriminator(img)
    
        loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)])/len(fake_feat)
        print(loss)
        
        if use_wandb:
            wandb.log({"loss": loss}, step=idx)
            if idx % log_interval == 0:
                generator.eval()
                my_sample = generator(my_w, input_is_latent=True)
                generator.train()
                print(my_sample.shape)
                my_sample = transforms.ToPILImage()(my_sample[0])
                wandb.log(
                {"Current stylization": [wandb.Image(my_sample)]},
                step=idx)
    
        g_optim.zero_grad()
        loss.backward()
        g_optim.step()
        


if __name__ == "__main__":
    main()