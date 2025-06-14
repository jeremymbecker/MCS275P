#Name: Jeremy Becker

import os
import random
import json
from pathlib import Path
import requests
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from PIL import Image
from torch.utils.data import Dataset
import pickle

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import matplotlib.pyplot as plt
from diffusers.utils import make_image_grid

from datasets import load_dataset 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- Helper Function: Download images from URLs ------------

def download_images(captions_file, save_dir="./laion_sample_images", max_images=100):
    os.makedirs(save_dir, exist_ok=True)

    with open(captions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, item in tqdm(enumerate(data), total=min(max_images, len(data)), desc="Downloading images"):
        url = item["url"]
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
                img_path = os.path.join(save_dir, f"{i:04d}.jpg")
                with open(img_path, "wb") as img_file:
                    img_file.write(response.content)
        except Exception as e:
            print(f"Failed to download image {i}: {e}")

# ----------- Dataset and Augmentation ------------

def get_transform(img_size=128):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2()
    ])

class CustomImageCaptionDataset(Dataset):
    def __init__(self, images_dir, captions_file, transform=None, max_caption_len=7):
        self.images_dir = images_dir
        self.transform = transform
        self.max_caption_len = max_caption_len

        # Load captions JSON
        with open(captions_file, "r", encoding="utf-8") as f:
            self.captions_data = json.load(f)

        # Assume captions_data is a list with each item having "caption" and images saved as 0000.jpg, 0001.jpg, ...
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        if self.transform:
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"].float()  # <- make sure float
            # Normalize to [-1, 1]
            img_tensor = img_tensor * 2 - 1
        else:
            img_tensor = ToTensorV2()(image=img_np)["image"].float() * 2 - 1

        # Get captions matching this image file (idx maps directly to captions_data idx)
        captions_list = []
        if idx < len(self.captions_data):
            caption_text = self.captions_data[idx].get("caption", "")
            if caption_text and len(caption_text.split()) <= self.max_caption_len:
                captions_list.append(caption_text)

        # Pick a random caption if multiple, else empty string
        text_label = random.choice(captions_list) if captions_list else ""

        return img_tensor, text_label

# ----------- Simple Text Encoder ------------

class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab, embedding_dim=128, max_len=7):
        super().__init__()
        self.vocab = vocab
        self.word2idx = {w:i+1 for i, w in enumerate(vocab)}  # 0 for padding
        self.embedding = nn.Embedding(len(vocab)+1, embedding_dim, padding_idx=0)
        self.max_len = max_len
        self.embedding_dim = embedding_dim

    def forward(self, text_list):
        batch_indices = []
        for txt in text_list:
            tokens = txt.lower().split()[:self.max_len]
            indices = [self.word2idx.get(t, 0) for t in tokens]
            indices += [0] * (self.max_len - len(indices))
            batch_indices.append(indices)
        batch_indices = torch.tensor(batch_indices).to(device)  # (B, max_len)
        embeds = self.embedding(batch_indices)  # (B, max_len, embedding_dim)
        embeds = embeds.mean(dim=1)  # (B, embedding_dim)
        return embeds

# ----------- Conditional VAE ------------

class Encoder(nn.Module):
    def __init__(self, latent_dim=256, text_embed_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 64x64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 128x32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 256x16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),  # 512x8x8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(512*8*8 + text_embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(512*8*8 + text_embed_dim, latent_dim)

    def forward(self, x, text_embed):
        x = self.conv(x)  # (B, 512*8*8)
        x = torch.cat([x, text_embed], dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=256, text_embed_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim + text_embed_dim, 512*8*8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 256x16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128x32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64x64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 3x128x128
            nn.Tanh()
        )

    def forward(self, z, text_embed):
        x = torch.cat([z, text_embed], dim=1)
        x = self.fc(x)
        x = x.view(-1, 512, 8, 8)
        x = self.deconv(x)
        return x

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=256, text_embed_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim, text_embed_dim)
        self.decoder = Decoder(latent_dim, text_embed_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, text_embed):
        mu, logvar = self.encoder(x, text_embed)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, text_embed)
        return recon, mu, logvar

# ----------- Loss ------------

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

# ----------- Training ------------

def train(model, text_encoder, dataloader, optimizer, epochs=10):
    model.train()
    text_encoder.train()

    for epoch in range(epochs):
        total_loss = 0
        for imgs, texts in dataloader:
            imgs = imgs.to(device)
            text_embeds = text_encoder(texts).to(device)

            optimizer.zero_grad()
            recon_imgs, mu, logvar = model(imgs, text_embeds)
            loss, recon_loss, kl_loss = vae_loss(recon_imgs, imgs, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# ----------- Sampling ------------

def sample(model, text_encoder, text_prompt, latent_dim=256, num_samples=4):
    model.eval()
    text_encoder.eval()
    with torch.no_grad():
        text_embed = text_encoder([text_prompt]*num_samples).to(device)
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decoder(z, text_embed)
        samples = (samples + 1) / 2  # scale back to [0, 1]
        samples = samples.cpu()
    return samples

# ----------- Utility to show images ------------

def show_images(images, rows=1, cols=4):
    pil_images = [Image.fromarray((img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)) for img in images]
    grid = make_image_grid(pil_images, rows=rows, cols=cols)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

# ----------- Main ------------

def main():
    img_size = 128
    batch_size = 32
    epochs = 20
    latent_dim = 256
    text_embed_dim = 128

    # Load LAION-400M metadata (no images yet)
    dataset = load_dataset("laion/laion400m", split="train")

    # Sample 100 items
    sampled = dataset.shuffle(seed=42).select(range(100))

    # ✅ Fixed: Save captions and URLs using correct keys ('url' and 'caption')
    data_to_save = []
    for item in sampled:
        url = item.get("url")
        caption = item.get("caption")
        if url and caption:
            data_to_save.append({"url": url, "caption": caption})
        else:
            print("Skipping item due to missing 'url' or 'caption':", item)

    captions_file = "laion_sample_captions.json"
    with open(captions_file, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)

    # Download images from URLs
    download_images(captions_file, save_dir="./laion_sample_images", max_images=100)

    # Create dataset and dataloader
    transform = get_transform(img_size)
    dataset = CustomImageCaptionDataset("./laion_sample_images", captions_file, transform=transform)

    # Build vocabulary from captions
    all_texts = []
    for _, txt in dataset:
        all_texts.extend(txt.lower().split())
    vocab = list(set(all_texts))
    print(f"Vocabulary size: {len(vocab)}")

    # Save vocab for future use
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    text_encoder = SimpleTextEncoder(vocab, embedding_dim=text_embed_dim).to(device)
    model = ConditionalVAE(latent_dim, text_embed_dim).to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(text_encoder.parameters()), lr=1e-3)

    train(model, text_encoder, dataloader, optimizer, epochs=epochs)

    torch.save(model.state_dict(), "vae.pth")
    torch.save(text_encoder.state_dict(), "text_encoder.pth")

    prompts = ["anime cat", "pixar style character", "cartoon dog", "happy child"]
    for prompt in prompts:
        print(f"Sampling for prompt: '{prompt}'")
        samples = sample(model, text_encoder, prompt, latent_dim=latent_dim, num_samples=4)
        show_images(samples, rows=1, cols=4)
    

if __name__ == "__main__":
    main()