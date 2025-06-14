#Name: Jeremy Becker

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from custom_text_to_image_generator import SimpleTextEncoder, ConditionalVAE
import pickle

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define local cache path for models
cache_path = r"C:\Users\tonan\Downloads\Project\Project\model_cache"

# Load vocab
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

latent_dim = 256
text_embed_dim = 128

# Load your trained models
text_encoder = SimpleTextEncoder(vocab, embedding_dim=text_embed_dim).to(device)
vae_model = ConditionalVAE(latent_dim=latent_dim, text_embed_dim=text_embed_dim).to(device)

vae_model.load_state_dict(torch.load("vae.pth"))
text_encoder.load_state_dict(torch.load("text_encoder.pth"))

vae_model.eval()
text_encoder.eval()

# Sampling function
def sample_vae(model, text_encoder, text_prompt, latent_dim=256, num_samples=1):
    with torch.no_grad():
        text_embed = text_encoder([text_prompt] * num_samples).to(device)
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decoder(z, text_embed)
        samples = (samples + 1) / 2  # Scale to [0,1]
        return samples.cpu()

# Define diffusion model IDs
model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
model_id2 = "stabilityai/stable-diffusion-xl-base-1.0"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load diffusion pipelines with local cache
pipe1 = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=dtype, cache_dir=cache_path).to(device)
pipe2 = StableDiffusionXLPipeline.from_pretrained(model_id2, torch_dtype=dtype, cache_dir=cache_path).to(device)

# Prompt
#prompt = "anime cat"
#prompt = "pixar style character"
prompt = "rubber hose animation style dog"

# Generate VAE image
vae_images = sample_vae(vae_model, text_encoder, prompt, latent_dim=latent_dim, num_samples=1)
vae_img = vae_images[0].permute(1, 2, 0).numpy()
vae_img = (vae_img * 255).astype(np.uint8)
vae_img = Image.fromarray(vae_img)

# Generate diffusion images
sd_img1 = pipe1(prompt).images[0]
sd_img2 = pipe2(prompt=prompt, added_cond_kwargs={}).images[0]

# Show all images
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].imshow(vae_img)
axs[0].set_title("Your VAE")
axs[0].axis("off")

axs[1].imshow(sd_img1)
axs[1].set_title("Dreamlike Diffusion")
axs[1].axis("off")

axs[2].imshow(sd_img2)
axs[2].set_title("Stable Diffusion XL")
axs[2].axis("off")

plt.tight_layout()
plt.show()