import torch
from torch import nn
import torch.optim as optim
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys.path.append('..')
from dataset import TrainDataset
from unet import UNet
from vae import VAEEncoder, VAEDecoder, VAE
# from misc_util import get_config


def get_config(path='cfg/config.json'):
    config = None
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    assert config != None
    return config


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def linear_beta_schedule(timesteps, beta_range):
    return torch.linspace(beta_range[0], beta_range[1], timesteps)


def forward_diffusion_process(x, t, noise_schedule, device):
    beta_t = noise_schedule[t].to(device)
    normal_noise = torch.randn_like(x).to(device)  # noise ~ N(0, I)
    # for t in [torch.sqrt(1-beta_t), x]: print(t.shape)
    # print(beta_t)
    # x_scaled = torch.clone(x)
    # noise_scaled = torch.clone(normal_noise)
    # for i in range(x.shape[0]):
    #     torch.sqrt(1-beta_t[i])*x_scaled[i]
    #     torch.sqrt(beta_t[i])*noise_scaled[i]
    # a = x_scaled
    # b = noise_scaled
    a = torch.sqrt(1-beta_t)*x # [b]*[b c h w]
    b = torch.sqrt(beta_t)*normal_noise
    return a+b


def reverse_diffusion_step(x, t, noise_schedule, device):
    beta_t = noise_schedule[t].to(device)
    normal_noise = torch.randn_like(x).to(device)  # noise ~ N(0, I)
    a = x-torch.sqrt(beta_t)*normal_noise # [b c h w]*[b c h w]
    b = torch.sqrt(1-beta_t)
    return a/b


def plot_noisy_images(image, noise_schedule, device, steps=[0, 100, 500, 999]):
    _, axes = plt.subplots(1, len(steps), figsize=(15, 5))
    for i, t in enumerate(steps):
        noisy_image = forward_diffusion_process(image, t, noise_schedule, device)
        axes[i].imshow(noisy_image.permute(1, 2, 0).numpy())
        axes[i].set_title(f"Timestep {t}")
        axes[i].axis("off")
    plt.show()


def save_samples(model, epoch, dataloader, beta_schedule, timesteps, device, save_dir="./out/"):
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            save_image(images, f"{save_dir}batch_{batch_idx}_input_epoch_{epoch}.png")
            t = torch.randint(0, timesteps, (images.size(0),))
            noisy_images = forward_diffusion_process(images, t, beta_schedule, device)
            denoised_images = model(noisy_images)  # Predicted clean images
            save_image(denoised_images, f"{save_dir}batch_{batch_idx}_generated_epoch_{epoch}.png")
            break
    model.train()


def main(training_config, dataset_config):

    set_seeds(training_config["seed"])

    dataset = TrainDataset(dataset_config, return_only_images=True)
    dataloader = DataLoader(dataset, batch_size=training_config["batch_size"], shuffle=True)

    loss_function = nn.MSELoss()

    model = UNet(in_channels=3, out_channels=3).to(training_config["device"])
    optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])

    epochs = training_config["num_train_epochs"]
    timesteps = 1000
    beta_schedule = linear_beta_schedule(timesteps, beta_range=training_config["beta_range"])

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(training_config["device"]) # [b c h w]

            # Forward diffusion: Add noise
            t = torch.randint(0, timesteps, (images.size(0),))  # Random timestep for each image
            noisy_images = forward_diffusion_process(images, t, beta_schedule, training_config["device"])
            noise = torch.randn_like(images)  # True noise

            # UNet prediction
            predicted_noise = model(noisy_images)

            # Compute loss
            loss = loss_function(predicted_noise, noise)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")

        save_samples(model, epoch, dataloader, beta_schedule, timesteps, training_config["device"])


if __name__ == "__main__":
    training_config = get_config(path="cfg/training_config.json")
    dataset_config = get_config(path="cfg/dataset_config.json")
    main(training_config, dataset_config)