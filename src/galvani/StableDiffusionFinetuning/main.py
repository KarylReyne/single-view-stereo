import torch
import sys
import json
from diffusers import StableDiffusionPipeline
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('..')
from dataset import TrainDataset
# from misc_util import get_config


def get_config(path='cfg/config.json'):
    config = None
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    assert config != None
    return config


class VAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(VAEEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, latent_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.encoder(x)


def main(training_config, dataset_config):
    pipeline = StableDiffusionPipeline.from_pretrained(
        training_config["model_path"],
        torch_dtype=torch.float16
    ).to(training_config["device"])
    
    accelerator = Accelerator()
    pipeline.enable_attention_slicing()

    dataset = TrainDataset(dataset_config, return_only_images=True, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=training_config["batch_size"])

    print(pipeline)

    pipeline.unet.train()
    if training_config["compile_pipeline"]:
        torch.compile(pipeline.unet)

    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=training_config["learning_rate"])
    criterion = torch.nn.MSELoss()

    for epoch in range(training_config["num_train_epochs"]):
        print(f"Epoch {epoch + 1}/{training_config['num_train_epochs']}")

        for batch in tqdm(dataloader):
            images = batch[0].to(training_config["device"])
            noise = torch.randn_like(images).to(training_config["device"])

            outputs = pipeline.unet(images, noise)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    training_config = get_config(path="cfg/training_config.json")
    dataset_config = get_config(path="cfg/dataset_config.json")
    main(training_config, dataset_config)
