import json
import ijson
import sys
from torch.utils.data import IterableDataset
import os
import math
from PIL import Image
from torchvision import transforms
import base64
import random
import numpy as np


def Image_to_str(img):
    return base64.b64encode(img.tobytes()).decode()


def str_to_Image(str, img_size):
    b = base64.b64decode(str)
    img = Image.frombytes("RGB", (img_size), b, "raw")
    img.show()
    return img


def load_metadata_file(file_path):
    metadata = {}
    with open(file_path, "r") as file:
        metadata = json.load(file)
    assert metadata != {}
    return metadata


def load_dataset(dataset_config):
    print("loading stacked dataset...")
    dataset = []
    print()
    counter = 0
    try:
        with open(dataset_config["processed_dataset_path"], "rb") as infile:
            for item in ijson.items(infile, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {counter}")
                dataset.append(item)
                counter += 1
    except FileNotFoundError:
        for root, dirs, _ in os.walk(dataset_config["loose_dataset_path"]): # root: image_collection
            for dir in dirs:
                dir = f"{root}{os.sep}{dir}"
                for obj_root, cam_dirs, _ in os.walk(dir): # root: object (Car/Dino/Piano)
                    for cam_dir in cam_dirs:
                        sys.stdout.write("\033[F")
                        print(f"processing entry {counter}")

                        cam = cam_dir
                        cam_dir = f"{obj_root}{os.sep}{cam_dir}"

                        metadata = load_metadata_file(f"{cam_dir}{os.sep}meta.json")

                        half_final_height = math.floor(dataset_config["final_img_size"][1]/2)

                        left_img = Image.open(
                            f"{cam_dir}{os.sep}left.jpg"
                        ).convert("RGB").resize(
                            (dataset_config["final_img_size"][0], half_final_height)
                        )
                        right_img = Image.open(
                            f"{cam_dir}{os.sep}right.jpg"
                        ).convert("RGB").resize(
                            (dataset_config["final_img_size"][0], half_final_height)
                        )
                        stacked_img = Image.new("RGB", (dataset_config["final_img_size"][0], 2*half_final_height))

                        stacked_img.paste(left_img, (0, 0))
                        stacked_img.paste(right_img, (0, half_final_height))
                        
                        rec = {
                            "id": f"{obj_root.split(os.sep)[-1]}-{cam}",
                            "left_img": Image_to_str(left_img),
                            "left_size": left_img.size,
                            "right_img": Image_to_str(right_img),
                            "right_size": right_img.size,
                            "stacked_img": Image_to_str(stacked_img),
                            "stacked_size": stacked_img.size,
                            "metadata": metadata
                        }
                        dataset.append(rec)
                        counter += 1

                        with open(dataset_config["processed_dataset_path"], "a") as outfile:
                            json.dump(rec, outfile)
                            outfile.write("\n")
    print("stacked dataset loaded.")
    return dataset


class TrainDataset(IterableDataset):
    def __init__(self, config, return_only_images=False, shuffle=False):
        self.config = config
        self.only_images = return_only_images
        self.to_tensor = transforms.Compose([
            transforms.Resize((self.config["final_img_size"][0], self.config["final_img_size"][1])),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.data_list = load_dataset(self.config)
        self.indices = np.arange(len(self.data_list))
        if shuffle:
            random.shuffle(self.indices)

    def __iter__(self): # TODO maybe directly save tensors/embeddings instead of PIL Images?
        for i in self.indices:
            rec = self.data_list[i]
            for type in ["left", "right", "stacked"]:
                img = str_to_Image(rec[f"{type}_img"], rec[f"{type}_size"])
                rec[f"{type}_img"] = self.to_tensor(img)
                del rec[f"{type}_size"]
                
            if self.only_images:
                yield rec["stacked_img"]
            else:
                yield rec
    
    