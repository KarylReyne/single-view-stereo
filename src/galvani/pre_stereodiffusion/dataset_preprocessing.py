import json
import os
import sys
import math
from PIL import Image


def load_metadata_file(file_path):
    metadata = {}
    with open(file_path, "r") as file:
        metadata = json.load(file)
    assert metadata != {}
    return metadata


def stack_stereo_images(left_img_path, right_img_path, final_img_size, save_stacked_img=False):
    half_final_height = math.floor(final_img_size[1]/2)

    left_img = Image.open(
        left_img_path
    ).convert("RGB").resize(
        (final_img_size[0], half_final_height)
    )
    right_img = Image.open(
        right_img_path
    ).convert("RGB").resize(
        (final_img_size[0], half_final_height)
    )
    stacked_img = Image.new("RGB", (final_img_size[0], 2*half_final_height))

    stacked_img.paste(left_img, (0, 0))
    stacked_img.paste(right_img, (0, half_final_height))

    if save_stacked_img:
        stacked_img_path = left_img_path.replace("left", "stacked")
        stacked_img.save(stacked_img_path)

    return stacked_img


def precompute_stacked_images(dataset_path="../../data/galvani/fixed_baselines", save_stacked_images=False):
    stacked_images = []
    metadata = []

    for _, dirs, _ in os.walk(dataset_path):
        current = 0
        total = len(dirs)
        skipped = 0
        print()
        for dir in dirs:
            sys.stdout.write("\033[F")
            print(f"processing {dataset_path}{os.sep}{dir}{os.sep}... ({current}/{total}, {skipped} skipped)")

            sample_id = dir.split(os.sep)[-1]
            left_img_path = f"{dataset_path}{os.sep}{dir}{os.sep}{sample_id}_left.png"
            right_img_path = f"{dataset_path}{os.sep}{dir}{os.sep}{sample_id}_right.png"
            metadata_path = f"{dataset_path}{os.sep}{dir}{os.sep}{sample_id}_meta.json"
            try:
                stacked_images.append(stack_stereo_images(
                    left_img_path, 
                    right_img_path, 
                    final_img_size=(512, 1024), 
                    save_stacked_img=save_stacked_images
                ))
                metadata.append(load_metadata_file(
                    metadata_path
                ))
                current += 1
            except FileNotFoundError:
                skipped += 1
                continue

    print(f"successfully loaded {len(stacked_images)} stacked images.")
    return zip(stacked_images, metadata)


if __name__ == '__main__':
    _ = precompute_stacked_images(save_stacked_images=True)
