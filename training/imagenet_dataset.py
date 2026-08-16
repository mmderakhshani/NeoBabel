# This work is based in part on code from Show-o (https://github.com/showlab/Show-o).
# Significant modifications for NeoBabel.

import collections
import os
from typing import Any, Callable, Optional

import torch
from torchvision.datasets.folder import DatasetFolder, default_loader
from training.utils import image_transform
import random


class ImageNetDataset(DatasetFolder):
    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        image_size=256,
    ):
        IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

        self.transform = image_transform
        self.image_size = image_size

        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=self.transform,
            target_transform=None,
            is_valid_file=is_valid_file,
        )

        self.destination_language = ["english", "french", "hindi", "persian", "dutch", "mandarin"]

        self.labels = {}
        loaded_languages = []
        for language in self.destination_language:
            mapping_path = f'./training/imagenet_label_mapping_{language}.txt'
            if not os.path.exists(mapping_path):
                print(f"Warning: {mapping_path} not found, skipping '{language}' class labels.")
                continue
            loaded_languages.append(language)
            with open(mapping_path, 'r') as f:
                for l in f:
                    num, description = l.split(":", 1)
                    if self.labels.get(int(num)) is None:
                        self.labels[int(num)] = [description.strip()]
                    else:
                        self.labels[int(num)].append(description.strip())

        if not self.labels:
            raise FileNotFoundError(
                "No imagenet_label_mapping_<language>.txt files found under ./training/ "
                f"(looked for languages: {self.destination_language})"
            )

        print(f"ImageNet dataset loaded with class labels in: {loaded_languages}")

    def __getitem__(self, idx):

        try:
            path, target = self.samples[idx]
            image = self.loader(path)
            image = self.transform(image, resolution=self.image_size)
            input_ids = "{}".format(random.choice(self.labels[target]))
            class_ids = torch.tensor(target)
            # print(f"Sanity Check: {input_ids}")

            return {'images': image, 'input_ids': input_ids, 'class_ids': class_ids}

        except Exception as e:
            print(e)
            return self.__getitem__((idx + 1) % len(self.samples))

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('input_ids'):
                batched[k] = torch.stack(v, dim=0)

        return batched


if __name__ == '__main__':
    pass
