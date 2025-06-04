import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter
import random

from PIL import Image
from scipy.ndimage import label

from .preprocessing import normalize_mask, uint_to_float, reshape_array

class ThermalDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str = None,
        height: int = 240,
        width: int = 320,
        multiclass: bool = False,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(
            [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
        )
        self.mask_files = (
            sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])
            if masks_dir is not None
            else None
        )
        self.width = width
        self.height = height
        self.multiclass = multiclass
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if self.mask_files is not None:
            assert len(self.image_files) == len(
                self.mask_files
            ), "Numero di immagini e maschere non corrisponde!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # [H, W] o [H, W, C]

        if image.ndim == 3:
            image = np.mean(image, axis=-1)

        # Normalizzazione immagine termica
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image * 255.0
        image = self.clahe.apply(image.astype(np.uint8))
        image = uint_to_float(image)  # float32 [0,1]
        image = reshape_array(image, self.height, self.width)
        image = torch.from_numpy(image)
        if image.ndim == 2:
            image = image.unsqueeze(0)

        if self.mask_files is not None:
            mask_path = self.mask_files[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {mask_path}")
            mask = normalize_mask(mask)
            mask = reshape_array(mask, self.height, self.width)

            if self.multiclass:
                labeled, num = label(mask > 0)
                if num == 2:
                    vals = np.unique(labeled)
                    for i, v in enumerate(vals[1:], start=1):
                        labeled[labeled == v] = i

                    labeled = np.clip(labeled, 0, 2)
                    mask = labeled
                else:
                    print(f"⚠️ Warning: {num} region(s) found in {mask_path}")
                mask = torch.from_numpy(mask).long()  # [H, W]
                mask = mask.squeeze(0)  # [1, H, W]
            else:
                mask = torch.from_numpy(mask).float()
        else:
            mask = torch.zeros_like(image)

        return image, mask
