import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


FOV_THRESH = 10


def fov_mask(img: Image.Image, thresh: int = FOV_THRESH) -> np.ndarray:
    """Boolean mask of the (non-black) fundus field of view."""
    return np.asarray(img.convert("L")) > thresh


def crop_black_borders(img: Image.Image, thresh: int = FOV_THRESH) -> Image.Image:
    """Crop the black background around the circular fundus field of view.
    Fundus cameras differ in how much border they leave, so this removes one easy source of shift."""
    mask = fov_mask(img, thresh)
    if mask.sum() < 0.05 * mask.size:  # nearly black image: leave as is
        return img
    rows, cols = np.where(mask)
    return img.crop((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1))


def pad_to_square(img: Image.Image) -> Image.Image:
    """Pad with black to a square, centred. Many fundus images have the circular field cut off at the
    top and bottom; resizing those straight to a square would stretch the retina."""
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    out = Image.new(img.mode, (side, side))
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out


def make_transforms(size: int, train: bool):
    if train:
        return T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(), T.Normalize(MEAN, STD),
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


class FundusDataset(Dataset):
    def __init__(self, csv_path: str, split: str | None, size: int, train: bool):
        df = pd.read_csv(csv_path)
        if split is not None:
            df = df[df["split"] == split]
        self.df = df.reset_index(drop=True)
        self.tf = make_transforms(size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        # cheap no-op on images already cached by preprocess.py
        img = pad_to_square(crop_black_borders(Image.open(row["image_path"]).convert("RGB")))
        return self.tf(img), torch.tensor(float(row["label"]))
