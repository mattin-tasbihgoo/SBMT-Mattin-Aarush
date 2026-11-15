import cv2
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

FLAIR_CH = 0
T1GD_CH = 2


def normalize_volume(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    brain = img[img > 0]
    if brain.size == 0:
        return img.astype(np.float32, copy=False)

    lo, hi = np.percentile(brain, [1, 99])
    img = np.clip(img, lo, hi)

    brain = img[img > 0]
    if brain.size == 0:
        return img.astype(np.float32, copy=False)

    img = (img - brain.mean()) / (brain.std() + 1e-8)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return img.astype(np.float32, copy=False)


def resize_image(x, size=(256, 256)):
    x = np.ascontiguousarray(x, dtype=np.float32)
    return cv2.resize(x, size, interpolation=cv2.INTER_LINEAR)


def resize_mask(x, size=(256, 256)):
    x = np.ascontiguousarray(x, dtype=np.uint8)
    return cv2.resize(x, size, interpolation=cv2.INTER_NEAREST)


class BrainTumorSliceDataset(Dataset):
    def __init__(self, df, augment=False, out_size=(256, 256), preload_subjects=True):
        self.df = df.reset_index(drop=True).copy()
        self.augment = augment
        self.out_size = tuple(out_size)
        self.offsets = (-2, -1, 0, 1, 2)
        self.preload_subjects = preload_subjects

        self.subject_data = {}

        if self.preload_subjects:
            self._preload_all_subjects()

    def __len__(self):
        return len(self.df)

    def _flip_lr(self, img, msk):
        img = np.flip(img, axis=2).copy()   # [C, H, W]
        msk = np.flip(msk, axis=1).copy()   # [H, W]
        return img, msk

    def _flip_ud(self, img, msk):
        img = np.flip(img, axis=1).copy()   # [C, H, W]
        msk = np.flip(msk, axis=0).copy()   # [H, W]
        return img, msk

    def _clamp_slice_idx(self, z, depth):
        return max(0, min(int(z), depth - 1))

    def _load_subject(self, image_path, label_path):
        img4d = nib.load(image_path).get_fdata().astype(np.float32)
        lbl = nib.load(label_path).get_fdata().astype(np.float32)

        flair = normalize_volume(img4d[:, :, :, FLAIR_CH])
        t1gd = normalize_volume(img4d[:, :, :, T1GD_CH])
        lbl = (lbl > 0).astype(np.uint8)

        # Convert from [H, W, D] -> [D, H, W]
        # This makes individual axial slices contiguous in memory.
        flair = np.ascontiguousarray(np.transpose(flair, (2, 0, 1)), dtype=np.float32)
        t1gd = np.ascontiguousarray(np.transpose(t1gd, (2, 0, 1)), dtype=np.float32)
        lbl = np.ascontiguousarray(np.transpose(lbl, (2, 0, 1)), dtype=np.uint8)

        return {
            "flair": flair,   # [D, H, W]
            "t1gd": t1gd,     # [D, H, W]
            "label": lbl,     # [D, H, W]
        }

    def _preload_all_subjects(self):
        unique_subjects = (
            self.df[["subject_id", "image_path", "label_path"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        for row in tqdm(
            unique_subjects.itertuples(index=False),
            total=len(unique_subjects),
            desc="Preloading subjects into RAM",
        ):
            self.subject_data[row.subject_id] = self._load_subject(
                row.image_path,
                row.label_path,
            )

    def _get_subject_data(self, subject_id, image_path, label_path):
        if subject_id in self.subject_data:
            return self.subject_data[subject_id]

        subject = self._load_subject(image_path, label_path)
        self.subject_data[subject_id] = subject
        return subject

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        subject_id = row.subject_id
        z = int(row.slice_idx)
        image_path = row.image_path
        label_path = row.label_path

        try:
            subject = self._get_subject_data(subject_id, image_path, label_path)

            flair = subject["flair"]   # [D, H, W]
            t1gd = subject["t1gd"]     # [D, H, W]
            lbl = subject["label"]     # [D, H, W]

            depth = flair.shape[0]
            center_z = self._clamp_slice_idx(z, depth)
            context_indices = [self._clamp_slice_idx(center_z + dz, depth) for dz in self.offsets]

            flair_slices = [
                resize_image(flair[zi], self.out_size)
                for zi in context_indices
            ]
            t1gd_slices = [
                resize_image(t1gd[zi], self.out_size)
                for zi in context_indices
            ]
            msk = resize_mask(lbl[center_z], self.out_size).astype(np.float32)

            img = np.stack(flair_slices + t1gd_slices, axis=0).astype(np.float32)  # [10, H, W]

        except Exception as e:
            raise RuntimeError(
                f"Failed loading runtime sample idx={idx}, "
                f"subject={subject_id}, image={image_path}, label={label_path}, z={z}"
            ) from e

        if self.augment:
            if np.random.rand() < 0.5:
                img, msk = self._flip_lr(img, msk)

            if np.random.rand() < 0.5:
                img, msk = self._flip_ud(img, msk)

        img = torch.from_numpy(img)
        msk = torch.from_numpy(msk).unsqueeze(0)
        return img, msk