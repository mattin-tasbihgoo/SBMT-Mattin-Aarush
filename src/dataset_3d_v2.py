"""Patch-based 3D dataset V2 for brain tumor segmentation.

Changes from V1:
  - Foreground-biased sampling ratio configurable via fg_ratio (default 0.7).
  - Intensity augmentation: random Gaussian noise + brightness scaling
    in addition to the spatial flips from V1.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class BrainTumorDataset3D_V2(Dataset):
    def __init__(
        self,
        subject_ids,
        image_dir,
        mask_dir,
        patch_size=(128, 128, 128),
        patches_per_subject=4,
        mode="train",
        augment=False,
        fg_ratio=0.7,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.mode = mode
        self.augment = augment
        self.fg_ratio = fg_ratio

        if mode == "train":
            self.samples = [
                (sid, pi)
                for sid in subject_ids
                for pi in range(patches_per_subject)
            ]
        else:
            self.samples = [(sid, 0) for sid in subject_ids]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subject_id, _ = self.samples[idx]

        image = np.load(f"{self.image_dir}/{subject_id}.npy")
        mask = np.load(f"{self.mask_dir}/{subject_id}.npy")

        if self.mode == "train":
            image, mask = self._random_patch(image, mask)
            if self.augment:
                image, mask = self._augment(image, mask)
        else:
            image, mask = self._pad_to_patch_multiple(image, mask)

        return (
            torch.as_tensor(image.copy(), dtype=torch.float32),
            torch.as_tensor(mask[None].copy(), dtype=torch.float32),
            subject_id,
        )

    def _random_patch(self, image, mask):
        ph, pw, pd = self.patch_size
        _, H, W, D = image.shape

        if np.random.rand() < self.fg_ratio and mask.sum() > 0:
            fg = np.argwhere(mask > 0)
            center = fg[np.random.randint(len(fg))]
        else:
            center = np.array([
                np.random.randint(H),
                np.random.randint(W),
                np.random.randint(D),
            ])

        h0 = int(np.clip(center[0] - ph // 2, 0, max(H - ph, 0)))
        w0 = int(np.clip(center[1] - pw // 2, 0, max(W - pw, 0)))
        d0 = int(np.clip(center[2] - pd // 2, 0, max(D - pd, 0)))

        img_patch = image[:, h0:h0 + ph, w0:w0 + pw, d0:d0 + pd]
        msk_patch = mask[h0:h0 + ph, w0:w0 + pw, d0:d0 + pd]

        img_patch = self._maybe_pad(img_patch, self.patch_size, is_image=True)
        msk_patch = self._maybe_pad(msk_patch, self.patch_size, is_image=False)

        return img_patch, msk_patch

    @staticmethod
    def _maybe_pad(arr, target_shape, is_image=True):
        if is_image:
            spatial = arr.shape[1:]
        else:
            spatial = arr.shape
        pads = [(0, max(t - s, 0)) for s, t in zip(spatial, target_shape)]
        if all(p == (0, 0) for p in pads):
            return arr
        if is_image:
            pads = [(0, 0)] + pads
        return np.pad(arr, pads, mode="constant", constant_values=0)

    def _pad_to_patch_multiple(self, image, mask):
        _, H, W, D = image.shape
        ph, pw, pd = self.patch_size

        def _ceil(x, m):
            return x if x % m == 0 else x + (m - x % m)

        nH, nW, nD = _ceil(H, ph), _ceil(W, pw), _ceil(D, pd)

        if (nH, nW, nD) != (H, W, D):
            image = np.pad(
                image,
                [(0, 0), (0, nH - H), (0, nW - W), (0, nD - D)],
                mode="constant",
            )
            mask = np.pad(
                mask,
                [(0, nH - H), (0, nW - W), (0, nD - D)],
                mode="constant",
            )
        return image, mask

    def _augment(self, image, mask):
        image = image.copy()
        mask = mask.copy()

        for axis_img, axis_msk in [(1, 0), (2, 1), (3, 2)]:
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=axis_img).copy()
                mask = np.flip(mask, axis=axis_msk).copy()

        if np.random.rand() < 0.3:
            sigma = np.random.uniform(0.0, 0.05)
            image = image + np.random.normal(0, sigma, size=image.shape).astype(np.float32)

        if np.random.rand() < 0.3:
            for c in range(image.shape[0]):
                image[c] = image[c] * np.random.uniform(0.85, 1.15)

        np.clip(image, 0.0, 1.0, out=image)

        return image, mask
