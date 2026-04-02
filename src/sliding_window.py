"""Sliding-window inference for 3D U-Net.

Accumulates sigmoid probabilities with an overlap count map so
boundary artifacts are averaged away.
"""

import numpy as np
import torch
from torch.amp import autocast


def sliding_window_inference(
    model,
    image: np.ndarray,
    patch_size=(128, 128, 128),
    overlap=0.5,
    device="cuda",
    amp_enabled=True,
):
    """Run overlapping patch inference on a full volume.

    Parameters
    ----------
    model : nn.Module
        Trained 3D U-Net (eval mode expected).
    image : np.ndarray
        Shape ``(C, H, W, D)`` — same layout as saved .npy.
    patch_size : tuple
        Spatial patch dimensions ``(pH, pW, pD)``.
    overlap : float
        Fraction of patch overlap (0.5 = 50 %).
    device : str
        CUDA or CPU.
    amp_enabled : bool
        Use mixed precision during inference.

    Returns
    -------
    np.ndarray
        Probability map, shape ``(H, W, D)`` in ``[0, 1]``.
    """
    model.eval()
    C, H, W, D = image.shape
    ph, pw, pd = patch_size
    stride = [max(1, int(p * (1 - overlap))) for p in patch_size]

    def _starts(length, patch, step):
        starts = list(range(0, max(length - patch + 1, 1), step))
        if starts[-1] + patch < length:
            starts.append(length - patch)
        return starts

    h_starts = _starts(H, ph, stride[0])
    w_starts = _starts(W, pw, stride[1])
    d_starts = _starts(D, pd, stride[2])

    pred_sum = np.zeros((H, W, D), dtype=np.float32)
    count_map = np.zeros((H, W, D), dtype=np.float32)

    with torch.no_grad():
        for h0 in h_starts:
            for w0 in w_starts:
                for d0 in d_starts:
                    patch = image[:, h0:h0 + ph, w0:w0 + pw, d0:d0 + pd]

                    pad_h = ph - patch.shape[1]
                    pad_w = pw - patch.shape[2]
                    pad_d = pd - patch.shape[3]
                    if pad_h > 0 or pad_w > 0 or pad_d > 0:
                        patch = np.pad(
                            patch,
                            [(0, 0), (0, pad_h), (0, pad_w), (0, pad_d)],
                            mode="constant",
                        )

                    t = torch.as_tensor(patch[None], dtype=torch.float32, device=device)
                    with autocast(device_type=device if device != "cpu" else "cpu", enabled=amp_enabled):
                        logits = model(t)
                    prob = torch.sigmoid(logits).squeeze().cpu().numpy()

                    if pad_h > 0 or pad_w > 0 or pad_d > 0:
                        prob = prob[:ph - pad_h or None, :pw - pad_w or None, :pd - pad_d or None]

                    pred_sum[h0:h0 + prob.shape[0], w0:w0 + prob.shape[1], d0:d0 + prob.shape[2]] += prob
                    count_map[h0:h0 + prob.shape[0], w0:w0 + prob.shape[1], d0:d0 + prob.shape[2]] += 1.0

    return pred_sum / np.maximum(count_map, 1e-8)
