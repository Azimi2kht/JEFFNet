from dataclasses import dataclass

import numpy as np
from PIL import Image
from torchvision import transforms


def to_uint8_numpy(img):
    """Convert PIL or numpy image to uint8 numpy array in RGB (if color)."""
    if isinstance(img, Image.Image):
        return np.array(img, dtype=np.uint8)
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    raise TypeError(f"Unsupported type: {type(img)}")


def to_pil_if_needed(original, arr_uint8):
    """Return PIL if input was PIL, otherwise return numpy."""
    if isinstance(original, Image.Image):
        return Image.fromarray(arr_uint8)
    return arr_uint8


class GaussianNoise:
    """Gaussian noise, mean=0, clip to [0,255]."""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img, thresh=5):
        img_np = to_uint8_numpy(img)
        mask = np.all(img_np < thresh, axis=2)
        
        noise = np.random.normal(0, self.sigma, img_np.shape).astype(np.float32)
        out = img_np.astype(np.float32) + noise
        out = np.clip(out, 0, 255).astype(np.uint8)
        
        out[mask] = 0 
        return to_pil_if_needed(img, out)
        
    def __repr__(self):
        return f"{self.__class__.__name__}(sigma={self.sigma})"


class PoissonNoise:
    """Poisson (shot) noise, clip to [0,255]."""

    def __init__(self, peak=30.0):
        """
        Args:
            peak (float): Controls Poisson noise strength.
        """
        self.peak = float(peak)

    def __call__(self, img):
        x = to_uint8_numpy(img).astype(np.float32) / 255.0  # [0,1]

        noisy_counts = np.random.poisson(x * self.peak).astype(np.float32)

        # back to value domain
        y = noisy_counts / self.peak

        out = np.clip(y * 255.0, 0, 255).astype(np.uint8)

        return to_pil_if_needed(img, out)

    def __repr__(self):
        return f"{self.__class__.__name__}(peak={self.peak})"


@dataclass(frozen=True)
class NoiseConfig:
    kind: str  # "gaussian" | "poisson" | "none"
    level: (
        int | float | None
    )  # sigma for gaussian_noise, peak for poisson, ignored if kind="none"


def build_eval_transform(cfg: NoiseConfig, image_size):
    aug = []
    if cfg.kind == "gaussian":
        aug.append(GaussianNoise(sigma=int(cfg.level)))
    elif cfg.kind == "poisson":
        aug.append(PoissonNoise(peak=int(cfg.level)))
    elif cfg.kind == "none":
        pass
    else:
        raise ValueError(f"Unknown kind: {cfg.kind}")

    return transforms.Compose(
        [
            transforms.Resize(image_size),
            *aug,
            transforms.ToTensor(),
        ]
    )
