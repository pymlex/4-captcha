import cv2
import numpy as np
from albumentations import (
    Compose,
    ElasticTransform,
    GaussianBlur,
    RandomBrightnessContrast,
)


def draw_bezier_curves(
    image: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw 2-4 dark Bézier curves on a grayscale image."""
    canvas = image.copy()
    n_curves = int(rng.integers(2, 5))
    h, w = canvas.shape
    for _ in range(n_curves):
        pts = rng.integers(0, [w, h], size=(4, 2)).astype(np.int32)
        thickness = int(rng.integers(1, 4))
        color = int(rng.integers(60, 140))
        t_vals = np.linspace(0.0, 1.0, 80)
        curve = np.zeros((len(t_vals), 2), dtype=np.float64)
        for i, t in enumerate(t_vals):
            p0, p1, p2, p3 = pts
            curve[i] = (
                (1 - t) ** 3 * p0
                + 3 * (1 - t) ** 2 * t * p1
                + 3 * (1 - t) * t ** 2 * p2
                + t ** 3 * p3
            )
        curve = curve.astype(np.int32)
        for j in range(len(curve) - 1):
            cv2.line(
                canvas,
                tuple(curve[j]),
                tuple(curve[j + 1]),
                int(color),
                thickness,
            )
    return canvas


def apply_gaussian_noise(
    image: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sigma = float(rng.uniform(5.0, 20.0))
    noise = rng.normal(0.0, sigma, size=image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def global_augment(
    image: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply elastic deformation, blur, brightness changes, and Gaussian noise."""
    kernel = int(rng.choice([3, 5]))
    transform = Compose(
        [
            ElasticTransform(
                alpha=float(rng.uniform(40.0, 90.0)),
                sigma=float(rng.uniform(5.0, 8.0)),
                p=1.0,
            ),
            GaussianBlur(blur_limit=(kernel, kernel), p=1.0),
            RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=1.0,
            ),
        ],
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    augmented = transform(image=image)["image"]
    with_curves = draw_bezier_curves(augmented, rng)
    gamma = float(rng.uniform(0.85, 1.15))
    adjusted = np.clip(
        255.0 * np.power(with_curves.astype(np.float64) / 255.0, gamma),
        0,
        255,
    ).astype(np.uint8)
    return apply_gaussian_noise(adjusted, rng)
