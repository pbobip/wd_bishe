from __future__ import annotations

import cv2
import numpy as np
from skimage import exposure, restoration

from backend.app.schemas.run import PreprocessConfig
from backend.app.utils.image_io import ensure_odd


class PreprocessService:
    def apply(self, image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
        if not config.enabled:
            return image.copy()

        processed = image.copy()
        processed = self.apply_background(processed, config)
        processed = self.apply_denoise(processed, config)
        processed = self.apply_enhance(processed, config)
        processed = self.apply_extras(processed, config)
        return processed

    def apply_background(self, image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
        method = config.background.method
        if method == "none":
            return image

        radius = ensure_odd(max(3, config.background.radius))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        if method == "tophat":
            corrected = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        else:
            background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            corrected = cv2.subtract(image, background)
        return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    def apply_denoise(self, image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
        method = config.denoise.method
        if method == "none":
            return image
        if method == "wavelet":
            return self.wavelet_denoise(image, config.denoise.wavelet_strength)
        if method == "mean":
            kernel = ensure_odd(max(1, config.denoise.mean_kernel))
            return cv2.blur(image, (kernel, kernel))
        if method == "gaussian":
            kernel = ensure_odd(max(1, config.denoise.gaussian_kernel))
            return cv2.GaussianBlur(image, (kernel, kernel), 0)
        if method == "median":
            kernel = ensure_odd(max(1, config.denoise.median_kernel))
            return cv2.medianBlur(image, kernel)
        diameter = ensure_odd(max(3, config.denoise.bilateral_diameter))
        return cv2.bilateralFilter(
            image,
            diameter,
            sigmaColor=max(1.0, float(config.denoise.bilateral_sigma_color)),
            sigmaSpace=max(1.0, float(config.denoise.bilateral_sigma_space)),
        )

    def apply_enhance(self, image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
        method = config.enhance.method
        if method == "none":
            return image
        if method == "clahe":
            tile = max(2, int(config.enhance.clahe_tile_size))
            clahe = cv2.createCLAHE(clipLimit=max(0.1, float(config.enhance.clahe_clip_limit)), tileGridSize=(tile, tile))
            return clahe.apply(image)
        if method == "hist_equalization":
            return cv2.equalizeHist(image)
        gamma = max(0.1, float(config.enhance.gamma))
        normalized = image.astype(np.float32) / 255.0
        corrected = np.power(normalized, gamma)
        return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

    def apply_extras(self, image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
        if not config.extras.unsharp:
            return image
        radius = ensure_odd(max(1, config.extras.unsharp_radius))
        blur = cv2.GaussianBlur(image, (radius, radius), 0)
        amount = max(0.0, float(config.extras.unsharp_amount))
        sharpened = cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def wavelet_denoise(self, image: np.ndarray, strength: float) -> np.ndarray:
        normalized = image.astype(np.float32) / 255.0
        denoised = restoration.denoise_wavelet(
            normalized,
            channel_axis=None,
            mode="soft",
            method="BayesShrink",
            rescale_sigma=True,
            sigma=max(0.02, strength),
        )
        denoised = exposure.rescale_intensity(denoised, out_range=(0, 255))
        return denoised.astype(np.uint8)


preprocess_service = PreprocessService()
