"""Mammogram preprocessing reused from the prior MasterNotebooks project.

The 512x512 grayscale JPEGs on disk are already YOLO breast-cropped. The single
biggest booster in the prior work (+3% accuracy) was CLAHE, which was NOT baked
into the saved JPEGs. We add it here deterministically at embed time:

    cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6))  on the 8-bit grayscale,
    then replicate to a 3-channel image for the (ImageNet-pretrained) encoders.

This is a PIL->PIL transform, prepended to every encoder's transform, so all
encoders receive identical, CLAHE-enhanced input. It is deterministic (no random
augmentation) because the pipeline is training-free.
"""
import cv2
import numpy as np
from PIL import Image

import config


class CLAHETransform:
    """Apply CLAHE on the 8-bit grayscale image, return a 3-channel RGB PIL image."""

    def __init__(self, clip=None, tile=None):
        self.clip = config.CLAHE_CLIP if clip is None else clip
        self.tile = config.CLAHE_TILE if tile is None else tile
        # A fresh CLAHE object per instance; created lazily to stay picklable.
        self._clahe = None

    def _op(self):
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=self.clip,
                                          tileGridSize=tuple(self.tile))
        return self._clahe

    def __call__(self, img):
        gray = np.asarray(img.convert("L"), dtype=np.uint8)   # 8-bit grayscale
        eq = self._op().apply(gray)
        rgb = np.repeat(eq[:, :, None], 3, axis=2)             # replicate to 3ch
        return Image.fromarray(rgb, mode="RGB")


def load_image(path):
    """Open an image as PIL (grayscale on disk). Returns None on failure."""
    try:
        return Image.open(path)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not open {path}: {e}")
        return None


if __name__ == "__main__":
    import glob, os
    fs = glob.glob(os.path.join(config.CBIS_PREPROC_DIR, "*.jpg"))
    t = CLAHETransform()
    out = t(Image.open(fs[0]))
    print(f"CLAHE ok: in={Image.open(fs[0]).mode} out={out.mode} size={out.size} "
          f"clip={t.clip} tile={t.tile}")
