"""ianpan/mammo-crop — breast-region cropper (bbox regressor). mammoscreen's docs
recommend cropping before inference; we feed it the CBIS full jpegs, so this may
raise mammoscreen's zero-shot contribution. Loaded manually (AutoModel bug)."""
import albumentations as A
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_M, _IC = None, 1


def load():
    global _M, _IC
    if _M is None:
        cfg = AutoConfig.from_pretrained("ianpan/mammo-crop", trust_remote_code=True)
        _IC = int(getattr(cfg, "in_chans", 1))
        cls = get_class_from_dynamic_module(cfg.auto_map["AutoModel"], "ianpan/mammo-crop")
        m = cls(cfg)
        m.load_state_dict(load_file(hf_hub_download("ianpan/mammo-crop", "model.safetensors")),
                          strict=False)
        _M = m.to(_DEV).eval()
    return _M


@torch.no_grad()
def crop_array(img):
    """img: HxW uint8 grayscale -> cropped HxW uint8 (breast region)."""
    m = load()
    h, w = img.shape[:2]
    x = A.Resize(256, 256, p=1)(image=img)["image"]
    t = torch.from_numpy(x).float()
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.shape[0] != _IC:
        t = t.repeat(_IC, 1, 1)
    t = t.unsqueeze(0).to(_DEV)                       # [1, IC, 256, 256]
    box = m(t, img_shape=torch.tensor([[h, w]], device=_DEV))[0].detach().cpu().numpy()
    x0, y0, bw, bh = [int(round(float(v))) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    bw, bh = max(1, bw), max(1, bh)
    c = img[y0:y0 + bh, x0:x0 + bw]
    return c if c.size > 0 else img


if __name__ == "__main__":
    import glob, os, config
    from PIL import Image
    f = glob.glob(os.path.join(config.DATA_ROOT, "cbis-ddsm", "jpeg", "*", "*.jpg"))[0]
    a = np.array(Image.open(f).convert("L"))
    c = crop_array(a)
    print(f"crop: {a.shape} -> {c.shape}")
