"""RadImageNet frozen encoders (ResNet50 / DenseNet121) — CNNs pretrained on 1.35M
radiology images (Lab-Rasool/RadImageNet on HF). Used as extra frozen feature
extractors for the ablation. No CLAHE (hurts frozen encoders), ImageNet-style norm.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from huggingface_hub import hf_hub_download

from preprocessing import CLAHETransform

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_REPO = "Lab-Rasool/RadImageNet"


class RadResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        r = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(r.children())[:-2])   # conv1..layer4

    def forward(self, x):
        return self.backbone(x).mean(dim=[2, 3])                  # [B,2048]


class RadDenseNet121(nn.Module):
    def __init__(self):
        super().__init__()
        d = models.densenet121(weights=None)
        self.backbone = nn.Sequential(d.features)                 # backbone.0 = features

    def forward(self, x):
        return F.relu(self.backbone(x)).mean(dim=[2, 3])          # [B,1024]


_SPEC = {
    "radimagenet_resnet50":    ("ResNet50.pt", RadResNet50, 2048),
    "radimagenet_densenet121": ("DenseNet121.pt", RadDenseNet121, 1024),
}


class RadImageNetEmbedder:
    def __init__(self, name, use_clahe=False, size=224, device=_DEVICE):
        fn, cls, dim = _SPEC[name]
        model = cls()
        path = hf_hub_download(repo_id=HF_REPO, filename=fn)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [m for m in missing if m.startswith("backbone")]
        assert not bad, f"RadImageNet backbone keys missing: {bad[:5]}"
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        print(f"  [radimagenet] loaded {name} ({len(sd)} tensors, dim={dim})")

        self.name = name
        self.model = model.to(device).eval()
        self.device = device
        self.dim = dim
        tfm = transforms.Compose([
            transforms.Resize((size, size)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = transforms.Compose([CLAHETransform(), tfm]) if use_clahe else tfm

    @torch.no_grad()
    def embed_batch(self, pil_images):
        b = torch.stack([self.transform(im.convert("RGB")) for im in pil_images]).to(self.device)
        f = self.model(b).float().cpu().numpy()
        assert f.shape[1] == self.dim
        return f


def build(name):
    return RadImageNetEmbedder(name)


if __name__ == "__main__":
    import sys, glob, os
    from PIL import Image
    import config
    name = sys.argv[1] if len(sys.argv) > 1 else "radimagenet_resnet50"
    emb = build(name)
    fs = glob.glob(os.path.join(config.CBIS_PREPROC_DIR, "*.jpg"))[:3]
    v = emb.embed_batch([Image.open(f) for f in fs])
    print(f"{name}: dim={emb.dim} out={v.shape} norm={np.linalg.norm(v[0]):.3f} nan={bool(np.isnan(v).any())}")
