"""Mammo-CLIP (MICCAI 2024) image encoder as a frozen embedder.

Domain-specific: an EfficientNet-B5 trained with CLIP on screening
mammogram-report pairs (batmanlab/Mammo-CLIP). We load ONLY the pretrained
vision encoder and extract its 2048-d pooled features. NO training / fine-tuning.

The released checkpoint's image encoder is a lukemelas `efficientnet_pytorch`
EfficientNet-B5 (config name 'tf_efficientnet_b5_ns-detect'), 3-channel input,
with the '_fc' head dropped. Keys match `EfficientNet.from_name('efficientnet-b5')`
exactly (verified). Mammo-CLIP expects grayscale-replicated 3-channel input at
1520x912 with mean=0.3089279, std=0.25053555408335154 — its own pipeline,
distinct from the RGB+CLAHE path used by the other encoders.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAMMO_MEAN = 0.3089279
MAMMO_STD = 0.25053555408335154
HF_REPO = "shawn24/Mammo-CLIP"      # public checkpoints
CKPTS = {
    "mammo_clip_b5": ("Pre-trained-checkpoints/b5-model-best-epoch-7.tar", "efficientnet-b5"),
    "mammo_clip_b2": ("Pre-trained-checkpoints/b2-model-best-epoch-10.tar", "efficientnet-b2"),
}


class MammoClipEmbedder:
    """Interface-compatible with embedders.Embedder (.name/.dim/.embed_batch)."""

    def __init__(self, name, ckpt_path, arch, size=(1520, 912), device=_DEVICE,
                 max_bs=8, use_clahe=False):
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_name(arch)
        # Trusted official checkpoint stores an omegaconf config -> weights_only=False.
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw = ck["model"] if "model" in ck else ck
        weights = {k[len("image_encoder."):]: v for k, v in raw.items()
                   if k.startswith("image_encoder.")}
        missing, unexpected = model.load_state_dict(weights, strict=False)
        backbone_missing = [m for m in missing if not m.startswith("_fc")]
        assert not backbone_missing, f"Mammo-CLIP backbone keys missing: {backbone_missing[:5]}"
        assert not unexpected, f"unexpected Mammo-CLIP keys: {unexpected[:5]}"
        print(f"  [mammo_clip] loaded {len(weights)} tensors into {arch} "
              f"(dropped unused _fc head)")

        self.name = name
        self.model = model.to(device).eval()
        self.device = device
        self.dim = model._conv_head.out_channels   # 2048 for b5
        self.size = size                            # (H, W)
        self.max_bs = max_bs
        self.use_clahe = use_clahe
        if use_clahe:
            from preprocessing import CLAHETransform
            self._clahe = CLAHETransform()

    def _to_tensor(self, im):
        g = im.convert("L")
        if self.use_clahe:
            g = self._clahe(g).convert("L")
        g = g.resize((self.size[1], self.size[0]), Image.BILINEAR)   # PIL wants (W,H)
        arr = np.asarray(g, dtype=np.float32) / 255.0
        arr = (arr - MAMMO_MEAN) / MAMMO_STD
        return torch.from_numpy(arr)[None].repeat(3, 1, 1)           # 3-channel

    @torch.no_grad()
    def embed_batch(self, pil_images):
        feats = []
        for i in range(0, len(pil_images), self.max_bs):
            chunk = pil_images[i:i + self.max_bs]
            batch = torch.stack([self._to_tensor(im) for im in chunk]).to(self.device)
            fmap = self.model.extract_features(batch)                # [B,2048,h,w]
            pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1)       # [B,2048]
            feats.append(pooled.float().cpu().numpy())
        feats = np.concatenate(feats, axis=0)
        assert feats.shape[1] == self.dim
        return feats


def download_ckpt(name):
    from huggingface_hub import hf_hub_download
    rel, _arch = CKPTS[name]
    return hf_hub_download(repo_id=HF_REPO, filename=rel)


def build(name):
    if name not in CKPTS:
        raise KeyError(f"unknown mammo-clip variant {name!r}")
    rel, arch = CKPTS[name]
    return MammoClipEmbedder(name, download_ckpt(name), arch)


if __name__ == "__main__":
    import sys, glob
    name = sys.argv[1] if len(sys.argv) > 1 else "mammo_clip_b5"
    emb = build(name)
    fs = glob.glob(os.path.join(config.CBIS_PREPROC_DIR, "*.jpg"))[:3]
    vecs = emb.embed_batch([Image.open(f) for f in fs])
    print(f"{emb.name}: dim={emb.dim} out={vecs.shape} "
          f"norm={np.linalg.norm(vecs[0]):.3f} nan={bool(np.isnan(vecs).any())}")
