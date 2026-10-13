"""Frozen-encoder registry. Each embedder loads a PRETRAINED model, runs in
eval/no_grad, and returns a pooled/CLS embedding. NO training, NO fine-tuning,
NO classifier head — ever.

Every encoder receives identical input: CLAHE-enhanced (clip=4.0, tile=6x6),
grayscale-replicated-to-RGB, square-resized to the model's native size, then the
model's own normalization. Square resize (not center-crop) preserves the whole
YOLO-cropped breast, matching the prior MasterNotebooks pipeline.

Add an encoder by registering one zero-arg builder. Output dim is auto-detected.
"""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from preprocessing import CLAHETransform
import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(self, name, model, transform, dim=None, forward_fn=None,
                 device=_DEVICE, use_clahe=None):
        self.name = name
        self.model = model.to(device).eval()
        use_clahe = config.USE_CLAHE if use_clahe is None else use_clahe
        # Prepend CLAHE (PIL->PIL) so all encoders see identical enhanced input.
        self.transform = transforms.Compose([CLAHETransform(), transform]) \
            if use_clahe else transform
        self.device = device
        self._forward_fn = forward_fn or (lambda m, x: m(x))
        self.dim = dim if dim is not None else self._infer_dim()

    @torch.no_grad()
    def _infer_dim(self):
        dummy = Image.new("RGB", (256, 256), (128, 128, 128))
        feats = self._run(torch.stack([self.transform(dummy)]).to(self.device))
        return int(feats.shape[1])

    def _run(self, batch):
        feats = self._forward_fn(self.model, batch)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats.float()

    @torch.no_grad()
    def embed_batch(self, pil_images):
        batch = torch.stack([self.transform(im.convert("RGB")) for im in pil_images])
        feats = self._run(batch.to(self.device)).cpu().numpy()
        assert feats.shape[1] == self.dim, \
            f"{self.name}: expected dim {self.dim}, got {feats.shape[1]}"
        return feats


def _square_transform(size, mean, std):
    """CLAHE is prepended in Embedder; here: square resize -> tensor -> normalize."""
    return transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# --- DINOv2 (torch.hub) ------------------------------------------------------

def _dinov2(name, variant, size=224, use_clahe=None):
    model = torch.hub.load("facebookresearch/dinov2", variant)
    return Embedder(name, model, _square_transform(size, IMAGENET_MEAN, IMAGENET_STD),
                    use_clahe=use_clahe)


# --- timm models -------------------------------------------------------------

def _build_timm(name, model_id, use_clahe=None):
    import timm
    from timm.data import resolve_data_config
    model = timm.create_model(model_id, pretrained=True, num_classes=0)
    cfg = resolve_data_config({}, model=model)
    size = cfg["input_size"][-1]
    transform = _square_transform(size, cfg["mean"], cfg["std"])
    return Embedder(name, model, transform, use_clahe=use_clahe)


# --- Hugging Face ViT encoders ----------------------------------------------

class _HFCLSWrapper(nn.Module):
    """HF ViT -> pooling of last_hidden_state.
    pooling: 'cls' (token 0) | 'mean' (patch tokens) | 'clsmean' (concat)."""
    def __init__(self, model, pooling="cls"):
        super().__init__()
        self.model = model
        self.pooling = pooling

    def forward(self, pixel_values):
        h = self.model(pixel_values=pixel_values).last_hidden_state  # [B,1+P,D]
        cls = h[:, 0]
        if self.pooling == "cls":
            return cls
        mean = h[:, 1:].mean(dim=1)
        if self.pooling == "mean":
            return mean
        return torch.cat([cls, mean], dim=1)


def _extract_size(obj, default=224):
    if obj is None:
        return default
    for attr in ("height", "shortest_edge"):
        v = getattr(obj, attr, None)
        if v:
            return int(v)
    if isinstance(obj, dict):
        for key in ("height", "shortest_edge"):
            if obj.get(key):
                return int(obj[key])
    try:
        return int(obj)
    except (TypeError, ValueError):
        return default


def _build_hf_vit(name, model_id, pooling="cls", trust_remote_code=False, use_clahe=None):
    from transformers import AutoModel, AutoImageProcessor
    try:
        proc = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        size = _extract_size(getattr(proc, "crop_size", None)) if getattr(proc, "crop_size", None) \
            else _extract_size(proc.size)
        mean = getattr(proc, "image_mean", IMAGENET_MEAN)
        std = getattr(proc, "image_std", IMAGENET_STD)
        transform = _square_transform(size, mean, std)
    except Exception:  # noqa: BLE001
        transform = _square_transform(224, IMAGENET_MEAN, IMAGENET_STD)
    model = _HFCLSWrapper(AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code),
                          pooling=pooling)
    return Embedder(name, model, transform, use_clahe=use_clahe)


# name -> zero-arg builder (lazy: only the requested encoder is loaded)
REGISTRY = {
    # general self-supervised (strong retrieval features)
    "dinov2_vits14": lambda: _dinov2("dinov2_vits14", "dinov2_vits14"),
    "dinov2_vitb14": lambda: _dinov2("dinov2_vitb14", "dinov2_vitb14"),
    "dinov2_vitl14": lambda: _dinov2("dinov2_vitl14", "dinov2_vitl14"),
    # diagnostics: higher input resolution (fine mammographic cues) + CLAHE ablation
    "dinov2_vitb14_r518":   lambda: _dinov2("dinov2_vitb14_r518", "dinov2_vitb14", size=518),
    "dinov2_vitl14_r518":   lambda: _dinov2("dinov2_vitl14_r518", "dinov2_vitl14", size=518),
    "dinov2_vitb14_noclahe": lambda: _dinov2("dinov2_vitb14_noclahe", "dinov2_vitb14", use_clahe=False),
    "dinov2_vitl14_noclahe": lambda: _dinov2("dinov2_vitl14_noclahe", "dinov2_vitl14", use_clahe=False),
    "dinov2_vitl14_r518_noclahe": lambda: _dinov2("dinov2_vitl14_r518_noclahe", "dinov2_vitl14", size=518, use_clahe=False),
    "rad_dino_noclahe":       lambda: _build_hf_vit("rad_dino_noclahe", "microsoft/rad-dino", pooling="cls", use_clahe=False),
    "convnext_base_noclahe":  lambda: _build_timm("convnext_base_noclahe", "convnext_base.fb_in22k_ft_in1k", use_clahe=False),
    "efficientnet_b5_noclahe": lambda: _build_timm("efficientnet_b5_noclahe", "tf_efficientnet_b5.ns_jft_in1k", use_clahe=False),
    "densenet121_noclahe":    lambda: _build_timm("densenet121_noclahe", "densenet121.ra_in1k", use_clahe=False),
    # ImageNet-pretrained baselines (frozen feature extractors)
    "resnet50":        lambda: _build_timm("resnet50", "resnet50.a1_in1k"),
    "densenet121":     lambda: _build_timm("densenet121", "densenet121.ra_in1k"),
    "efficientnet_b5": lambda: _build_timm("efficientnet_b5", "tf_efficientnet_b5.ns_jft_in1k"),
    "convnext_base":   lambda: _build_timm("convnext_base", "convnext_base.fb_in22k_ft_in1k"),
    "convnextv2_base": lambda: _build_timm("convnextv2_base", "convnextv2_base.fcmae_ft_in22k_in1k"),
    "vit_base":        lambda: _build_timm("vit_base", "vit_base_patch16_224.augreg2_in21k_ft_in1k"),
    "davit_base":      lambda: _build_timm("davit_base", "davit_base.msft_in1k"),
    # radiology / medical foundation encoder
    "rad_dino":        lambda: _build_hf_vit("rad_dino", "microsoft/rad-dino", pooling="cls"),
    # mammography-specific foundation model (bespoke grayscale pipeline in mammo_clip.py)
    "mammo_clip_b5":   lambda: __import__("mammo_clip").build("mammo_clip_b5"),
    "mammo_clip_b2":   lambda: __import__("mammo_clip").build("mammo_clip_b2"),
    # radiology CNNs pretrained on RadImageNet (1.35M radiology images)
    "radimagenet_resnet50":    lambda: __import__("radimagenet").build("radimagenet_resnet50"),
    "radimagenet_densenet121": lambda: __import__("radimagenet").build("radimagenet_densenet121"),
    # mammoscreen internal features (3x EfficientNetV2-S pooled) as an encoder
    "mammoscreen_feat":        lambda: __import__("mammoscreen").MammoscreenFeatEmbedder(),
}

# Encoders that load without any HF gating/token, safe to run out of the box.
UNGATED = ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "resnet50",
           "densenet121", "efficientnet_b5", "convnext_base", "convnextv2_base",
           "vit_base", "davit_base", "rad_dino"]


def load_embedder(name):
    if name not in REGISTRY:
        raise KeyError(f"unknown embedder {name!r}; available: {list(REGISTRY)}")
    return REGISTRY[name]()


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
    emb = load_embedder(name)
    dummy = Image.new("L", (512, 512), 128)
    vecs = emb.embed_batch([dummy, dummy])
    print(f"{emb.name}: dim={emb.dim} out={vecs.shape} "
          f"norm={np.linalg.norm(vecs[0]):.3f} nan={bool(np.isnan(vecs).any())}")
