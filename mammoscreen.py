"""mammoscreen (ianpan/mammoscreen) as a zero-shot cancer predictor / feature source.

Purpose-built breast-cancer model: 3x EfficientNetV2-S ensemble, takes {cc, mlo}
uint8 grayscale arrays and outputs a cancer score (averaging the two views). We
load it manually (AutoModel hits a transformers-version bug) and evaluate its
zero-shot cancer score on CBIS, then can late-fuse it with our multi-view head.
"""
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_M = None


def load():
    global _M
    if _M is None:
        cfg = AutoConfig.from_pretrained("ianpan/mammoscreen", trust_remote_code=True)
        ME = get_class_from_dynamic_module("modeling.MammoEnsemble", "ianpan/mammoscreen")
        m = ME(cfg)
        m.load_state_dict(load_file(hf_hub_download("ianpan/mammoscreen", "model.safetensors")),
                          strict=False)
        _M = m.to(_DEV).eval()
    return _M


@torch.no_grad()
def cancer(cc_path, mlo_path=None, crop=False):
    m = load()

    def _arr(p):
        a = np.array(Image.open(p).convert("L"))
        if crop:
            import mammo_crop
            a = mammo_crop.crop_array(a)
        return a

    sample = {"cc": _arr(cc_path)}
    if mlo_path is not None:
        sample["mlo"] = _arr(mlo_path)
    return float(m(sample, device=_DEV)["cancer"])


@torch.no_grad()
def features_single(a):
    """a: HxW uint8 grayscale -> concatenated pooled features from the 3 backbones."""
    m = load()
    parts = []
    for i in range(m.num_models):
        net = getattr(m, f"net{i}")
        pp = net.preprocess([{"img": a}], device=_DEV)      # [{"img": tensor[1,H,W]}]
        t = net.normalize(pp[0]["img"].unsqueeze(0))         # [1,1,H,W]
        f = net.pooling(net.backbone(t)).reshape(1, -1)      # [1,feature_dim]
        parts.append(f.cpu().numpy()[0])
    return np.concatenate(parts).astype(np.float32)


class MammoscreenFeatEmbedder:
    """Interface-compatible encoder using mammoscreen's internal features."""
    def __init__(self):
        self.name = "mammoscreen_feat"
        load()
        self.device = _DEV
        self.dim = len(features_single(np.full((512, 512), 128, np.uint8)))

    @torch.no_grad()
    def embed_batch(self, pil_images):
        return np.stack([features_single(np.array(im.convert("L"))) for im in pil_images])


def case_scores(dataset="cbis_full", split="test", paired_only=True, crop=False):
    """Return (case_ids, labels, scores, is_paired) for CBIS cases."""
    import db
    conn = db.connect()
    rows = conn.execute(
        f"SELECT case_id, MAX(binary_label) y, GROUP_CONCAT(view) vs FROM samples "
        f"WHERE split=? AND dataset=? GROUP BY case_id", (split, dataset)).fetchall()
    cids, yy, sc, pr = [], [], [], []
    for r in rows:
        views = set(r["vs"].split(","))
        is_paired = ("CC" in views and "MLO" in views)
        if paired_only and not is_paired:
            continue
        paths = {x["view"]: x["file_path"] for x in conn.execute(
            "SELECT view,file_path FROM samples WHERE case_id=? AND split=?", (r["case_id"], split))}
        cc = paths.get("CC"); mlo = paths.get("MLO")
        if cc is None:
            cc, mlo = mlo, None
        cids.append(r["case_id"]); yy.append(r["y"]); sc.append(cancer(cc, mlo, crop=crop)); pr.append(is_paired)
    return np.array(cids), np.array(yy), np.array(sc), np.array(pr)


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    _, y, s, _ = case_scores(paired_only=True)
    print(f"mammoscreen ZERO-SHOT on CBIS paired-{len(y)}: AUC={roc_auc_score(y, s):.4f} "
          f"(score range {s.min():.4f}-{s.max():.4f})")
