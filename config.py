"""Central configuration for the training-free MamoRAC pipeline.

MamoRAC = Retrieval-Augmented Classification for mammography. It ports the
training-free RAC-on-BreakHis design (frozen encoder -> cached embeddings ->
FAISS kNN over train-only -> parameter-free weighted/tip vote) to CBIS-DDSM,
reusing the mammogram preprocessing (YOLO breast crop + CLAHE) from the prior
MasterNotebooks project. NOTHING is trained or fine-tuned.

Single source of truth for paths, the experiment grid, and reproducibility.
"""
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
DB_PATH = os.path.join(PROJECT_DIR, "mamorac.db")

# --- Datasets (read-only) ----------------------------------------------------
# Preprocessed 512x512 grayscale JPEGs already exist on disk (YOLO breast-crop
# applied; CLAHE is NOT baked in and is added at embed time — see preprocessing).
DATA_ROOT = r"F:\Development\MasterNotebooks"

# CBIS-DDSM mass, official train/test split (patient-disjoint, verified 0 overlap)
CBIS_PREPROC_DIR = os.path.join(DATA_ROOT, "preproc_jpg")
CBIS_TRAIN_CSV = os.path.join(DATA_ROOT, "train_split_preproc.csv")   # 1231 imgs (gallery)
CBIS_TEST_CSV = os.path.join(DATA_ROOT, "val_split_preproc.csv")      # 361 imgs (query)

# InBreast (optional): extra gallery data OR external zero-shot test set
INBREAST_PREPROC_DIR = os.path.join(DATA_ROOT, "preproc_jpg_inbreast")  # 410 imgs
INBREAST_CSV = os.path.join(DATA_ROOT, "inbreast_split_preproc.csv")

# --- Reproducibility ---------------------------------------------------------
SEED = 42
TASK = "binary"                 # benign(0) vs malignant(1)
VIEWS = ["CC", "MLO"]

# --- Preprocessing (reused from MasterNotebooks; the +3% booster) ------------
# Fixed/inference CLAHE from the prior project: cv2.createCLAHE(clipLimit=4.0,
# tileGridSize=(6,6)) on the 8-bit grayscale crop. Applied deterministically at
# embed time (no random augmentation — we are training-free).
CLAHE_CLIP = 4.0
CLAHE_TILE = (6, 6)
USE_CLAHE = True                # ablation lever: embed with/without CLAHE

# --- Embedding ---------------------------------------------------------------
BATCH_SIZE = 32
DEFAULT_EMBEDDER = "resnet50"   # cheap encoder used for the first end-to-end gate

# --- Retrieval / fusion experiment grid --------------------------------------
K_GRID = [9, 15, 25, 40, 75]
DISTANCE_GRID = ["cosine", "l2"]
HEAD_GRID = ["vote", "weighted", "tip"]

# Tip-Adapter affinity (training-free): A = exp(-beta*(1 - cosine))
TIP_BETA = 5.5

# Default single config for the end-to-end gate
DEFAULT_K = 25
DEFAULT_DISTANCE = "cosine"
DEFAULT_HEAD = "tip"

os.makedirs(OUTPUT_DIR, exist_ok=True)
