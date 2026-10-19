# -*- coding: utf-8 -*-
r"""
Beyond the Color Wheel - analysis pipeline
==========================================
Depth-aware color harmony in interior scenes: validation against human judgments (AADB), depth-free control
variants, agreement between measured and monocular depth (SUN RGB-D), and within-corpus analyses
(SUN RGB-D, Interior-Style). This script produces every number, table and figure reported in the article.

Tested on Windows 10, Python 3.14 and an NVIDIA RTX 3080 (CUDA). It also runs on Linux/macOS and on Python 3.10-3.13;
without a CUDA GPU the full pipeline runs on the CPU (slower).

Two modes
---------
1) Full pipeline (downloads the data, runs CLIP and MiDaS, extracts all features, runs all analyses):
       python beyond_color_wheel.py
   Data: AADB from the Hugging Face Hub (about 2.6 GB), SUN RGB-D V1 (about 6.5 GB), Interior-Style corpus (streamed).
   The run is resumable: downloads and intermediate results are cached and reused after an interruption.

2) Reanalysis from the shared per-image features (no image data, no GPU, a few minutes):
       python beyond_color_wheel.py --from-features
   Reads outputs/data/*_per_image_features.csv and annotations/*.xlsx and recomputes all statistics, tables and
   figures. Items that show images (Figure 1 and the contact sheets) and the AADB label-matching table require the
   image data and are produced by the full pipeline only.

Folder layout (relative to this file)
-------------------------------------
    beyond_color_wheel.py          this script
    make_annotation_sheets.py      creates the sheets used for the manual interior annotation
    requirements.txt               packages for the reanalysis mode (installation commands in its header)
    requirements-full.txt          additional packages for the full pipeline
    annotations/                   manual interior labels (first annotator) and second_rater/ (reliability sample)
    data/                          downloaded datasets (full pipeline only; not part of the repository)
    cache_hf/, cache_torch/        model caches (full pipeline only)
    outputs/                       figures, tables, supplementary files, per-image data, reproducibility files
"""
import os
import sys
import time
import importlib.util
from pathlib import Path

RUN_START = time.time()
BASE_DIR = Path(__file__).resolve().parent

# Keep every cache on the same drive as the script and make Windows runs quieter and more stable.
os.environ.setdefault("HF_HOME", str(BASE_DIR / "cache_hf"))
os.environ.setdefault("TORCH_HOME", str(BASE_DIR / "cache_torch"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")          # avoids the scikit-learn KMeans memory leak on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoids the libiomp5md.dll double-initialization crash

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


class _Tee:
    # Mirrors console output into outputs\run_log.txt
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# =========================== SECTION: dependency check ===========================
FROM_FEATURES = "--from-features" in sys.argv[1:]
_REQUIRED = [
    ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
    ("skimage", "scikit-image"), ("cv2", "opencv-python"), ("PIL", "pillow"), ("matplotlib", "matplotlib"),
    ("openpyxl", "openpyxl"),
]
if not FROM_FEATURES:
    _REQUIRED += [("tqdm", "tqdm"), ("torch", "torch"), ("transformers", "transformers"),
                  ("huggingface_hub", "huggingface_hub"), ("datasets", "datasets"), ("timm", "timm")]
_missing = [pip_name for mod, pip_name in _REQUIRED if importlib.util.find_spec(mod) is None]
if _missing:
    print("Missing packages:", ", ".join(_missing))
    print("Install them with the commands in the header of requirements.txt"
          + ("" if FROM_FEATURES else " (full pipeline: requirements-full.txt)") + ", then run this script again.")
    sys.exit(1)
if sys.version_info < (3, 10):
    print(f"Python {sys.version.split()[0]} is too old: Python 3.10 or newer is required.")
    sys.exit(1)
print(f"Python {sys.version.split()[0]} | {sys.executable}")
print("Mode:", "reanalysis from the shared per-image features (--from-features)" if FROM_FEATURES else "full pipeline")
# ========================= END SECTION: dependency check =========================


# =========================== SECTION: setup ===========================
# ---------------------------------------------------------------
# Imports, configuration, output folders, figure style, helpers
# ---------------------------------------------------------------
import os, io, re, sys, json, math, time, random, zipfile, hashlib, platform, warnings, subprocess, importlib
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageCms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from skimage import color as skcolor
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, r2_score
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")

try:
    from tqdm.auto import tqdm
except ImportError:
    class _NoProgressBar:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else [])

        def update(self, n=1):
            pass

        def close(self):
            pass

    def tqdm(iterable=None, **kwargs):
        return _NoProgressBar(iterable, **kwargs)

IN_COLAB = False

# ---------------- configuration (every analysis constant lives here) ----------------
CONFIG = OrderedDict(
    RANDOM_SEED=42,
    # image processing
    WORK_MAX_SIDE=384,             # longest side (px) used for all color/depth computations; aspect ratio preserved
    DEPTH_PCT_LOW=1.0,             # percentile clipping for per-image depth normalization
    DEPTH_PCT_HIGH=99.0,
    K_LAYERS=3,                    # primary number of depth layers (near / middle / far)
    K_SENSITIVITY=[2, 3, 4, 5],    # sensitivity analysis for the number of layers
    DEPTH_KMEANS_SAMPLE=20000,     # depth values sampled per image for 1-D k-means layering
    PIXELS_PER_REGION=2000,        # CIELAB pixels sampled per region for dominant-color estimation
    REGION_COLOR_CLUSTERS=3,      # k-means clusters within a region; the largest cluster center = dominant color
    GLOBAL_PALETTE_SAMPLE=5000,    # pixels sampled for the depth-free global palette
    MIN_REGION_PIXELS=50,          # regions smaller than this are treated as missing
    LAB_HIST_BINS=[10, 16, 16],    # L*, a*, b* bins for color entropy
    CANNY_LOW=100,
    CANNY_HIGH=200,
    # models
    MIDAS_MODEL="MiDaS_small",
    CLIP_MODEL="openai/clip-vit-base-patch32",
    CLIP_BATCH=64,
    INTERIOR_PROB_THRESHOLD=0.60,  # broad indoor subset: CLIP probability mass on indoor prompts
    STRICT_INTERIOR_THRESHOLD=0.60,  # strict interior subset: CLIP probability mass on interior-space prompts
    AUDIT_N=100,                   # random strict-interior images exported for a manual precision audit
    # Study 1: AADB
    AADB_REPO="Iceclear/AADB",
    AADB_FILE="AADB.zip",
    AADB_GENERAL_N=1500,           # random general (all scene types) validation sample
    # Study 2 / 3: SUN RGB-D
    SUN_URL="https://rgbd.cs.princeton.edu/data/SUNRGBD.zip",
    N_SUN=300,
    SUN_MIN_DEPTH_M=0.1,
    SUN_MAX_DEPTH_M=15.0,
    # Study 3: Interior-Style corpus
    STYLE_REPO="hammer888/interior_style_dataset",
    N_STYLE=300,
    STYLE_SHUFFLE_BUFFER=10000,
    WATERMARK_CROP_BOTTOM=0.08,    # bottom strip removed from Interior-Style images (corner watermark)
    # statistics
    N_BOOT=2000,
    CV_SPLITS=10,
    CV_REPEATS=10,
    ALPHA=0.05,
    MIN_CATEGORY_N=10,
    PIXEL_AGREEMENT_SAMPLE=20000,
    SESOI_RHO=0.10,                # smallest positive correlation of interest for the one-sided test of H1
)

OUT = BASE_DIR / "outputs"
DATA_ROOT = BASE_DIR / "data"
FIG_DIR, TAB_DIR, DAT_DIR = OUT / "figures", OUT / "tables", OUT / "data"
ANNOTATION_DIR = OUT.parent / "annotations"   # manual interior annotations (made with make_annotation_sheets.py)
SUP_DIR, REP_DIR, CACHE_DIR, THUMB_DIR = OUT / "supplementary", OUT / "reproducibility", OUT / "cache", OUT / "thumbnails"
for d in [FIG_DIR, TAB_DIR, DAT_DIR, SUP_DIR, REP_DIR] + ([] if FROM_FEATURES else [CACHE_DIR, THUMB_DIR, DATA_ROOT]):
    d.mkdir(parents=True, exist_ok=True)
_LOG_FILE = open(OUT / "run_log.txt", "a", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _LOG_FILE)
print("\n" + "=" * 70 + f"\nRun started {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "=" * 70)

random.seed(CONFIG["RANDOM_SEED"])
np.random.seed(CONFIG["RANDOM_SEED"])


def image_rng(key):
    # Deterministic random generator per image: results do not depend on processing order or on caching.
    h = int(hashlib.sha256(f"{CONFIG['RANDOM_SEED']}|{key}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(h)


def sub_seed(rng):
    return int(rng.integers(0, 2**31 - 1))


# ---------------- figure style (English, publication format) ----------------
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 110,
    "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO),
})


def save_fig(fig, name, folder=None):
    folder = folder or FIG_DIR
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {name} (.png, .pdf)")


def save_table(df, name, folder=None, show=True):
    # Standard CSV (comma separator, dot decimal) for the public repository + native XLSX for Excel users.
    # XLSX stores real numbers, so locale-specific decimal/thousand separators cannot corrupt the values.
    folder = folder or TAB_DIR
    df.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8")
    df.to_excel(folder / f"{name}.xlsx", index=False, sheet_name=name[:31], engine="openpyxl")
    print(f"Saved table: {name} (.csv, .xlsx) | {len(df)} rows")
    if show:
        with pd.option_context("display.max_columns", 50, "display.width", 200):
            print(df.head(40).to_string(index=False))
    return df


# Column names written by earlier versions of this script (British spelling) are mapped to the current names,
# so that cached features of earlier runs are reused without recomputation.
LEGACY_COLUMNS = {"colourfulness": "colorfulness", "colour_entropy_bits": "color_entropy_bits"}


def read_features_csv(path):
    return pd.read_csv(path).rename(columns=LEGACY_COLUMNS)


# The full pipeline documents its environment in environment.json and pip_freeze.txt; a reanalysis run writes
# separate files, so the record of the original run is kept.
ENV_TAG = "_from_features" if FROM_FEATURES else ""


def log_environment():
    versions = OrderedDict(python=platform.python_version(), platform=platform.platform())
    for m in ["numpy", "pandas", "scipy", "sklearn", "skimage", "cv2", "PIL", "matplotlib", "torch",
              "transformers", "datasets", "huggingface_hub", "timm"]:
        try:
            versions[m] = getattr(importlib.import_module(m), "__version__", "unknown")
        except Exception:
            versions[m] = "not installed"
    try:
        import torch
        versions["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        versions["cuda_device"] = "n/a"
    (REP_DIR / f"environment{ENV_TAG}.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
        (REP_DIR / f"pip_freeze{ENV_TAG}.txt").write_text(freeze, encoding="utf-8")
    except Exception:
        pass
    return versions


(REP_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
print("Base folder:", BASE_DIR, "| outputs ->", OUT)
print(json.dumps(log_environment(), indent=2))
# ========================= END SECTION: setup =========================

# =========================== SECTION: color ===========================
# ---------------------------------------------------------------
# Color management, CIELAB, exact Ou & Luo (2006) model, color statistics
# ---------------------------------------------------------------
_REL_COLORIMETRIC = ImageCms.Intent.RELATIVE_COLORIMETRIC if hasattr(ImageCms, "Intent") else 1
_SRGB_PROFILE = ImageCms.createProfile("sRGB")


def to_srgb_image(img):
    # Returns (PIL RGB image in sRGB, info dict). Embedded ICC profiles are converted to sRGB.
    info = {"mode_in": img.mode, "had_icc": False, "icc_converted": False}
    icc = img.info.get("icc_profile")
    if icc and img.mode in ("RGB", "RGBA", "CMYK"):
        info["had_icc"] = True
        try:
            src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            base = img.convert("RGB") if img.mode == "RGBA" else img
            img = ImageCms.profileToProfile(base, src_profile, _SRGB_PROFILE,
                                            renderingIntent=_REL_COLORIMETRIC, outputMode="RGB")
            info["icc_converted"] = True
        except Exception as exc:
            info["icc_error"] = str(exc)[:80]
    elif icc:
        info["had_icc"] = True
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img, info


def resize_max_side(img, max_side):
    w, h = img.size
    s = max_side / float(max(w, h))
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return img


def rgb_to_lab(rgb_uint8):
    return skcolor.rgb2lab(rgb_uint8.astype(np.float64) / 255.0, illuminant="D65", observer="2").astype(np.float32)


def lab_to_lch(lab):
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360.0
    return L, C, h


OU_LUO_2006 = dict(
    HC=(0.04, 0.53, 0.8, 0.045), DC_CHROMA_DIVISOR=1.46,
    HLSUM=(0.28, 0.54, -3.88, 0.029),
    HDL=(0.14, 0.15, -2.0, 0.2),
    EC=(0.5, 0.5, -2.0, 0.5),
    HS=(-0.08, -0.14, 50.0, -0.07, 90.0),
    EY=(0.22, -12.8, 10.0),
)


def _hsy(L, C, h):
    ec0, ec1, ec2, ec3 = OU_LUO_2006["EC"]
    hs0, hs1, hs2, hs3, hs4 = OU_LUO_2006["HS"]
    ey0, ey1, ey2 = OU_LUO_2006["EY"]
    EC = ec0 + ec1 * np.tanh(ec2 + ec3 * C)
    HS = hs0 + hs1 * np.sin(np.radians(h + hs2)) + hs3 * np.sin(np.radians(2.0 * h + hs4))
    x = (90.0 - h) / 10.0
    EY = ((ey0 * L + ey1) / ey2) * np.exp(x - np.exp(x))
    return EC * (HS + EY), EC, HS, EY


def ou_luo_2006(lab1, lab2, return_components=False):
    # Exact two-color harmony model of Ou & Luo (2006). Inputs: CIELAB arrays [..., 3]. Symmetric in its arguments.
    L1, C1, h1 = lab_to_lch(lab1)
    L2, C2, h2 = lab_to_lch(lab2)
    dh = (h2 - h1 + 180.0) % 360.0 - 180.0
    dH_ab = 2.0 * np.sqrt(C1 * C2) * np.sin(np.radians(dh) / 2.0)
    dC_ab = C1 - C2
    dC = np.sqrt(dH_ab ** 2 + (dC_ab / OU_LUO_2006["DC_CHROMA_DIVISOR"]) ** 2)
    a, b, c, d = OU_LUO_2006["HC"]
    HC = a + b * np.tanh(c - d * dC)
    a, b, c, d = OU_LUO_2006["HLSUM"]
    HLsum = a + b * np.tanh(c + d * (L1 + L2))
    a, b, c, d = OU_LUO_2006["HDL"]
    HdL = a + b * np.tanh(c + d * np.abs(L1 - L2))
    HL = HLsum + HdL
    hsy1 = _hsy(L1, C1, h1)[0]
    hsy2 = _hsy(L2, C2, h2)[0]
    HH = hsy1 + hsy2
    CH = HC + HL + HH
    if return_components:
        return dict(CH=CH, HC=HC, HL=HL, HLsum=HLsum, HdL=HdL, HH=HH, HSY1=hsy1, HSY2=hsy2)
    return CH


def colorfulness_hs(rgb_uint8):
    # Hasler & Suesstrunk (2003) colorfulness metric M, computed on 8-bit sRGB values.
    rgb = rgb_uint8.astype(np.float64)
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def lab_entropy_bits(lab, bins=None):
    bins = bins or CONFIG["LAB_HIST_BINS"]
    hist, _ = np.histogramdd(lab.reshape(-1, 3), bins=bins, range=[(0, 100), (-128, 128), (-128, 128)])
    p = hist.ravel() / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def edge_density(rgb_uint8):
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    return float((cv2.Canny(gray, CONFIG["CANNY_LOW"], CONFIG["CANNY_HIGH"]) > 0).mean())


def ou_luo_self_checks():
    # Analytical checks of the implementation against closed-form values and the model's stated principles.
    rng = np.random.default_rng(0)
    rows = []
    c1 = np.column_stack([rng.uniform(5, 95, 500), rng.uniform(-80, 80, 500), rng.uniform(-80, 80, 500)])
    c2 = np.column_stack([rng.uniform(5, 95, 500), rng.uniform(-80, 80, 500), rng.uniform(-80, 80, 500)])
    sym = float(np.max(np.abs(ou_luo_2006(c1, c2) - ou_luo_2006(c2, c1))))
    rows.append(("Symmetry: max |CH(a,b) - CH(b,a)|", sym, "0", sym < 1e-12))
    same = np.array([50.0, 20.0, 10.0])
    comp = ou_luo_2006(same, same, return_components=True)
    exp_hc = 0.04 + 0.53 * np.tanh(0.8)
    rows.append(("H_C for identical colors", float(comp["HC"]), f"{exp_hc:.6f}", abs(comp["HC"] - exp_hc) < 1e-12))
    exp_hdl = 0.14 + 0.15 * np.tanh(-2.0)
    rows.append(("H_dL for dL = 0", float(comp["HdL"]), f"{exp_hdl:.6f}", abs(comp["HdL"] - exp_hdl) < 1e-12))
    hs_blue = _hsy(60.0, 60.0, 250.0)[2]
    hs_red = _hsy(60.0, 60.0, 30.0)[2]
    rows.append(("Hue effect H_S: blue (250 deg) > red (30 deg)", float(hs_blue - hs_red), "> 0", hs_blue > hs_red))
    ey = _hsy(80.0, 60.0, 90.0)[3]
    exp_ey = ((0.22 * 80 - 12.8) / 10.0) * np.exp(-1.0)
    rows.append(("E_Y at h = 90 deg, L* = 80", float(ey), f"{exp_ey:.6f}", abs(ey - exp_ey) < 1e-12))
    bright = ou_luo_2006(np.array([85.0, 5.0, 5.0]), np.array([80.0, 5.0, 5.0]), True)["HLsum"]
    dark = ou_luo_2006(np.array([25.0, 5.0, 5.0]), np.array([20.0, 5.0, 5.0]), True)["HLsum"]
    rows.append(("Lightness-sum effect: light pair > dark pair", float(bright - dark), "> 0", bright > dark))
    ch = ou_luo_2006(c1, c2)
    rows.append(("Range of CH over 500 random pairs within theoretical bounds [-2.28, 2.62]",
                 float(ch.min()), f"max = {ch.max():.3f}", (ch.min() > -2.28) and (ch.max() < 2.62)))
    df = pd.DataFrame(rows, columns=["check", "value", "expected", "passed"])
    assert df["passed"].all(), df
    return df


ou_luo_checks = ou_luo_self_checks()
save_table(ou_luo_checks, "TableS1_ou_luo_implementation_checks", folder=SUP_DIR)
# ========================= END SECTION: color =========================

# =========================== SECTION: depth ===========================
# ---------------------------------------------------------------
# Depth normalization, layering, partitions and per-image features
# ---------------------------------------------------------------
def normalize_closeness(disparity, low=None, high=None):
    low = CONFIG["DEPTH_PCT_LOW"] if low is None else low
    high = CONFIG["DEPTH_PCT_HIGH"] if high is None else high
    d = np.asarray(disparity, dtype=np.float32).copy()
    valid = np.isfinite(d)
    out = np.full(d.shape, np.nan, dtype=np.float32)
    if valid.sum() < 100:
        return out
    lo, hi = np.percentile(d[valid], [low, high])
    if not np.isfinite(hi - lo) or (hi - lo) < 1e-8:
        return out
    out[valid] = np.clip((d[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out


def depth_layers(closeness, K, rng):
    # 1-D k-means on closeness; returns int map, 0 = nearest layer, -1 = invalid pixel.
    labels = np.full(closeness.shape, -1, dtype=np.int16)
    valid = np.isfinite(closeness)
    vals = closeness[valid]
    if vals.size < K * 50:
        return labels
    n = min(CONFIG["DEPTH_KMEANS_SAMPLE"], vals.size)
    sample = rng.choice(vals, size=n, replace=False)
    if np.unique(np.round(sample, 4)).size < K:
        return labels
    km = KMeans(n_clusters=K, n_init=5, random_state=sub_seed(rng)).fit(sample.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.ravel())[::-1]          # descending closeness: index 0 = nearest
    labels[valid] = np.abs(vals[:, None] - centers[None, :]).argmin(axis=1)
    return labels


def band_layers(shape, K):
    # Horizontal bands; band 0 = bottom of the image (typically nearest floor area in interiors).
    h, w = shape
    rows = np.arange(h)
    band = np.minimum(((h - 1 - rows) * K) // h, K - 1).astype(np.int16)
    return np.repeat(band[:, None], w, axis=1)


def dominant_color(px, rng):
    k = min(CONFIG["REGION_COLOR_CLUSTERS"], len(np.unique(np.round(px, 1), axis=0)))
    if k <= 1:
        return px.mean(axis=0)
    km = KMeans(n_clusters=k, n_init=3, random_state=sub_seed(rng)).fit(px)
    counts = np.bincount(km.labels_, minlength=k)
    return km.cluster_centers_[int(np.argmax(counts))]


def region_colors(lab, labels, K, rng):
    flat_lab = lab.reshape(-1, 3)
    flat_lab_lbl = labels.ravel()
    total = max(int((flat_lab_lbl >= 0).sum()), 1)
    cols, areas = [], []
    for k in range(K):
        idx = np.flatnonzero(flat_lab_lbl == k)
        areas.append(idx.size / total)
        if idx.size < CONFIG["MIN_REGION_PIXELS"]:
            cols.append(None)
            continue
        if idx.size > CONFIG["PIXELS_PER_REGION"]:
            idx = rng.choice(idx, CONFIG["PIXELS_PER_REGION"], replace=False)
        cols.append(dominant_color(flat_lab[idx], rng))
    return cols, np.asarray(areas)


def global_palette(lab, K, rng):
    flat = lab.reshape(-1, 3)
    n = min(CONFIG["GLOBAL_PALETTE_SAMPLE"], flat.shape[0])
    px = flat[rng.choice(flat.shape[0], n, replace=False)]
    k = min(K, len(np.unique(np.round(px, 1), axis=0)))
    if k < 2:
        return [None] * K, np.zeros(K)
    km = KMeans(n_clusters=k, n_init=3, random_state=sub_seed(rng)).fit(px)
    props = np.bincount(km.labels_, minlength=k) / n
    cols = [km.cluster_centers_[i] for i in range(k)] + [None] * (K - k)
    return cols, np.concatenate([props, np.zeros(K - k)])


def pooled_harmony(cols, areas, scheme="equal"):
    valid = [i for i, c in enumerate(cols) if c is not None]
    if len(valid) < 2:
        return np.nan
    num = den = 0.0
    for a in range(len(valid)):
        for b in range(a + 1, len(valid)):
            i, j = valid[a], valid[b]
            ch = float(ou_luo_2006(cols[i], cols[j]))
            if scheme == "equal":
                w = 1.0
            elif scheme == "adjacency":
                w = 1.0 / abs(i - j)
            elif scheme == "area":
                w = float(areas[i] * areas[j])
            else:
                raise ValueError(scheme)
            num += w * ch
            den += w
    return num / den if den > 0 else np.nan


def image_features(rgb, closeness, rng, donor_closeness=None, k_list=None, keep_maps=False):
    k_main = CONFIG["K_LAYERS"]
    k_list = sorted(set(k_list or CONFIG["K_SENSITIVITY"]) | {k_main})
    lab = rgb_to_lab(rgb)
    L, C, _ = lab_to_lch(lab)
    f = OrderedDict()
    f["colorfulness"] = colorfulness_hs(rgb)
    f["color_entropy_bits"] = lab_entropy_bits(lab)
    f["mean_L"] = float(L.mean())
    f["mean_C"] = float(C.mean())
    f["edge_density"] = edge_density(rgb)
    f["depth_dispersion"] = float(np.nanvar(closeness)) if np.isfinite(closeness).any() else np.nan
    f["valid_depth_fraction"] = float(np.isfinite(closeness).mean())
    maps = {}
    for K in k_list:
        lay = depth_layers(closeness, K, rng)
        cols, areas = region_colors(lab, lay, K, rng)
        f[f"H_depth_K{K}"] = pooled_harmony(cols, areas, "equal")
        if K == k_main:
            f["H_depth"] = f[f"H_depth_K{K}"]
            f["H_depth_adjw"] = pooled_harmony(cols, areas, "adjacency")
            f["H_depth_areaw"] = pooled_harmony(cols, areas, "area")
            f["n_valid_layers"] = int(sum(c is not None for c in cols))
            maps.update(layers=lay, layer_colors=cols)
    bands = band_layers(rgb.shape[:2], k_main)
    cols_b, areas_b = region_colors(lab, bands, k_main, rng)
    f["H_bands"] = pooled_harmony(cols_b, areas_b, "equal")
    cols_g, areas_g = global_palette(lab, k_main, rng)
    f["H_global"] = pooled_harmony(cols_g, areas_g, "equal")
    if donor_closeness is not None:
        donor = cv2.resize(donor_closeness.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        lay_s = depth_layers(donor, k_main, rng)
        cols_s, areas_s = region_colors(lab, lay_s, k_main, rng)
        f["H_shuffled"] = pooled_harmony(cols_s, areas_s, "equal")
    else:
        f["H_shuffled"] = np.nan
    if keep_maps:
        maps.update(closeness=closeness)
        return f, maps
    return f


def decode_sun_depth(raw):
    # SUN RGB-D toolbox convention: 16-bit value rotated right by 3 bits gives millimeters.
    raw = np.asarray(raw).astype(np.uint16)
    mm = np.bitwise_or(np.right_shift(raw, 3), np.left_shift(raw, 13)).astype(np.float32)
    meters = mm / 1000.0
    bad = (raw == 0) | (meters < CONFIG["SUN_MIN_DEPTH_M"]) | (meters > CONFIG["SUN_MAX_DEPTH_M"])
    meters[bad] = np.nan
    return meters


def make_thumb(rgb, size=160):
    im = Image.fromarray(rgb)
    im.thumbnail((size, size), Image.LANCZOS)
    return np.asarray(im)


def contact_sheet(items, title, name, ncols=8, folder=None):
    if not items:
        print("No images for contact sheet", name)
        return
    nrows = int(math.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.35, nrows * 1.45))
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < len(items):
            img, cap = items[i]
            ax.imshow(img)
            ax.set_title(str(cap)[:22], fontsize=5.5)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    save_fig(fig, name, folder=folder or SUP_DIR)
# ========================= END SECTION: depth =========================

# =========================== SECTION: selftest ===========================
# ---------------------------------------------------------------
# Synthetic self-test of the full feature pipeline (runs in seconds)
# ---------------------------------------------------------------
def _synthetic_scene(h=240, w=320):
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[: h // 3] = (70, 110, 190)          # far: blue wall
    rgb[h // 3: 2 * h // 3] = (225, 210, 180)   # middle: beige
    rgb[2 * h // 3:] = (120, 80, 50)        # near: brown floor
    rgb = np.clip(rgb.astype(int) + np.random.default_rng(1).integers(-6, 7, rgb.shape), 0, 255).astype(np.uint8)
    disparity = np.repeat(np.linspace(0.2, 1.0, h)[:, None], w, axis=1).astype(np.float32)  # bottom = closest
    return rgb, disparity


_rgb, _disp = _synthetic_scene()
_clo = normalize_closeness(_disp)
_rng = image_rng("selftest")
_lay = depth_layers(_clo, 3, _rng)
assert _lay[-1, 0] == 0 and _lay[0, 0] == 2, "Layer 0 must be the nearest (bottom) region in the synthetic scene"
_lay_affine = depth_layers(normalize_closeness(3.0 * _disp + 7.0), 3, image_rng("selftest"))
assert (depth_layers(_clo, 3, image_rng("selftest")) == _lay_affine).all(), "Layering must be invariant to affine rescaling of disparity"
_bands = band_layers(_rgb.shape[:2], 3)
assert _bands[-1, 0] == 0 and _bands[0, 0] == 2
_f = image_features(_rgb, _clo, image_rng("selftest"), donor_closeness=_clo[::-1].copy())
for key in ["H_depth", "H_global", "H_bands", "H_shuffled", "H_depth_K2", "H_depth_K5", "colorfulness"]:
    assert np.isfinite(_f[key]), key
_mm = np.array([[2500, 0], [800, 16000]], dtype=np.uint16)
_raw = (np.left_shift(_mm, 3) | np.right_shift(_mm, 13)).astype(np.uint16)
_dec = decode_sun_depth(_raw)
assert abs(_dec[0, 0] - 2.5) < 1e-6 and np.isnan(_dec[0, 1]) and abs(_dec[1, 0] - 0.8) < 1e-6 and np.isnan(_dec[1, 1])
_icc_img, _icc_info = to_srgb_image(Image.fromarray(_rgb))
assert _icc_info["icc_converted"] is False and np.array_equal(np.asarray(_icc_img), _rgb)
print("Self-test passed. Synthetic features:")
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in _f.items()})
# ========================= END SECTION: selftest =========================

# =========================== SECTION: models ===========================
# ---------------------------------------------------------------
# Models: MiDaS (monocular depth) and CLIP (zero-shot interior filter), with local GPU checks
# ---------------------------------------------------------------
if not FROM_FEATURES:   # needs the image data and/or the models
    import torch
    import torch.nn.functional as F

    torch.manual_seed(CONFIG["RANDOM_SEED"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    # Full FP32 precision on Ampere GPUs (no TF32), so that depth maps match other machines as closely as possible.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass


    DEVICE = torch.device("cpu")
    print("PyTorch", torch.__version__, "| built for CUDA", torch.version.cuda)
    if torch.cuda.is_available():
        try:
            _x = torch.ones(1, 3, 16, 16, device="cuda")
            _w = torch.ones(4, 3, 3, 3, device="cuda")
            float(F.conv2d(_x, _w).sum().item())
            DEVICE = torch.device("cuda")
            cap = torch.cuda.get_device_capability(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {torch.cuda.get_device_name(0)} | compute capability {cap[0]}.{cap[1]} | {mem:.1f} GB")
        except Exception as exc:
            print("WARNING: a CUDA GPU was found but this PyTorch build cannot run on it:", repr(exc)[:160])
            print("  Update the NVIDIA driver or reinstall torch with the command in requirements-full.txt. Continuing on CPU (slow).")
    else:
        print("WARNING: PyTorch sees no CUDA GPU. On Windows, a plain 'pip install torch' gives a CPU-only build.")
        print("  Reinstall torch with the CUDA command in requirements-full.txt. Running on CPU (slow).")

    # ---- GitHub-tolerant loading of MiDaS through torch.hub ----
    # torch.hub normally asks github.com on every call which branch is the default one; a GitHub hiccup (e.g. HTTP 504)
    # then aborts the run. Here the branches of the three official repositories are fixed, fork validation is skipped for
    # them, the repositories are pre-trusted (no console prompt), and every download is retried. After the first
    # successful run everything is loaded from the local cache in cache_torch without contacting GitHub.
    import torch.hub as _torch_hub

    _HUB_REFS = {("isl-org", "MiDaS"): "master", ("intel-isl", "MiDaS"): "master",
                 ("rwightman", "gen-efficientnet-pytorch"): "master"}
    if hasattr(_torch_hub, "_parse_repo_info"):
        _orig_parse_repo_info = _torch_hub._parse_repo_info

        def _parse_repo_info_fixed(github):
            if ":" not in github:
                owner, name = github.split("/")
                if (owner, name) in _HUB_REFS:
                    return owner, name, _HUB_REFS[(owner, name)]
            return _orig_parse_repo_info(github)

        _torch_hub._parse_repo_info = _parse_repo_info_fixed
    if hasattr(_torch_hub, "_validate_not_a_forked_repo"):
        _torch_hub._validate_not_a_forked_repo = lambda *args, **kwargs: None

    _hub_dir = Path(torch.hub.get_dir())
    _hub_dir.mkdir(parents=True, exist_ok=True)
    _trusted_file = _hub_dir / "trusted_list"
    _trusted = set(_trusted_file.read_text(encoding="utf-8").split()) if _trusted_file.exists() else set()
    _needed = {"isl-org_MiDaS", "intel-isl_MiDaS", "rwightman_gen-efficientnet-pytorch"}
    if not _needed <= _trusted:
        _trusted_file.write_text("\n".join(sorted(_trusted | _needed)) + "\n", encoding="utf-8")


    def hub_load_with_retry(repo, entry, attempts=6, wait=20):
        last = None
        for i in range(1, attempts + 1):
            try:
                return torch.hub.load(repo, entry, trust_repo=True, skip_validation=True)
            except Exception as exc:
                last = exc
                print(f"  torch.hub: could not load {repo} / {entry} ({type(exc).__name__}: {str(exc)[:100]}); "
                      f"attempt {i}/{attempts}, retrying in {wait} s")
                time.sleep(wait)
        raise RuntimeError(
            f"MiDaS could not be downloaded from GitHub after {attempts} attempts. Check the internet connection and run "
            "the script again (everything already done is reused).") from last


    MIDAS = hub_load_with_retry("isl-org/MiDaS", CONFIG["MIDAS_MODEL"]).to(DEVICE).eval()
    _midas_transforms = hub_load_with_retry("isl-org/MiDaS", "transforms")
    MIDAS_TRANSFORM = _midas_transforms.small_transform if CONFIG["MIDAS_MODEL"] == "MiDaS_small" else _midas_transforms.dpt_transform


    @torch.no_grad()
    def midas_disparity(rgb):
        # Relative inverse depth (larger = closer), resampled to the input resolution.
        batch = MIDAS_TRANSFORM(rgb).to(DEVICE)
        pred = MIDAS(batch)
        pred = F.interpolate(pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False).squeeze()
        return pred.float().cpu().numpy()


    INTERIOR_PROMPTS = [
        "a photo of an indoor room", "a photo of a living room", "a photo of a bedroom", "a photo of a kitchen",
        "a photo of a bathroom", "a photo of an office interior", "a photo of a dining room",
        "a photo of a hotel lobby", "a photo of a restaurant interior", "a photo of a classroom",
    ]
    NON_INTERIOR_PROMPTS = [
        "a photo of an outdoor landscape", "a photo of a city street", "a photo of a building exterior",
        "a portrait photo of a person", "a close-up photo of an object", "a photo of an animal",
        "a close-up photo of food", "a photo of the sky", "a photo of a beach or the sea", "a photo of a forest",
    ]
    _CLIP = {}


    def get_clip(attempts=6, wait=30):
        if not _CLIP:
            from transformers import CLIPModel, CLIPProcessor
            for i in range(1, attempts + 1):
                try:
                    _CLIP["model"] = CLIPModel.from_pretrained(CONFIG["CLIP_MODEL"]).to(DEVICE).eval()
                    _CLIP["proc"] = CLIPProcessor.from_pretrained(CONFIG["CLIP_MODEL"])
                    break
                except Exception as exc:
                    _CLIP.clear()
                    if i == attempts:
                        raise
                    print(f"  CLIP download problem ({type(exc).__name__}); attempt {i}/{attempts}, retrying in {wait} s")
                    time.sleep(wait)
        return _CLIP["model"], _CLIP["proc"]



    @torch.no_grad()
    def clip_interior_probability(pil_images):
        model, proc = get_clip()
        prompts = INTERIOR_PROMPTS + NON_INTERIOR_PROMPTS
        inputs = proc(text=prompts, images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        probs = model(**inputs).logits_per_image.softmax(dim=-1).float().cpu().numpy()
        return probs[:, : len(INTERIOR_PROMPTS)].sum(axis=1)


    def release_clip():
        _CLIP.clear()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


    # Prompts for the strict interior filter: interior-space photographs compete with people-centered indoor photographs
    # (events, portraits, crowds) and with outdoor/object scenes.
    SPACE_PROMPTS = [
        "a photo of an empty room", "an interior design photo of a living room", "an interior design photo of a bedroom",
        "an interior design photo of a kitchen", "an interior design photo of a bathroom",
        "an interior design photo of a dining room", "a photo of an office interior", "a photo of a hotel room",
        "a photo of a hallway", "a photo of a library interior", "a photo of a restaurant interior",
        "an architectural photo of an interior space",
    ]
    PEOPLE_PROMPTS = [
        "a photo of a group of people", "a portrait photo of a person", "a photo of people at an event",
        "a photo of a crowd", "a photo of people at a party", "a photo of a speaker giving a talk",
    ]
    OTHER_PROMPTS = [
        "a photo of an outdoor landscape", "a photo of a city street", "a photo of a building exterior",
        "a close-up photo of an object", "a photo of an animal", "a close-up photo of food", "a photo of the sky",
        "a photo of a beach or the sea", "a photo of a forest",
    ]


    @torch.no_grad()
    def clip_prompt_probabilities(pil_images, prompts):
        # Softmax probabilities over the given prompt set (one row per image).
        model, proc = get_clip()
        inputs = proc(text=prompts, images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        return model(**inputs).logits_per_image.softmax(dim=-1).float().cpu().numpy()


    _test_disp = midas_disparity(_synthetic_scene()[0])
    print("MiDaS ready on", DEVICE, "| output shape:", _test_disp.shape)
# ========================= END SECTION: models =========================

# =========================== SECTION: stats ===========================
# ---------------------------------------------------------------
# Statistical helper functions (implemented with numpy/scipy only)
# ---------------------------------------------------------------
def _complete(*arrays):
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    mask = np.all([np.isfinite(a) for a in arrs], axis=0)
    return [a[mask] for a in arrs]


def _spearman(x, y):
    return float(np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1])


def spearman_ci(x, y, n_boot=None, seed=None):
    n_boot = n_boot or CONFIG["N_BOOT"]
    x, y = _complete(x, y)
    n = len(x)
    if n < 10:
        return dict(n=n, rho=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan)
    res = stats.spearmanr(x, y)
    rng = np.random.default_rng(CONFIG["RANDOM_SEED"] if seed is None else seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = _spearman(x[idx], y[idx])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return dict(n=n, rho=float(res.statistic), ci_low=float(lo), ci_high=float(hi), p=float(res.pvalue))


def mrr_test(r1, r2, r12, n):
    # Meng, Rosenthal & Rubin (1992): compare two dependent correlations sharing one variable.
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    rbar2 = (r1 ** 2 + r2 ** 2) / 2.0
    f = min((1.0 - r12) / (2.0 * (1.0 - rbar2)), 1.0)
    h = (1.0 - f * rbar2) / (1.0 - rbar2)
    z = (z1 - z2) * np.sqrt((n - 3) / (2.0 * (1.0 - r12) * h))
    return float(z), float(2.0 * stats.norm.sf(abs(z)))


def delta_rho(y, x1, x2, n_boot=None, seed=None):
    n_boot = n_boot or CONFIG["N_BOOT"]
    y, x1, x2 = _complete(y, x1, x2)
    n = len(y)
    r1, r2, r12 = _spearman(y, x1), _spearman(y, x2), _spearman(x1, x2)
    rng = np.random.default_rng(CONFIG["RANDOM_SEED"] if seed is None else seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        d[b] = _spearman(y[idx], x1[idx]) - _spearman(y[idx], x2[idx])
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    z, p = mrr_test(r1, r2, r12, n)
    p_boot = float(min(1.0, 2.0 * min(np.mean(d <= 0), np.mean(d >= 0))))
    return dict(n=n, rho_1=r1, rho_2=r2, rho_12=r12, delta_rho=r1 - r2, ci_low=float(lo), ci_high=float(hi),
                z_mrr=z, p_mrr=p, p_bootstrap=p_boot)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def partial_spearman(y, x, covariates, n_boot=None, seed=None):
    n_boot = n_boot or CONFIG["N_BOOT"]
    cols = _complete(y, x, *[covariates[:, j] for j in range(covariates.shape[1])])
    y, x, Z = cols[0], cols[1], np.column_stack(cols[2:])
    n, k = len(y), Z.shape[1]

    def _pr(idx):
        ry, rx = stats.rankdata(y[idx]), stats.rankdata(x[idx])
        rz = np.column_stack([stats.rankdata(Z[idx, j]) for j in range(k)])
        D = np.column_stack([np.ones(len(idx)), rz])
        ey = ry - D @ np.linalg.lstsq(D, ry, rcond=None)[0]
        ex = rx - D @ np.linalg.lstsq(D, rx, rcond=None)[0]
        return float(np.corrcoef(ey, ex)[0, 1])

    r = _pr(np.arange(n))
    dfree = n - 2 - k
    t = r * np.sqrt(dfree / max(1e-12, 1 - r ** 2))
    p = float(2 * stats.t.sf(abs(t), dfree))
    rng = np.random.default_rng(CONFIG["RANDOM_SEED"] if seed is None else seed)
    boots = np.array([_pr(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return dict(n=n, partial_rho=r, ci_low=float(lo), ci_high=float(hi), df=dfree, p=p)


def corrected_resampled_ttest(diffs, n_train, n_test, alpha=None):
    alpha = alpha or CONFIG["ALPHA"]
    d = np.asarray(diffs, dtype=float)
    J = len(d)
    mean, var = d.mean(), d.var(ddof=1)
    se = np.sqrt((1.0 / J + n_test / n_train) * var)
    t = mean / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), J - 1)) if np.isfinite(t) else np.nan
    crit = stats.t.ppf(1 - alpha / 2, J - 1)
    return dict(mean_diff=float(mean), ci_low=float(mean - crit * se), ci_high=float(mean + crit * se),
                t=float(t), df=J - 1, p=p)


def incremental_validity(df, outcome, base_cols, models, n_splits=None, n_repeats=None, seed=None):
    # models: dict name -> list of predictor columns. Returns per-fold R^2 table and fold sizes.
    n_splits = n_splits or CONFIG["CV_SPLITS"]
    n_repeats = n_repeats or CONFIG["CV_REPEATS"]
    needed = sorted(set([outcome] + base_cols + [c for cols in models.values() for c in cols]))
    data = df[needed].replace([np.inf, -np.inf], np.nan).dropna()
    y = data[outcome].to_numpy()
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                        random_state=CONFIG["RANDOM_SEED"] if seed is None else seed)
    rows, sizes = [], []
    for fold, (tr, te) in enumerate(rkf.split(data)):
        sizes.append((len(tr), len(te)))
        row = {"fold": fold}
        for name, cols in models.items():
            X = data[cols].to_numpy()
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
            model.fit(X[tr], y[tr])
            row[name] = r2_score(y[te], model.predict(X[te]))
        rows.append(row)
    n_train = float(np.mean([s[0] for s in sizes]))
    n_test = float(np.mean([s[1] for s in sizes]))
    return pd.DataFrame(rows), n_train, n_test, len(data)


def lin_ccc(x, y):
    x, y = _complete(x, y)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    return float(2 * cov / (vx + vy + (mx - my) ** 2))


def bootstrap_stat(func, *arrays, n_boot=None, seed=None):
    n_boot = n_boot or CONFIG["N_BOOT"]
    arrs = _complete(*arrays)
    n = len(arrs[0])
    rng = np.random.default_rng(CONFIG["RANDOM_SEED"] if seed is None else seed)
    vals = np.array([func(*[a[idx] for a in arrs]) for idx in (rng.integers(0, n, n) for _ in range(n_boot))])
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def bland_altman(reference, new):
    reference, new = _complete(reference, new)
    diff = new - reference
    md, sd = diff.mean(), diff.std(ddof=1)
    return dict(n=len(diff), mean_diff=float(md), sd_diff=float(sd), loa_low=float(md - 1.96 * sd), loa_high=float(md + 1.96 * sd))


def ols_hc3(y, X, names):
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    names = ["intercept"] + list(names)
    n, p = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    hat = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    omega = (resid / (1 - hat)) ** 2
    cov = XtX_inv @ (X.T * omega) @ X @ XtX_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    pv = 2 * stats.t.sf(np.abs(t), n - p)
    crit = stats.t.ppf(0.975, n - p)
    r2 = 1 - resid.var() / y.var()
    return pd.DataFrame(dict(term=names, estimate=beta, se_hc3=se, t=t, p=pv,
                             ci_low=beta - crit * se, ci_high=beta + crit * se)), float(r2), n


def kruskal_epsilon2(groups):
    groups = [np.asarray(g, float)[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g) > 0]
    H, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    return dict(H=float(H), df=len(groups) - 1, p=float(p), epsilon2=float(H / (n - 1)), n=n, k=len(groups))


def mann_whitney_summary(a, b, n_boot=None, seed=None):
    a = np.asarray(a, float)[np.isfinite(a)]
    b = np.asarray(b, float)[np.isfinite(b)]
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    r_rb = 2.0 * res.statistic / (len(a) * len(b)) - 1.0       # > 0: first sample tends to be larger
    rng = np.random.default_rng(CONFIG["RANDOM_SEED"] if seed is None else seed)
    n_boot = n_boot or CONFIG["N_BOOT"]
    diffs = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(n_boot)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(n_1=len(a), n_2=len(b), median_1=float(np.median(a)), median_2=float(np.median(b)),
                median_diff=float(np.median(a) - np.median(b)), ci_low=float(lo), ci_high=float(hi),
                U=float(res.statistic), p=float(res.pvalue), rank_biserial_r=float(r_rb))


def compare_independent_spearman(r1, n1, r2, n2):
    # Fisher z with the Fieller, Hartley & Pearson (1957) variance for Spearman coefficients.
    se = np.sqrt(1.06 / (n1 - 3) + 1.06 / (n2 - 3))
    z = (np.arctanh(r1) - np.arctanh(r2)) / se
    return float(z), float(2 * stats.norm.sf(abs(z)))


def as_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


ANNOTATION_FILE = ANNOTATION_DIR / "interior_annotation.xlsx"
SECOND_RATER_FILE = ANNOTATION_DIR / "second_rater" / "second_rater_annotation.xlsx"


def attach_manual_labels(df):
    # Manual verification of the strict CLIP candidates (annotations/interior_annotation.xlsx, sheet "labels":
    # key, label 1 = interior space, 0 = not). Adds interior_label and is_interior_verified.
    df = df.copy()
    df["interior_label"] = np.nan
    if ANNOTATION_FILE.exists():
        ann = pd.read_excel(ANNOTATION_FILE, sheet_name="labels")
        ann["label"] = pd.to_numeric(ann["label"], errors="coerce")
        df["interior_label"] = df["key"].astype(str).map(dict(zip(ann["key"].astype(str), ann["label"])))
        cand = df["is_interior_strict"]
        n_lab = int(df.loc[cand, "interior_label"].notna().sum())
        print(f"Manual annotation: {n_lab} of {int(cand.sum())} strict candidates labeled | "
              f"verified interiors: {int((df.loc[cand, 'interior_label'] == 1).sum())}")
        if n_lab < int(cand.sum()):
            print("WARNING: not all strict candidates are labeled; the verified subset is incomplete.")
    else:
        print(f"No manual annotation file found ({ANNOTATION_FILE}); the verified interior subset is skipped.")
    df["is_interior_verified"] = df["is_interior_strict"] & (df["interior_label"] == 1)
    return df


print("Statistical toolbox ready.")
# ========================= END SECTION: stats =========================

# =========================== SECTION: aadb load ===========================
# ---------------------------------------------------------------
# Study 1 - AADB download, label parsing and image matching
# ---------------------------------------------------------------
if not FROM_FEATURES:   # needs the image data and/or the models
    from huggingface_hub import hf_hub_download

    AADB_LOCAL_LABEL_DIR = DATA_ROOT / "aadb_labels"
    AADB_LOCAL_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

    def hf_download_with_retry(attempts=8, wait=30, **kwargs):
        # Hugging Face downloads resume automatically; transient network errors are retried.
        for i in range(1, attempts + 1):
            try:
                return hf_hub_download(**kwargs)
            except Exception as exc:
                if i == attempts:
                    raise
                print(f"  Hugging Face download problem ({type(exc).__name__}: {str(exc)[:100]}); attempt {i}/{attempts}, retrying in {wait} s")
                time.sleep(wait)


    aadb_zip_path = hf_download_with_retry(repo_id=CONFIG["AADB_REPO"], filename=CONFIG["AADB_FILE"], repo_type="dataset",
                                           local_dir=str(DATA_ROOT / "aadb"))
    AADB_ZIP = zipfile.ZipFile(aadb_zip_path)
    aadb_members = [n for n in AADB_ZIP.namelist() if not n.endswith("/")]
    aadb_images = [n for n in aadb_members if n.lower().endswith(IMG_EXT)]
    aadb_other = [n for n in aadb_members if not n.lower().endswith(IMG_EXT)]
    print(f"AADB archive: {len(aadb_members)} files | {len(aadb_images)} images | {len(aadb_other)} other files")
    print(pd.Series([("/".join(n.split("/")[:-1]) or "(root)") for n in aadb_images]).value_counts().to_string())
    print("Non-image files (first 40):", aadb_other[:40])

    LABEL_RE = re.compile(r"imgList(?P<split>[A-Za-z]+?)Regression_(?P<attr>[A-Za-z]+)\.txt$", re.IGNORECASE)
    KEEP_ATTRS = {"colorharmony": "ColorHarmony", "score": "AestheticScore"}


    def aadb_key(path):
        base = os.path.basename(str(path)).lower().strip()
        base = re.sub(r"\.(jpg|jpeg|png|bmp)$", "", base)
        return re.sub(r"^[0-9]+(\.[0-9]+)?_", "", base)   # the 'new test' images carry a score prefix in their file name


    def parse_label_file(text, split, attr, source):
        rows = []
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                value = float(parts[-1])
            except ValueError:
                continue
            rows.append(dict(key=aadb_key(parts[0]), raw_name=parts[0], split=split, attribute=attr, value=value, source=source))
        return rows


    label_rows = []
    for n in aadb_other:
        m = LABEL_RE.search(os.path.basename(n))
        if m and m.group("attr").lower() in KEEP_ATTRS:
            label_rows += parse_label_file(AADB_ZIP.read(n).decode("utf-8", errors="ignore"), m.group("split"),
                                           KEEP_ATTRS[m.group("attr").lower()], "HF archive")
    if not any(r["attribute"] == "ColorHarmony" for r in label_rows):
        for p in AADB_LOCAL_LABEL_DIR.rglob("*.txt"):
            m = LABEL_RE.search(p.name)
            if m and m.group("attr").lower() in KEEP_ATTRS:
                label_rows += parse_label_file(p.read_text(encoding="utf-8", errors="ignore"), m.group("split"),
                                               KEEP_ATTRS[m.group("attr").lower()], "local upload")
    if not any(r["attribute"] == "ColorHarmony" for r in label_rows):
        raise RuntimeError(
            "AADB Color Harmony labels were not found in the Hugging Face archive.\n"
            "Download the 'imgListFiles_label' folder of the official AADB release (linked from the README of "
            "https://github.com/aimerykong/deepImageAestheticsAnalysis), upload its .txt files to "
            f"{AADB_LOCAL_LABEL_DIR} and run this script again (downloads and caches are reused). "
            "Expected names: imgList<Split>Regression_ColorHarmony.txt")

    labels_long = pd.DataFrame(label_rows)
    print(labels_long.groupby(["attribute", "split", "source"]).size().to_string())
    n_conflicts = int((labels_long.groupby(["key", "attribute"])["value"].nunique() > 1).sum())
    print("Keys with conflicting duplicate labels (averaged):", n_conflicts)
    labels_wide = labels_long.pivot_table(index="key", columns="attribute", values="value", aggfunc="mean").reset_index()
    labels_wide.columns.name = None
    for col in ["ColorHarmony", "AestheticScore"]:
        if col not in labels_wide:
            labels_wide[col] = np.nan
    labels_wide["aadb_split"] = labels_wide["key"].map(labels_long[labels_long.attribute == "ColorHarmony"].groupby("key")["split"].first())


    def member_priority(n):
        low = n.lower()
        return 0 if "originalsize" in low else (2 if "warp" in low else 1)


    image_index = {}
    for n in aadb_images:
        k = aadb_key(n)
        if k not in image_index or member_priority(n) < member_priority(image_index[k]):
            image_index[k] = n
    labels_wide["member"] = labels_wide["key"].map(image_index)
    labels_wide["member_is_warped"] = labels_wide["member"].fillna("").str.lower().str.contains("warp")

    has_ch = labels_wide["ColorHarmony"].notna()
    aadb_pool = labels_wide[has_ch & labels_wide["member"].notna() & ~labels_wide["member_is_warped"]].copy()
    aadb_pool = aadb_pool.sort_values("key").reset_index(drop=True)
    save_table(pd.DataFrame([
        dict(step="Images with a Color Harmony label", n=int(has_ch.sum())),
        dict(step="Matched to an image file in the archive", n=int((has_ch & labels_wide["member"].notna()).sum())),
        dict(step="Excluded: only a warped 256x256 copy available", n=int((has_ch & labels_wide["member_is_warped"]).sum())),
        dict(step="Analyzable pool (original-size images)", n=len(aadb_pool)),
        dict(step="Keys with conflicting duplicate labels (averaged)", n=n_conflicts),
    ]), "TableS2_aadb_label_matching", folder=SUP_DIR)
    print(aadb_pool["ColorHarmony"].describe().to_string())
# ========================= END SECTION: aadb load =========================

# =========================== SECTION: aadb scene ===========================
# ---------------------------------------------------------------
# Study 1 - CLIP zero-shot scene filters and sample definition
# ---------------------------------------------------------------
if not FROM_FEATURES:   # needs the image data and/or the models
    def load_aadb_image(member, draft_side=None):
        img = Image.open(io.BytesIO(AADB_ZIP.read(member)))
        if draft_side and img.format == "JPEG":
            img.draft(img.mode, (draft_side, draft_side))
        return img


    def run_clip_pass(cache_path, prompts, columns, label):
        # columns: output column -> indices of the prompts whose probabilities are summed. Results are cached per image.
        done = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["key"] + list(columns))
        todo = aadb_pool[~aadb_pool["key"].isin(set(done["key"].astype(str)))]
        print(f"CLIP {label}: {len(done)} cached | {len(todo)} to classify")
        new_rows, b_keys, b_imgs = [], [], []

        def flush():
            if b_imgs:
                probs = clip_prompt_probabilities(b_imgs, prompts)
                for k, pr in zip(b_keys, probs):
                    new_rows.append(dict(key=k, **{c: float(pr[idx].sum()) for c, idx in columns.items()}))
            b_keys.clear()
            b_imgs.clear()

        for row in tqdm(todo.itertuples(index=False), total=len(todo), desc=f"CLIP {label}"):
            try:
                b_imgs.append(load_aadb_image(row.member, draft_side=448).convert("RGB"))
                b_keys.append(row.key)
            except Exception as exc:
                print("Unreadable image skipped:", row.member, exc)
            if len(b_imgs) == CONFIG["CLIP_BATCH"]:
                flush()
                if len(new_rows) % (CONFIG["CLIP_BATCH"] * 20) == 0:
                    pd.concat([done, pd.DataFrame(new_rows)]).to_csv(cache_path, index=False)
        flush()
        out = pd.concat([done, pd.DataFrame(new_rows, columns=["key"] + list(columns))]).drop_duplicates("key")
        for c in columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out.to_csv(cache_path, index=False)
        return out


    scene_broad = run_clip_pass(CACHE_DIR / "aadb_scene_filter.csv", INTERIOR_PROMPTS + NON_INTERIOR_PROMPTS,
                                {"p_interior": list(range(len(INTERIOR_PROMPTS)))}, "indoor filter")
    _ns, _np = len(SPACE_PROMPTS), len(PEOPLE_PROMPTS)
    scene_strict = run_clip_pass(CACHE_DIR / "aadb_scene_filter_strict.csv", SPACE_PROMPTS + PEOPLE_PROMPTS + OTHER_PROMPTS,
                                 {"p_space": list(range(_ns)), "p_people": list(range(_ns, _ns + _np))}, "strict interior filter")
    release_clip()   # frees GPU memory before MiDaS feature extraction

    aadb_pool = (aadb_pool.drop(columns=["p_interior", "p_space", "p_people"], errors="ignore")
                 .merge(scene_broad, on="key", how="left").merge(scene_strict, on="key", how="left"))
    aadb_pool["is_interior"] = aadb_pool["p_interior"] >= CONFIG["INTERIOR_PROB_THRESHOLD"]
    aadb_pool["is_interior_strict"] = aadb_pool["p_space"] >= CONFIG["STRICT_INTERIOR_THRESHOLD"]

    aadb_pool = attach_manual_labels(aadb_pool)
    _rng = np.random.default_rng(CONFIG["RANDOM_SEED"])
    general_keys = set(_rng.choice(aadb_pool["key"].to_numpy(), size=min(CONFIG["AADB_GENERAL_N"], len(aadb_pool)), replace=False))
    aadb_pool["in_general_sample"] = aadb_pool["key"].isin(general_keys)
    aadb_sel = aadb_pool[aadb_pool["is_interior_strict"] | aadb_pool["is_interior"] | aadb_pool["in_general_sample"]].copy()
    print(f"Interior (strict): {int(aadb_pool['is_interior_strict'].sum())} | Indoor (broad): {int(aadb_pool['is_interior'].sum())} "
          f"| strict within broad: {int((aadb_pool['is_interior_strict'] & aadb_pool['is_interior']).sum())} "
          f"| general sample: {int(aadb_pool['in_general_sample'].sum())} | images to process: {len(aadb_sel)}")
    if aadb_pool["is_interior_strict"].sum() < 150:
        print("WARNING: fewer than 150 strict interior images; estimates for this subset will have wide confidence intervals.")
    save_table(aadb_pool[["key", "member", "aadb_split", "ColorHarmony", "AestheticScore", "p_interior", "p_space", "p_people",
                          "is_interior", "is_interior_strict", "in_general_sample"]], "aadb_scene_filter_and_samples", folder=DAT_DIR, show=False)


    def _sheet(keys, caption_fn):
        items = []
        for k in keys:
            r = aadb_pool.loc[aadb_pool["key"] == k].iloc[0]
            items.append((make_thumb(np.asarray(load_aadb_image(r.member, 256).convert("RGB"))), caption_fn(r)))
        return items


    _sheet_rng = np.random.default_rng(CONFIG["RANDOM_SEED"] + 1)
    _broad = aadb_pool.loc[aadb_pool["is_interior"], "key"].to_numpy()
    _show = _sheet_rng.choice(_broad, size=min(48, len(_broad)), replace=False) if len(_broad) else []
    contact_sheet(_sheet(_show, lambda r: f"p={r.p_interior:.2f} CH={r.ColorHarmony:+.1f}"),
                  "AADB 'Indoor (broad)' subset (random 48; p = CLIP indoor probability, CH = human Color Harmony)",
                  "FigS1_aadb_indoor_broad_contact_sheet")
    for _ext in ("png", "pdf"):   # superseded file name of earlier runs
        (SUP_DIR / f"FigS1_aadb_interior_contact_sheet.{_ext}").unlink(missing_ok=True)
    _strict = aadb_pool.loc[aadb_pool["is_interior_strict"], "key"].to_numpy()
    _show = _sheet_rng.choice(_strict, size=min(48, len(_strict)), replace=False) if len(_strict) else []
    contact_sheet(_sheet(_show, lambda r: f"p={r.p_space:.2f} CH={r.ColorHarmony:+.1f}"),
                  "AADB 'Interior (strict)' subset (random 48; p = CLIP interior-space probability, CH = human Color Harmony)",
                  "FigS5_aadb_interior_strict_contact_sheet")

    # Manual precision audit: a numbered random sample of the strict subset and an Excel template to fill in
    _audit_rng = np.random.default_rng(CONFIG["RANDOM_SEED"] + 2)
    _audit_keys = list(_audit_rng.choice(_strict, size=min(CONFIG["AUDIT_N"], len(_strict)), replace=False)) if len(_strict) else []
    # Blind audit: the template shows neither the human ratings nor the CLIP probabilities
    audit = aadb_pool.set_index("key").loc[_audit_keys, ["member"]].reset_index()
    audit.insert(0, "audit_no", np.arange(1, len(audit) + 1))
    audit["is_interior_space_1_or_0"] = ""
    audit["note"] = ""
    _tmpl = SUP_DIR / "TableS5_strict_interior_audit_template.xlsx"
    _filled = False
    if _tmpl.exists():
        try:
            _filled = pd.to_numeric(pd.read_excel(_tmpl)["is_interior_space_1_or_0"], errors="coerce").notna().any()
        except Exception:
            _filled = False
    if _filled:
        print("A filled audit template already exists; it is kept unchanged.")
    else:
        save_table(audit, "TableS5_strict_interior_audit_template", folder=SUP_DIR, show=False)
    contact_sheet(_sheet(_audit_keys, lambda r: f"#{int(audit.loc[audit['key'] == r.key, 'audit_no'].iloc[0])}"),
                  "Strict interior subset: manual audit sample (numbers match TableS5_strict_interior_audit_template)",
                  "FigS6_strict_interior_audit_sheet", ncols=10)
# ========================= END SECTION: aadb scene =========================

# =========================== SECTION: aadb features ===========================
# ---------------------------------------------------------------
# Study 1 - per-image features (resumable: partial results are cached every 200 images)
# ---------------------------------------------------------------
if not FROM_FEATURES:   # needs the image data and/or the models
    # v2 cache: recomputed for the final sample definition so that the processing order (which defines the
    # shuffled-depth donors) is exactly reproducible by a single fresh run of this script
    feat_cache = CACHE_DIR / "aadb_features_v2_partial.csv"
    cached = read_features_csv(feat_cache) if feat_cache.exists() else pd.DataFrame()
    done_keys = set(cached["key"].astype(str)) if len(cached) else set()
    order = aadb_sel.sample(frac=1.0, random_state=CONFIG["RANDOM_SEED"]).reset_index(drop=True)
    todo = order[~order["key"].isin(done_keys)]
    print(f"AADB features: {len(done_keys)} cached | {len(todo)} to compute")

    rows, prev_clo, t0 = [], None, time.time()
    example_path = CACHE_DIR / "example_aadb_interior.png"
    for r in tqdm(todo.itertuples(index=False), total=len(todo), desc="AADB features"):
        try:
            img = Image.open(io.BytesIO(AADB_ZIP.read(r.member)))
            orig_w, orig_h = img.size
            if img.format == "JPEG":
                img.draft(img.mode, (CONFIG["WORK_MAX_SIDE"] * 2, CONFIG["WORK_MAX_SIDE"] * 2))
            img, cm = to_srgb_image(img)
            img = resize_max_side(img, CONFIG["WORK_MAX_SIDE"])
            rgb = np.asarray(img, dtype=np.uint8)
            clo = normalize_closeness(midas_disparity(rgb))
            feats = image_features(rgb, clo, image_rng(f"aadb|{r.key}"), donor_closeness=prev_clo)
            prev_clo = clo
            if r.is_interior_strict and not example_path.exists():
                Image.fromarray(rgb).save(example_path)
            Image.fromarray(make_thumb(rgb)).save(THUMB_DIR / f"aadb_{r.key}.jpg", quality=85)
            rows.append(dict(key=r.key, member=r.member, aadb_split=r.aadb_split, ColorHarmony=r.ColorHarmony,
                             AestheticScore=r.AestheticScore, p_interior=r.p_interior, p_space=r.p_space, p_people=r.p_people,
                             is_interior=bool(r.is_interior), is_interior_strict=bool(r.is_interior_strict),
                             in_general_sample=bool(r.in_general_sample), orig_width=orig_w, orig_height=orig_h,
                             work_width=rgb.shape[1], work_height=rgb.shape[0], had_icc=cm["had_icc"],
                             icc_converted=cm["icc_converted"], **feats))
        except Exception as exc:
            print("Skipped", r.key, "->", repr(exc)[:120])
        if rows and len(rows) % 200 == 0:
            pd.concat([cached, pd.DataFrame(rows)], ignore_index=True).to_csv(feat_cache, index=False)

    aadb_feat = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("key").reset_index(drop=True)
    aadb_feat.to_csv(feat_cache, index=False)
    print(f"Done: {len(aadb_feat)} images | {time.time() - t0:.0f} s")
else:
    aadb_feat = read_features_csv(DAT_DIR / "aadb_per_image_features.csv")
    aadb_feat["is_interior_strict"] = as_bool(aadb_feat["is_interior_strict"])
    aadb_pool = attach_manual_labels(aadb_feat.drop(columns=["interior_label", "is_interior_verified"], errors="ignore"))
    print(f"Loaded per-image AADB features: {len(aadb_feat)} images")
# ========================= END SECTION: aadb features =========================

# =========================== SECTION: aadb analysis ===========================
# ---------------------------------------------------------------
# Study 1 - statistical analyses
# ---------------------------------------------------------------
aadb_feat["is_interior"] = as_bool(aadb_feat["is_interior"])
aadb_feat["in_general_sample"] = as_bool(aadb_feat["in_general_sample"])
aadb_feat["is_interior_strict"] = as_bool(aadb_feat["is_interior_strict"])
aadb_feat = (aadb_feat.drop(columns=["interior_label", "is_interior_verified"], errors="ignore")
             .merge(aadb_pool[["key", "interior_label", "is_interior_verified"]], on="key", how="left"))
aadb_feat["is_interior_verified"] = aadb_feat["is_interior_verified"].fillna(False).astype(bool)
save_table(aadb_feat, "aadb_per_image_features", folder=DAT_DIR, show=False)

COVARIATES = ["colorfulness", "color_entropy_bits", "mean_L", "mean_C", "edge_density"]
LABELS = OrderedDict([
    ("H_depth", "Depth layers (primary)"), ("H_global", "Global palette (depth-free)"),
    ("H_bands", "Horizontal bands (depth-free)"), ("H_shuffled", "Shuffled donor depth (control)"),
    ("H_depth_adjw", "Depth layers, adjacency-weighted"), ("H_depth_areaw", "Depth layers, area-weighted"),
    ("colorfulness", "Colorfulness (Hasler–Süsstrunk)"), ("color_entropy_bits", "Color entropy"),
    ("mean_L", "Mean lightness L*"), ("mean_C", "Mean chroma C*ab"), ("edge_density", "Edge density"),
])
for k in CONFIG["K_SENSITIVITY"]:
    LABELS[f"H_depth_K{k}"] = f"Depth layers, K = {k}"
HARMONY_MAIN = ["H_depth", "H_global", "H_bands", "H_shuffled"]
SUBSETS = OrderedDict()
if aadb_feat["is_interior_verified"].sum() >= 10:
    SUBSETS["Interior (verified)"] = aadb_feat[aadb_feat["is_interior_verified"]]
SUBSETS["Interior (CLIP strict)"] = aadb_feat[aadb_feat["is_interior_strict"]]
SUBSETS["General"] = aadb_feat[aadb_feat["in_general_sample"]]
SUBSET_COLORS = dict(zip(SUBSETS, OKABE_ITO))

# ---- Table S6: manual annotation summary (CLIP precision and inter-rater agreement) ----
from sklearn.metrics import cohen_kappa_score
rows = []
_cand = aadb_pool[aadb_pool["is_interior_strict"]]
_lab = _cand["interior_label"].dropna()
if len(_lab):
    rows.append(dict(measure="Strict CLIP candidates", n=len(_cand), estimate=np.nan, ci_low=np.nan, ci_high=np.nan))
    rows.append(dict(measure="Candidates labeled by the first annotator", n=len(_lab), estimate=np.nan, ci_low=np.nan, ci_high=np.nan))
    for th in [CONFIG["STRICT_INTERIOR_THRESHOLD"], 0.80, 0.90, 0.95]:
        _l = _cand.loc[(_cand["p_space"] >= th) & _cand["interior_label"].notna(), "interior_label"]
        if len(_l):
            ci = stats.binomtest(int(_l.sum()), len(_l)).proportion_ci(confidence_level=0.95, method="wilson")
            rows.append(dict(measure=f"Precision of CLIP filter at p_space >= {th:.2f} (share labeled interior)", n=len(_l),
                             estimate=float(_l.mean()), ci_low=float(ci.low), ci_high=float(ci.high)))
    if SECOND_RATER_FILE.exists():
        _sr = pd.read_excel(SECOND_RATER_FILE, sheet_name="labels")
        _sr["label_2"] = pd.to_numeric(_sr["label"], errors="coerce")
        _m = _sr[["key", "label_2"]].merge(_cand[["key", "interior_label"]], on="key").dropna()
        if len(_m) >= 10:
            y1, y2 = _m["interior_label"].astype(int).to_numpy(), _m["label_2"].astype(int).to_numpy()
            kap = cohen_kappa_score(y1, y2)
            _r = np.random.default_rng(CONFIG["RANDOM_SEED"])
            boots = []
            for _ in range(CONFIG["N_BOOT"]):
                idx = _r.integers(0, len(y1), len(y1))
                if len(np.unique(np.concatenate([y1[idx], y2[idx]]))) > 1:
                    boots.append(cohen_kappa_score(y1[idx], y2[idx]))
            rows.append(dict(measure="Inter-rater agreement: percent agreement", n=len(_m), estimate=float((y1 == y2).mean()),
                             ci_low=np.nan, ci_high=np.nan))
            rows.append(dict(measure="Inter-rater agreement: Cohen's kappa (95% bootstrap CI)", n=len(_m), estimate=float(kap),
                             ci_low=float(np.nanpercentile(boots, 2.5)), ci_high=float(np.nanpercentile(boots, 97.5))))
    save_table(pd.DataFrame(rows), "TableS6_interior_annotation_summary", folder=SUP_DIR)

# ---- Table 2: descriptives ----
rows = []
for name, d in SUBSETS.items():
    ch = d["ColorHarmony"].to_numpy(float)
    row = OrderedDict(subset=name, n=len(d), color_harmony_mean=np.nanmean(ch), color_harmony_sd=np.nanstd(ch, ddof=1),
                      color_harmony_median=np.nanmedian(ch), share_negative=np.mean(ch < 0), share_zero=np.mean(ch == 0),
                      share_positive=np.mean(ch > 0), aesthetic_score_mean=d["AestheticScore"].mean())
    for v in HARMONY_MAIN + COVARIATES + ["depth_dispersion"]:
        row[f"{v}_mean"] = d[v].mean()
        row[f"{v}_sd"] = d[v].std()
    rows.append(row)
T2 = save_table(pd.DataFrame(rows), "Table2_aadb_descriptives")

# ---- Table 3: correlations with human ratings ----
rows = []
for name, d in SUBSETS.items():
    for outcome in ["ColorHarmony", "AestheticScore"]:
        if d[outcome].notna().sum() < 10:
            continue
        for v in HARMONY_MAIN + COVARIATES:
            res = spearman_ci(d[v], d[outcome])
            rows.append(OrderedDict(subset=name, outcome=outcome, predictor=v, label=LABELS[v], **res))
T3 = pd.DataFrame(rows)
T3["p_holm_within_subset_outcome"] = T3.groupby(["subset", "outcome"])["p"].transform(lambda p: holm(p.to_numpy()))
T3 = save_table(T3, "Table3_aadb_correlations")

# ---- Table 4: depth-aware vs control variants (dependent correlations) ----
rows = []
for name, d in SUBSETS.items():
    block = []
    for ctrl in ["H_global", "H_bands", "H_shuffled"]:
        res = delta_rho(d["ColorHarmony"], d["H_depth"], d[ctrl])
        block.append(OrderedDict(subset=name, comparison=f"H_depth vs {ctrl}", control=LABELS[ctrl], **res))
    adj = holm([b["p_mrr"] for b in block])
    for b, a in zip(block, adj):
        b["p_mrr_holm"] = a
    rows += block
T4 = save_table(pd.DataFrame(rows), "Table4_aadb_depth_vs_controls")

# ---- Table 5: partial correlations controlling for basic color statistics ----
rows = []
for name, d in SUBSETS.items():
    for v in ["H_depth", "H_global", "H_bands"]:
        res = partial_spearman(d["ColorHarmony"], d[v], d[COVARIATES].to_numpy(float))
        rows.append(OrderedDict(subset=name, predictor=v, label=LABELS[v], controls=", ".join(COVARIATES), **res))
T5 = save_table(pd.DataFrame(rows), "Table5_aadb_partial_correlations")

# ---- Table 6: cross-validated incremental validity ----
MODELS = OrderedDict([("Base", COVARIATES), ("Base + H_depth", COVARIATES + ["H_depth"]),
                      ("Base + H_global", COVARIATES + ["H_global"]), ("Base + H_bands", COVARIATES + ["H_bands"])])
COMPARISONS = [("Base + H_depth", "Base"), ("Base + H_depth", "Base + H_global"), ("Base + H_depth", "Base + H_bands")]
rows, CV_FOLDS = [], {}
for name, d in SUBSETS.items():
    folds, n_tr, n_te, n_used = incremental_validity(d, "ColorHarmony", COVARIATES, MODELS)
    CV_FOLDS[name] = folds
    block = []
    for m in MODELS:
        block.append(OrderedDict(subset=name, row_type="model", model_or_comparison=m, n=n_used,
                                 mean_r2=folds[m].mean(), sd_r2=folds[m].std(ddof=1)))
    tests = []
    for a, b in COMPARISONS:
        res = corrected_resampled_ttest(folds[a] - folds[b], n_tr, n_te)
        tests.append(OrderedDict(subset=name, row_type="comparison", model_or_comparison=f"{a} minus {b}", n=n_used, **res))
    for t_, a_ in zip(tests, holm([t["p"] for t in tests])):
        t_["p_holm"] = a_
    rows += block + tests
T6 = save_table(pd.DataFrame(rows), "Table6_aadb_incremental_validity")

# ---- Table 7: sensitivity (number of layers, pooling weights) ----
rows = []
for name, d in SUBSETS.items():
    for v in [f"H_depth_K{k}" for k in CONFIG["K_SENSITIVITY"]] + ["H_depth_adjw", "H_depth_areaw"]:
        rows.append(OrderedDict(subset=name, predictor=v, label=LABELS[v], **spearman_ci(d[v], d["ColorHarmony"])))
T7 = save_table(pd.DataFrame(rows), "Table7_aadb_sensitivity")

# ---- Table S3: quality control ----
rows = []
for name, d in SUBSETS.items():
    rows.append(OrderedDict(subset=name, n=len(d), share_icc_tagged=as_bool(d["had_icc"]).mean(),
                            share_icc_converted=as_bool(d["icc_converted"]).mean(),
                            share_3_valid_depth_layers=(d["n_valid_layers"] == CONFIG["K_LAYERS"]).mean(),
                            share_missing_H_depth=d["H_depth"].isna().mean(), share_missing_H_shuffled=d["H_shuffled"].isna().mean()))
save_table(pd.DataFrame(rows), "TableS3_aadb_quality_control", folder=SUP_DIR)
# ========================= END SECTION: aadb analysis =========================

# =========================== SECTION: aadb figures ===========================
# ---------------------------------------------------------------
# Study 1 - figures
# ---------------------------------------------------------------
def fmt_ci(res, key="rho"):
    return f"{res[key]:.2f} [{res['ci_low']:.2f}, {res['ci_high']:.2f}]"


# FigS4: distribution of human ratings
fig, axes = plt.subplots(1, len(SUBSETS), figsize=(3.3 * len(SUBSETS), 2.6), sharey=False)
for ax, (name, d) in zip(axes, SUBSETS.items()):
    vc = d["ColorHarmony"].round(1).value_counts().sort_index()
    ax.bar(vc.index, vc.values, width=0.15, color=OKABE_ITO[0])
    ax.set_title(f"{name} (n = {len(d)})")
    ax.set_xlabel("Human Color Harmony rating (AADB)")
    ax.set_ylabel("Number of images")
fig.tight_layout()
save_fig(fig, "FigS4_aadb_rating_distribution", folder=SUP_DIR)

# Fig2: H_depth by human rating level
fig, axes = plt.subplots(1, len(SUBSETS), figsize=(3.6 * len(SUBSETS), 3.1))
for ax, (name, d) in zip(axes, SUBSETS.items()):
    dd = d[["ColorHarmony", "H_depth"]].dropna().copy()
    dd["level"] = dd["ColorHarmony"].round(1)
    counts = dd["level"].value_counts()
    levels = sorted(counts[counts >= 5].index)
    data = [dd.loc[dd["level"] == lv, "H_depth"].to_numpy() for lv in levels]
    if data:
        bp = ax.boxplot(data, positions=range(len(levels)), widths=0.6, showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set(facecolor="#DCE9F5", edgecolor="#333333")
        for i, arr in enumerate(data):
            jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(arr))
            ax.scatter(np.full(len(arr), i) + jit, arr, s=3, alpha=0.35, color=OKABE_ITO[0], linewidths=0)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{lv:+.1f}" for lv in levels], rotation=45)
    res = spearman_ci(d["H_depth"], d["ColorHarmony"])
    ax.set_title(f"{name}: ρ = {fmt_ci(res)}, n = {res['n']}", fontsize=8.5)
    ax.set_xlabel("Human Color Harmony rating (levels with n >= 5)")
    ax.set_ylabel("Depth-aware harmony, H_depth")
fig.tight_layout()
save_fig(fig, "Fig2_Hdepth_by_human_rating")

# Fig3: forest plot of correlations with human ratings
sub = T3[T3["outcome"] == "ColorHarmony"]
order = HARMONY_MAIN + COVARIATES
fig, axes = plt.subplots(1, len(SUBSETS), figsize=(2.6 + 2.5 * len(SUBSETS), 3.8), sharey=True, sharex=True)
for ax, name in zip(axes, SUBSETS):
    s = sub[sub["subset"] == name].set_index("predictor").loc[order]
    y = np.arange(len(order))[::-1]
    colors = [OKABE_ITO[0] if v == "H_depth" else (OKABE_ITO[1] if v.startswith("H_") else "#777777") for v in order]
    ax.errorbar(s["rho"], y, xerr=[s["rho"] - s["ci_low"], s["ci_high"] - s["rho"]], fmt="none", ecolor="#555555", elinewidth=1)
    ax.scatter(s["rho"], y, c=colors, s=28, zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[v] for v in order])
    ax.set_title(f"{name} (n = {int(s['n'].max())})")
fig.supxlabel("Spearman ρ with human Color Harmony rating (95% bootstrap CI)", fontsize=9)
fig.tight_layout()
save_fig(fig, "Fig3_correlations_with_human_ratings")

# Fig4: depth-aware minus control differences
fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(T4) + 0.9))
labels_, ypos = [], []
for i, r in enumerate(T4.itertuples(index=False)):
    y = len(T4) - 1 - i
    color = SUBSET_COLORS[r.subset]
    ax.errorbar(r.delta_rho, y, xerr=[[r.delta_rho - r.ci_low], [r.ci_high - r.delta_rho]], fmt="o", color=color, ms=5, capsize=2)
    ax.text(1.02, y, f"p(Holm) = {r.p_mrr_holm:.3f}", va="center", fontsize=7, transform=ax.get_yaxis_transform(), clip_on=False)
    labels_.append(f"{r.subset}: H_depth − {r.comparison.replace('H_depth vs ', '')}")
    ypos.append(y)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(ypos)
ax.set_yticklabels(labels_)
ax.set_xlabel("Δρ with human Color Harmony rating (95% bootstrap CI)")
fig.tight_layout()
save_fig(fig, "Fig4_depth_vs_control_differences")

# Fig5: sensitivity to K and pooling weights
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True, gridspec_kw=dict(width_ratios=[1.6, 1]))
for j, name in enumerate(SUBSETS):
    s = T7[(T7["subset"] == name) & T7["predictor"].str.contains("_K")]
    ks = [int(p.split("_K")[-1]) for p in s["predictor"]]
    off = j - (len(SUBSETS) - 1) / 2.0
    axes[0].errorbar(np.array(ks) + off * 0.08, s["rho"], yerr=[s["rho"] - s["ci_low"], s["ci_high"] - s["rho"]],
                     fmt="-o", color=SUBSET_COLORS[name], label=name, capsize=2, ms=4)
    w = pd.concat([T3[(T3.subset == name) & (T3.outcome == "ColorHarmony") & (T3.predictor == "H_depth")],
                   T7[(T7.subset == name) & T7.predictor.isin(["H_depth_adjw", "H_depth_areaw"])]])
    x = np.arange(len(w)) + off * 0.15
    axes[1].errorbar(x, w["rho"], yerr=[w["rho"] - w["ci_low"], w["ci_high"] - w["rho"]], fmt="o",
                     color=SUBSET_COLORS[name], capsize=2, ms=4)
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_xlabel("Number of depth layers K")
axes[0].set_ylabel("Spearman ρ with human rating")
axes[0].set_xticks(CONFIG["K_SENSITIVITY"])
axes[0].legend(frameon=False)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_xticks(range(3))
axes[1].set_xticklabels(["Equal\n(primary)", "Adjacency", "Area"])
axes[1].set_xlabel("Pooling weights (K = 3)")
fig.tight_layout()
save_fig(fig, "Fig5_sensitivity_K_and_weighting")

# Fig6: incremental validity
comp = T6[T6["row_type"] == "comparison"].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(comp) + 0.9))
for i, r in enumerate(comp.itertuples(index=False)):
    y = len(comp) - 1 - i
    color = SUBSET_COLORS[r.subset]
    ax.errorbar(r.mean_diff, y, xerr=[[r.mean_diff - r.ci_low], [r.ci_high - r.mean_diff]], fmt="o", color=color, capsize=2, ms=5)
    ax.text(1.02, y, f"p(Holm) = {r.p_holm:.3f}", va="center", fontsize=7, transform=ax.get_yaxis_transform(), clip_on=False)


def _cmp_label(s):
    a, b = s.split(" minus ")
    return f"({a}) − ({b})"


ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(range(len(comp))[::-1])
ax.set_yticklabels([f"{r.subset}: {_cmp_label(r.model_or_comparison)}" for r in comp.itertuples(index=False)], fontsize=7)
ax.set_xlabel(f"ΔR² in {CONFIG['CV_SPLITS']}-fold × {CONFIG['CV_REPEATS']} repeated cross-validation (corrected resampled t, 95% CI)")
fig.tight_layout()
save_fig(fig, "Fig6_incremental_validity")
# ========================= END SECTION: aadb figures =========================

# =========================== SECTION: sun load ===========================
# ---------------------------------------------------------------
# Study 2 - SUN RGB-D download and scene index
# ---------------------------------------------------------------
if not FROM_FEATURES:   # needs the image data and/or the models
    import ssl
    import urllib.request
    import urllib.error
    from urllib.parse import urlparse

    # Public dataset servers whose HTTPS certificate is known to be expired. Only for these hosts the download falls back
    # to an unverified HTTPS connection (certificate check disabled for that host only). Every file inside a zip archive
    # carries a CRC-32 checksum that Python verifies when the file is read, so a corrupted download cannot pass silently.
    UNVERIFIED_SSL_HOSTS = {"rgbd.cs.princeton.edu"}


    def download_with_resume(url, dest, chunk=8 * 1024 * 1024, retries=30):
        # Resumable HTTP download (pure Python, works on Windows). Partial data are kept in <dest>.part.
        dest = Path(dest)
        part = dest.with_name(dest.name + ".part")
        context = None
        for attempt in range(1, retries + 1):
            have = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "Mozilla/5.0"}
            if have:
                headers["Range"] = f"bytes={have}-"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60, context=context) as resp:
                    if have and resp.status == 200:      # server ignored the range request: start again
                        have = 0
                    length = resp.headers.get("Content-Length")
                    total = int(length) + have if length else None
                    done, last = have, 0.0
                    with open(part, "ab" if have else "wb") as fh:
                        while True:
                            buf = resp.read(chunk)
                            if not buf:
                                break
                            fh.write(buf)
                            done += len(buf)
                            if time.time() - last > 15:
                                pct = f"{100 * done / total:.1f}%" if total else "?"
                                print(f"  downloaded {done / 1e9:.2f} GB ({pct})")
                                last = time.time()
                if total is None or done >= total:
                    part.replace(dest)
                    return dest
            except urllib.error.HTTPError as exc:
                if exc.code == 416 and have:              # nothing left to download
                    part.replace(dest)
                    return dest
                print(f"  HTTP error {exc.code} (attempt {attempt}/{retries}); retrying in 15 s")
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", None)
                if isinstance(reason, ssl.SSLCertVerificationError):
                    host = urlparse(url).hostname
                    if context is None and host in UNVERIFIED_SSL_HOSTS:
                        print(f"  NOTE: the HTTPS certificate of {host} failed verification ({reason.verify_message}).")
                        print("  This public dataset server has an expired certificate; continuing without certificate")
                        print("  verification for this host only. Zip checksums still protect the data against corruption.")
                        context = ssl._create_unverified_context()
                        continue
                    raise RuntimeError(f"HTTPS certificate of {host} could not be verified: {reason}. "
                                       "Download the file manually (see the message below).") from exc
                print(f"  connection problem: {exc!r} (attempt {attempt}/{retries}); retrying in 15 s")
            except (TimeoutError, ConnectionError, OSError) as exc:
                print(f"  connection problem: {exc!r} (attempt {attempt}/{retries}); retrying in 15 s")
            time.sleep(15)
        raise RuntimeError(f"Download failed after {retries} attempts: {url}")

    SUN_ZIP_PATH = DATA_ROOT / "SUNRGBD.zip"
    if not SUN_ZIP_PATH.exists():
        print("Downloading SUN RGB-D V1 (about 6.5 GB; resumable) ...")
        try:
            download_with_resume(CONFIG["SUN_URL"], SUN_ZIP_PATH)
        except Exception:
            print("\nSUN RGB-D could not be downloaded automatically. Manual alternative: open", CONFIG["SUN_URL"])
            print("in a web browser, save the file as", SUN_ZIP_PATH, "and run this script again.")
            raise
    try:
        SUN_ZIP = zipfile.ZipFile(SUN_ZIP_PATH)
    except zipfile.BadZipFile:
        raise RuntimeError(f"{SUN_ZIP_PATH} is incomplete or corrupted. Delete it and run the script again.")

    sun_scenes = {}
    for n in SUN_ZIP.namelist():
        if n.endswith("/"):
            continue
        for marker, field in (("/image/", "image"), ("/depth_bfx/", "depth_bfx"), ("/depth/", "depth")):
            if marker in n:
                sun_scenes.setdefault(n.split(marker)[0], {}).setdefault(field, []).append(n)
                break
        else:
            if n.endswith("/scene.txt"):
                sun_scenes.setdefault(n[: -len("/scene.txt")], {})["scene"] = n

    valid_prefixes = sorted(p for p, v in sun_scenes.items()
                            if any(m.lower().endswith((".jpg", ".jpeg", ".png")) for m in v.get("image", []))
                            and any(m.lower().endswith(".png") for m in v.get("depth_bfx", []) + v.get("depth", [])))
    sun_prefixes = list(np.random.default_rng(CONFIG["RANDOM_SEED"]).permutation(valid_prefixes)[: CONFIG["N_SUN"]])
    print(f"SUN RGB-D scenes with RGB + depth: {len(valid_prefixes)} | sampled: {len(sun_prefixes)}")


    def sun_sensor(prefix):
        parts = prefix.split("/")
        return next((p for p in parts if p in ("kv1", "kv2", "realsense", "xtion")), "unknown")
# ========================= END SECTION: sun load =========================

# =========================== SECTION: sun features ===========================
# ---------------------------------------------------------------
# Study 2 - measured vs MiDaS depth, and MiDaS-based features for Study 3
# ---------------------------------------------------------------
sun_cache = CACHE_DIR / "sun_features.csv"
K = CONFIG["K_LAYERS"]
if FROM_FEATURES:
    sun_feat = read_features_csv(DAT_DIR / "sun_per_image_features.csv")
    print("Loaded per-image SUN RGB-D features:", len(sun_feat))
elif sun_cache.exists():
    sun_feat = read_features_csv(sun_cache)
    print("Loaded cached SUN RGB-D features:", len(sun_feat))
else:
    rows, prev_clo, t0 = [], None, time.time()
    for pre in tqdm(sun_prefixes, desc="SUN RGB-D"):
        try:
            v = sun_scenes[pre]
            img_member = sorted(m for m in v["image"] if m.lower().endswith((".jpg", ".jpeg", ".png")))[0]
            dep_member = sorted(m for m in (v.get("depth_bfx") or v.get("depth")) if m.lower().endswith(".png"))[0]
            scene_label = SUN_ZIP.read(v["scene"]).decode("utf-8", errors="ignore").strip().lower() if "scene" in v else "unknown"
            img = Image.open(io.BytesIO(SUN_ZIP.read(img_member)))
            orig_w, orig_h = img.size
            img, cm = to_srgb_image(img)
            img = resize_max_side(img, CONFIG["WORK_MAX_SIDE"])
            rgb = np.asarray(img, dtype=np.uint8)
            raw = cv2.imdecode(np.frombuffer(SUN_ZIP.read(dep_member), np.uint8), cv2.IMREAD_UNCHANGED)
            meters = cv2.resize(decode_sun_depth(raw), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            disp_meas = np.where(np.isfinite(meters), 1.0 / meters, np.nan).astype(np.float32)
            disp_mid = midas_disparity(rgb)
            valid = np.isfinite(disp_meas)
            clo_meas = normalize_closeness(disp_meas)
            clo_mid_masked = normalize_closeness(np.where(valid, disp_mid, np.nan))
            clo_mid_full = normalize_closeness(disp_mid)

            idx = np.flatnonzero(valid.ravel())
            s_rng = image_rng(f"sun|{pre}|pixels")
            if idx.size > CONFIG["PIXEL_AGREEMENT_SAMPLE"]:
                idx = s_rng.choice(idx, CONFIG["PIXEL_AGREEMENT_SAMPLE"], replace=False)
            pix_rho = _spearman(disp_meas.ravel()[idx], disp_mid.ravel()[idx]) if idx.size > 100 else np.nan

            lay_meas = depth_layers(clo_meas, K, image_rng(f"sun|{pre}|layers"))
            lay_mid = depth_layers(clo_mid_masked, K, image_rng(f"sun|{pre}|layers"))
            bands = band_layers(rgb.shape[:2], K)
            both = np.flatnonzero(((lay_meas >= 0) & (lay_mid >= 0)).ravel())
            if both.size > CONFIG["PIXEL_AGREEMENT_SAMPLE"]:
                both = s_rng.choice(both, CONFIG["PIXEL_AGREEMENT_SAMPLE"], replace=False)
            ari_mid = adjusted_rand_score(lay_meas.ravel()[both], lay_mid.ravel()[both]) if both.size > 100 else np.nan
            ari_bands = adjusted_rand_score(lay_meas.ravel()[both], bands.ravel()[both]) if both.size > 100 else np.nan

            lab = rgb_to_lab(rgb)
            h_meas = pooled_harmony(*region_colors(lab, lay_meas, K, image_rng(f"sun|{pre}|color")), "equal")
            h_mid = pooled_harmony(*region_colors(lab, lay_mid, K, image_rng(f"sun|{pre}|color")), "equal")

            feats = image_features(rgb, clo_mid_full, image_rng(f"sun|{pre}|full"), donor_closeness=prev_clo)
            prev_clo = clo_mid_full
            if not (CACHE_DIR / "example_sun.png").exists():
                Image.fromarray(rgb).save(CACHE_DIR / "example_sun.png")
            safe = re.sub(r"[^A-Za-z0-9]+", "_", pre)[-80:]
            Image.fromarray(make_thumb(rgb)).save(THUMB_DIR / f"sun_{safe}.jpg", quality=85)
            rows.append(dict(scene_prefix=pre, sensor=sun_sensor(pre), scene_label=scene_label, rgb_member=img_member,
                             depth_member=dep_member, orig_width=orig_w, orig_height=orig_h, had_icc=cm["had_icc"],
                             icc_converted=cm["icc_converted"], measured_valid_fraction=float(valid.mean()),
                             pixel_spearman_measured_vs_midas=pix_rho, ari_midas_vs_measured=ari_mid,
                             ari_bands_vs_measured=ari_bands, H_depth_measured=h_meas, H_depth_midas_same_pixels=h_mid,
                             **feats))
        except Exception as exc:
            print("Skipped", pre, "->", repr(exc)[:120])
    sun_feat = pd.DataFrame(rows)
    sun_feat.to_csv(sun_cache, index=False)
    print(f"Done: {len(sun_feat)} scenes | {time.time() - t0:.0f} s")
save_table(sun_feat, "sun_per_image_features", folder=DAT_DIR, show=False)
# ========================= END SECTION: sun features =========================

# =========================== SECTION: sun analysis ===========================
# ---------------------------------------------------------------
# Study 2 - agreement analyses and figures
# ---------------------------------------------------------------
def iqr_text(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return float(q2), float(q1), float(q3)


def matched_rank_biserial(a, b):
    a, b = _complete(a, b)
    d = a - b
    d = d[d != 0]
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


rows = []
med, q1, q3 = iqr_text(sun_feat["pixel_spearman_measured_vs_midas"])
rows.append(dict(level="Pixel", measure="Spearman rho, measured vs MiDaS disparity (per image)", n=int(sun_feat["pixel_spearman_measured_vs_midas"].notna().sum()),
                 estimate=med, q1_or_ci_low=q1, q3_or_ci_high=q3, statistic=np.nan, p=np.nan, note="median [IQR]"))
for col, lab in [("ari_midas_vs_measured", "ARI, MiDaS layers vs measured layers"), ("ari_bands_vs_measured", "ARI, horizontal bands vs measured layers")]:
    med, q1, q3 = iqr_text(sun_feat[col])
    rows.append(dict(level="Layer", measure=lab, n=int(sun_feat[col].notna().sum()), estimate=med, q1_or_ci_low=q1,
                     q3_or_ci_high=q3, statistic=np.nan, p=np.nan, note="median [IQR]"))
a, b = _complete(sun_feat["ari_midas_vs_measured"], sun_feat["ari_bands_vs_measured"])
w = stats.wilcoxon(a, b)
rows.append(dict(level="Layer", measure="ARI(MiDaS) vs ARI(bands), Wilcoxon signed-rank", n=len(a), estimate=matched_rank_biserial(a, b),
                 q1_or_ci_low=np.nan, q3_or_ci_high=np.nan, statistic=float(w.statistic), p=float(w.pvalue),
                 note="estimate = matched-pairs rank-biserial r"))
res = spearman_ci(sun_feat["H_depth_measured"], sun_feat["H_depth_midas_same_pixels"])
rows.append(dict(level="Metric", measure="Spearman rho, H_depth measured vs MiDaS (same pixels)", n=res["n"], estimate=res["rho"],
                 q1_or_ci_low=res["ci_low"], q3_or_ci_high=res["ci_high"], statistic=np.nan, p=res["p"], note="95% bootstrap CI"))
ccc = lin_ccc(sun_feat["H_depth_measured"], sun_feat["H_depth_midas_same_pixels"])
lo, hi = bootstrap_stat(lin_ccc, sun_feat["H_depth_measured"], sun_feat["H_depth_midas_same_pixels"])
rows.append(dict(level="Metric", measure="Lin's CCC, H_depth measured vs MiDaS (same pixels)", n=res["n"], estimate=ccc,
                 q1_or_ci_low=lo, q3_or_ci_high=hi, statistic=np.nan, p=np.nan, note="95% bootstrap CI"))
ccc_full = lin_ccc(sun_feat["H_depth_measured"], sun_feat["H_depth"])
lo, hi = bootstrap_stat(lin_ccc, sun_feat["H_depth_measured"], sun_feat["H_depth"])
rows.append(dict(level="Metric", measure="Lin's CCC, H_depth measured vs MiDaS (all pixels, as used in Studies 1 and 3)", n=res["n"],
                 estimate=ccc_full, q1_or_ci_low=lo, q3_or_ci_high=hi, statistic=np.nan, p=np.nan, note="95% bootstrap CI"))
ba = bland_altman(sun_feat["H_depth_measured"], sun_feat["H_depth_midas_same_pixels"])
rows.append(dict(level="Metric", measure="Bland-Altman mean difference (MiDaS minus measured)", n=ba["n"], estimate=ba["mean_diff"],
                 q1_or_ci_low=ba["loa_low"], q3_or_ci_high=ba["loa_high"], statistic=np.nan, p=np.nan, note="interval = 95% limits of agreement"))
T8 = save_table(pd.DataFrame(rows), "Table8_sun_depth_agreement")

rows = []
for sensor, d in sun_feat.groupby("sensor"):
    if len(d) < 10:
        continue
    rows.append(dict(sensor=sensor, n=len(d), median_pixel_spearman=d["pixel_spearman_measured_vs_midas"].median(),
                     median_ari_midas=d["ari_midas_vs_measured"].median(), median_ari_bands=d["ari_bands_vs_measured"].median(),
                     ccc_H_depth=lin_ccc(d["H_depth_measured"], d["H_depth_midas_same_pixels"])))
save_table(pd.DataFrame(rows), "TableS4_sun_agreement_by_sensor", folder=SUP_DIR)

# Fig7: pixel and layer agreement
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
x = sun_feat["pixel_spearman_measured_vs_midas"].dropna()
axes[0].hist(x, bins=30, color=OKABE_ITO[0], alpha=0.85)
axes[0].axvline(x.median(), color="black", ls="--", lw=1, label=f"median = {x.median():.2f}")
axes[0].set_xlabel("Per-image Spearman ρ\n(measured vs MiDaS disparity)")
axes[0].set_ylabel("Number of scenes")
axes[0].legend(frameon=False)
bins = np.linspace(-0.2, 1.0, 31)
for col, lab, c in [("ari_midas_vs_measured", "MiDaS layers", OKABE_ITO[0]), ("ari_bands_vs_measured", "Horizontal bands (depth-free)", OKABE_ITO[1])]:
    v = sun_feat[col].dropna()
    axes[1].hist(v, bins=bins, alpha=0.6, color=c, label=f"{lab} (median = {v.median():.2f})")
axes[1].set_xlabel("Adjusted Rand Index with measured-depth layers")
axes[1].set_ylabel("Number of scenes")
axes[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
save_fig(fig, "Fig7_sun_depth_agreement")

# Fig8: agreement of the harmony metric
m, e = _complete(sun_feat["H_depth_measured"], sun_feat["H_depth_midas_same_pixels"])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
axes[0].scatter(m, e, s=9, alpha=0.6, color=OKABE_ITO[0], linewidths=0)
lims = [min(m.min(), e.min()) - 0.02, max(m.max(), e.max()) + 0.02]
axes[0].plot(lims, lims, color="black", lw=0.8, ls="--")
axes[0].set_xlim(lims)
axes[0].set_ylim(lims)
axes[0].set_xlabel("H_depth from measured depth")
axes[0].set_ylabel("H_depth from MiDaS depth")
axes[0].set_title(f"CCC = {ccc:.2f}, ρ = {res['rho']:.2f} (n = {len(m)})", fontsize=8.5)
mean_, diff_ = (m + e) / 2, e - m
axes[1].scatter(mean_, diff_, s=9, alpha=0.6, color=OKABE_ITO[0], linewidths=0)
for yv, ls in [(ba["mean_diff"], "-"), (ba["loa_low"], "--"), (ba["loa_high"], "--")]:
    axes[1].axhline(yv, color="black", lw=0.8, ls=ls)
axes[1].set_xlabel("Mean of the two H_depth values")
axes[1].set_ylabel("Difference (MiDaS minus measured)")
axes[1].set_title(f"Bias = {ba['mean_diff']:.3f}; 95% LoA [{ba['loa_low']:.3f}, {ba['loa_high']:.3f}]", fontsize=8.5)
fig.tight_layout()
save_fig(fig, "Fig8_sun_harmony_agreement")
# ========================= END SECTION: sun analysis =========================

# =========================== SECTION: style features ===========================
# ---------------------------------------------------------------
# Study 3 - Interior-Style corpus (MiDaS depth, identical processing)
# ---------------------------------------------------------------
style_cache = CACHE_DIR / "style_features.csv"


def parse_style(text):
    m = re.search(r"of\s+(.+?)\s+style", str(text), flags=re.IGNORECASE)
    return m.group(1).strip().title() if m else (str(text).strip().title() or "Unknown")


if FROM_FEATURES:
    style_feat = read_features_csv(DAT_DIR / "style_per_image_features.csv")
    print("Loaded per-image Interior-Style features:", len(style_feat))
elif style_cache.exists():
    style_feat = read_features_csv(style_cache)
    print("Loaded cached Interior-Style features:", len(style_feat))
else:
    from datasets import load_dataset
    # The shuffled stream is deterministic, so after a network error it is reopened and the items already
    # handled are skipped: the final sample is identical to an uninterrupted run.
    rows, prev_clo, last_done, t0 = [], None, 0, time.time()
    pbar = tqdm(total=CONFIG["N_STYLE"], desc="Interior-Style")
    for attempt in range(1, 7):
      try:
        stream = load_dataset(CONFIG["STYLE_REPO"], split="train", streaming=True)
        stream = stream.shuffle(seed=CONFIG["RANDOM_SEED"], buffer_size=CONFIG["STYLE_SHUFFLE_BUFFER"])
        feat_names = getattr(stream, "features", None) or {}
        position = 0
        for ex in stream:
            if len(rows) >= CONFIG["N_STYLE"]:
                break
            position += 1
            if position <= last_done:
                continue
            last_done = position
            img = next((v for v in ex.values() if isinstance(v, Image.Image)), None)
            if img is None:
                continue
            label = ""
            for k in ("text", "label", "style", "caption", "category"):
                if k in ex and ex[k] is not None:
                    val = ex[k]
                    if isinstance(val, (int, np.integer)) and k in feat_names and hasattr(feat_names[k], "names"):
                        val = feat_names[k].names[int(val)]
                    label = str(val)
                    break
            try:
                image_id = hashlib.md5(np.asarray(img.convert("RGB")).tobytes()).hexdigest()[:16]
                orig_w, orig_h = img.size
                img, cm = to_srgb_image(img)
                w, h = img.size
                img = img.crop((0, 0, w, int(round(h * (1.0 - CONFIG["WATERMARK_CROP_BOTTOM"])))))
                img = resize_max_side(img, CONFIG["WORK_MAX_SIDE"])
                rgb = np.asarray(img, dtype=np.uint8)
                clo = normalize_closeness(midas_disparity(rgb))
                feats = image_features(rgb, clo, image_rng(f"style|{image_id}"), donor_closeness=prev_clo)
                prev_clo = clo
                if not (CACHE_DIR / "example_style.png").exists():
                    Image.fromarray(rgb).save(CACHE_DIR / "example_style.png")
                Image.fromarray(make_thumb(rgb)).save(THUMB_DIR / f"style_{image_id}.jpg", quality=85)
                rows.append(dict(image_id=image_id, stream_position=position, raw_label=label[:120], style_label=parse_style(label),
                                 orig_width=orig_w, orig_height=orig_h, had_icc=cm["had_icc"], icc_converted=cm["icc_converted"], **feats))
                pbar.update(1)
            except Exception as exc:
                print("Skipped stream item", position, "->", repr(exc)[:120])
        break
      except Exception as exc:
        if attempt == 6:
            raise
        print(f"  Interior-Style stream interrupted ({type(exc).__name__}: {str(exc)[:100]}); "
              f"reopening (attempt {attempt}/6) and continuing after item {last_done}")
        time.sleep(30)
    pbar.close()
    style_feat = pd.DataFrame(rows)
    style_feat.to_csv(style_cache, index=False)
    print(f"Done: {len(style_feat)} images | {time.time() - t0:.0f} s")
save_table(style_feat, "style_per_image_features", folder=DAT_DIR, show=False)
print(style_feat["style_label"].value_counts().to_string())
# ========================= END SECTION: style features =========================

# =========================== SECTION: corpus analysis ===========================
# ---------------------------------------------------------------
# Study 3 - within-corpus analyses and figures
# ---------------------------------------------------------------
CORPUS_VARS = ["H_depth", "colorfulness", "color_entropy_bits", "depth_dispersion", "edge_density", "mean_L", "mean_C"]
sun_c = sun_feat.assign(corpus="SUN RGB-D", category=sun_feat["scene_label"].astype(str))
sty_c = style_feat.assign(corpus="Interior-Style", category=style_feat["style_label"].astype(str))
corpus_df = pd.concat([sun_c[["corpus", "category"] + CORPUS_VARS], sty_c[["corpus", "category"] + CORPUS_VARS]], ignore_index=True)
CORPORA = ["SUN RGB-D", "Interior-Style"]

# Table 9: descriptives by corpus
rows = []
for c in CORPORA:
    d = corpus_df[corpus_df["corpus"] == c]
    for v in CORPUS_VARS:
        x = d[v].dropna()
        rows.append(dict(corpus=c, variable=v, n=len(x), mean=x.mean(), sd=x.std(), median=x.median(),
                         q1=x.quantile(0.25), q3=x.quantile(0.75)))
T9 = save_table(pd.DataFrame(rows), "Table9_corpus_descriptives")

# Table 10: within-corpus correlations (plus the pooled estimate for transparency)
PAIRS = [("colorfulness", "depth_dispersion"), ("H_depth", "depth_dispersion"), ("H_depth", "colorfulness")]
rows, within = [], {}
for x_, y_ in PAIRS:
    for c in CORPORA + ["Pooled (not interpretable: mixes corpora)"]:
        d = corpus_df if c.startswith("Pooled") else corpus_df[corpus_df["corpus"] == c]
        res = spearman_ci(d[x_], d[y_])
        within[(x_, y_, c)] = res
        rows.append(OrderedDict(variable_1=x_, variable_2=y_, sample=c, **res))
    r1, r2 = within[(x_, y_, CORPORA[0])], within[(x_, y_, CORPORA[1])]
    z, p = compare_independent_spearman(r1["rho"], r1["n"], r2["rho"], r2["n"])
    rows.append(OrderedDict(variable_1=x_, variable_2=y_, sample="Difference between corpora (Fisher z)", n=r1["n"] + r2["n"],
                            rho=r1["rho"] - r2["rho"], ci_low=np.nan, ci_high=np.nan, p=p))
T10 = save_table(pd.DataFrame(rows), "Table10_within_corpus_correlations")

# Table 11: interaction model, colorfulness ~ depth dispersion x corpus (HC3 SEs)
d = corpus_df.dropna(subset=["colorfulness", "depth_dispersion"]).copy()
zc = (d["colorfulness"] - d["colorfulness"].mean()) / d["colorfulness"].std()
zd = (d["depth_dispersion"] - d["depth_dispersion"].mean()) / d["depth_dispersion"].std()
grp = (d["corpus"] == "Interior-Style").astype(float)
T11, r2_int, n_int = ols_hc3(zc, np.column_stack([zd, grp, zd * grp]),
                             ["z_depth_dispersion", "corpus_InteriorStyle", "z_depth_dispersion x corpus_InteriorStyle"])
T11["model_r2"] = r2_int
T11["n"] = n_int
T11 = save_table(T11, "Table11_interaction_model")

# Table 12: descriptive corpus differences (dataset-level; not a design-quality effect)
rows = []
for v in ["H_depth", "colorfulness", "depth_dispersion"]:
    a = corpus_df.loc[corpus_df["corpus"] == CORPORA[1], v]
    b = corpus_df.loc[corpus_df["corpus"] == CORPORA[0], v]
    rows.append(OrderedDict(variable=v, comparison=f"{CORPORA[1]} vs {CORPORA[0]}", **mann_whitney_summary(a, b),
                            interpretation="Descriptive dataset-level difference; confounded with acquisition and content"))
T12 = save_table(pd.DataFrame(rows), "Table12_corpus_differences")

# Table 13: category effects tested within each corpus only
EXCLUDE = {"idk", "unknown", "", "nan"}
rows, CAT_DATA = [], {}
for c in CORPORA:
    d = corpus_df[(corpus_df["corpus"] == c) & ~corpus_df["category"].str.lower().isin(EXCLUDE)]
    counts = d["category"].value_counts()
    keep = counts[counts >= CONFIG["MIN_CATEGORY_N"]].index.tolist()
    d = d[d["category"].isin(keep)]
    CAT_DATA[c] = d
    for v in ["colorfulness", "H_depth"]:
        if len(keep) < 2:
            rows.append(dict(corpus=c, variable=v, k=len(keep), n=len(d), H=np.nan, df=np.nan, p=np.nan, epsilon2=np.nan,
                             categories="fewer than two categories with n >= 10"))
            continue
        res = kruskal_epsilon2([d.loc[d["category"] == k, v].to_numpy() for k in keep])
        rows.append(dict(corpus=c, variable=v, **res, categories="; ".join(f"{k} (n={counts[k]})" for k in keep)))
T13 = save_table(pd.DataFrame(rows), "Table13_within_corpus_categories")

# Fig9: within-corpus relationship colorfulness vs depth dispersion
fig, ax = plt.subplots(figsize=(4.8, 3.6))
for c, col in zip(CORPORA, [OKABE_ITO[0], OKABE_ITO[1]]):
    d = corpus_df[corpus_df["corpus"] == c].dropna(subset=["colorfulness", "depth_dispersion"])
    ax.scatter(d["depth_dispersion"], d["colorfulness"], s=9, alpha=0.5, color=col, linewidths=0)
    b1, b0 = np.polyfit(d["depth_dispersion"], d["colorfulness"], 1)
    xs = np.linspace(d["depth_dispersion"].min(), d["depth_dispersion"].max(), 50)
    r_ = within[("colorfulness", "depth_dispersion", c)]
    ax.plot(xs, b0 + b1 * xs, color=col, lw=1.6, label=f"{c}: ρ = {r_['rho']:.2f} [{r_['ci_low']:.2f}, {r_['ci_high']:.2f}]")
ax.set_xlabel("Depth dispersion (variance of normalized MiDaS depth)")
ax.set_ylabel("Colorfulness (Hasler–Süsstrunk)")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
save_fig(fig, "Fig9_within_corpus_colorfulness_depth")

# Fig10: distributions by corpus (descriptive)
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
for ax, v, lab in zip(axes, ["H_depth", "colorfulness", "depth_dispersion"], ["H_depth", "Colorfulness", "Depth dispersion"]):
    data = [corpus_df.loc[corpus_df["corpus"] == c, v].dropna().to_numpy() for c in CORPORA]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for body, col in zip(parts["bodies"], [OKABE_ITO[0], OKABE_ITO[1]]):
        body.set_facecolor(col)
        body.set_alpha(0.55)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["SUN RGB-D", "Interior-\nStyle"])
    ax.set_ylabel(lab)
fig.suptitle("Descriptive corpus distributions (not a design-quality comparison)", fontsize=9)
fig.tight_layout()
save_fig(fig, "Fig10_corpus_distributions")

# Fig11: within-corpus category distributions
fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6))
for j, c in enumerate(CORPORA):
    d = CAT_DATA[c]
    cats = d.groupby("category")["colorfulness"].median().sort_values().index.tolist()
    for i, v in enumerate(["colorfulness", "H_depth"]):
        ax = axes[i, j]
        if len(cats) >= 1:
            ax.boxplot([d.loc[d["category"] == k, v].dropna().to_numpy() for k in cats], vert=False, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor="#DCE9F5"), medianprops=dict(color="black"))
            ax.set_yticks(range(1, len(cats) + 1))
            ax.set_yticklabels([f"{k} (n={int((d['category'] == k).sum())})" for k in cats], fontsize=7)
        tr = T13[(T13["corpus"] == c) & (T13["variable"] == v)].iloc[0]
        title = f"{c}: {v}" + (f"\nKruskal–Wallis H = {tr['H']:.1f}, p = {tr['p']:.3g}, ε² = {tr['epsilon2']:.2f}" if np.isfinite(tr["H"]) else "")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Colorfulness" if v == "colorfulness" else "H_depth")
fig.tight_layout()
save_fig(fig, "Fig11_within_corpus_categories")

# Supplementary contact sheets of both corpora (need the thumbnails of the full pipeline)
if not FROM_FEATURES:
    def _thumbs(prefix, ids, captions):
        items = []
        for i, cap in zip(ids, captions):
            p = THUMB_DIR / f"{prefix}_{i}.jpg"
            if p.exists():
                items.append((np.asarray(Image.open(p)), cap))
        return items


    _sun_ids = [re.sub(r"[^A-Za-z0-9]+", "_", p)[-80:] for p in sun_feat["scene_prefix"].head(48)]
    contact_sheet(_thumbs("sun", _sun_ids, sun_feat["scene_label"].head(48)), "SUN RGB-D scenes (first 48 of the random sample)", "FigS2_sun_contact_sheet")
    contact_sheet(_thumbs("style", style_feat["image_id"].head(48), style_feat["style_label"].head(48)),
                  "Interior-Style images (first 48 of the shuffled stream, bottom 8% cropped)", "FigS3_style_contact_sheet")
# ========================= END SECTION: corpus analysis =========================

# =========================== SECTION: additional analyses ===========================
# ---------------------------------------------------------------
# Additional analyses (revision): smallest-effect test of H1, determinants of H_depth, comparison of H_depth with
# the image statistics, sample overlaps, color-content baselines for Study 2, multiplicity of the category tests
# ---------------------------------------------------------------
def one_sided_below(r, n, bound):
    # One-sided test of H0: rho >= bound (Fisher z with the variance of Fieller, Hartley & Pearson, 1957).
    se = np.sqrt(1.06 / (n - 3))
    z = (np.arctanh(r) - np.arctanh(bound)) / se
    return float(z), float(stats.norm.cdf(z))


def paired_ccc_difference(ref, a, b, n_boot=None, seed=None):
    # CCC(ref, a) - CCC(ref, b) with a paired percentile bootstrap CI.
    ref, a, b = _complete(ref, a, b)
    n_boot = n_boot or CONFIG["N_BOOT"]
    rng = np.random.default_rng(CONFIG["RANDOM_SEED"] if seed is None else seed)
    d = [lin_ccc(ref[i], a[i]) - lin_ccc(ref[i], b[i]) for i in (rng.integers(0, len(ref), len(ref)) for _ in range(n_boot))]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return len(ref), lin_ccc(ref, a) - lin_ccc(ref, b), float(lo), float(hi)


rows = []


def add_row(study, analysis, sample, measure, n, estimate, ci_low=np.nan, ci_high=np.nan, statistic=np.nan, p=np.nan,
            note=""):
    rows.append(OrderedDict(study=study, analysis=analysis, sample=sample, measure=measure, n=n, estimate=estimate,
                            ci_low=ci_low, ci_high=ci_high, statistic=statistic, p=p, p_holm=np.nan, note=note))


# Study 1a: one-sided test that the correlation of H_depth with the ratings is smaller than the SESOI
for name, d in SUBSETS.items():
    res = spearman_ci(d["H_depth"], d["ColorHarmony"])
    z, p = one_sided_below(res["rho"], res["n"], CONFIG["SESOI_RHO"])
    add_row("Study 1", "Smallest-effect test of H1", name, f"H0: rho(H_depth, Color Harmony) >= {CONFIG['SESOI_RHO']:.2f}",
            res["n"], res["rho"], res["ci_low"], res["ci_high"], z, p, "one-sided p; Fisher z with variance 1.06/(n - 3)")

# Study 1b: what H_depth and the ratings respond to (lightness, chroma, colorfulness)
for name, d in SUBSETS.items():
    for target, tlabel in [("H_depth", "H_depth"), ("ColorHarmony", "Color Harmony rating")]:
        for v in ["mean_L", "mean_C", "colorfulness"]:
            res = spearman_ci(d[target], d[v])
            add_row("Study 1", "Determinants of H_depth and of the ratings", name, f"rho({tlabel}, {v})", res["n"],
                    res["rho"], res["ci_low"], res["ci_high"], np.nan, res["p"], "Spearman rho, 95% bootstrap CI")

# Study 1c: are the image statistics more strongly associated with the ratings than H_depth? (Meng et al., 1992)
for name, d in SUBSETS.items():
    block = []
    for v in COVARIATES:
        res = delta_rho(d["ColorHarmony"], d[v], d["H_depth"])
        block.append(OrderedDict(study="Study 1", analysis="Image statistics vs H_depth (dependent correlations)", sample=name,
                                 measure=f"rho(rating, {v}) - rho(rating, H_depth)", n=res["n"], estimate=res["delta_rho"],
                                 ci_low=res["ci_low"], ci_high=res["ci_high"], statistic=res["z_mrr"], p=res["p_mrr"],
                                 p_holm=np.nan, note="Meng, Rosenthal & Rubin (1992) z; paired bootstrap 95% CI; Holm within sample"))
    for b, a in zip(block, holm([b["p"] for b in block])):
        b["p_holm"] = a
    rows += block

# Study 1d: overlap between the samples and between the two annotation subsamples
_ver, _str, _gen = (set(SUBSETS[k]["key"]) if k in SUBSETS else set() for k in ["Interior (verified)", "Interior (CLIP strict)", "General"])
add_row("Study 1", "Sample overlap", "Interior (verified) and General", "images in both samples", len(_ver & _gen), np.nan)
add_row("Study 1", "Sample overlap", "Interior (CLIP strict) and General", "images in both samples", len(_str & _gen), np.nan)
if ANNOTATION_FILE.exists() and SECOND_RATER_FILE.exists():
    _a1 = pd.read_excel(ANNOTATION_FILE, sheet_name="labels")
    _pilot = set(_a1.loc[_a1.get("source", pd.Series(dtype=str)) == "audit_100", "key"].astype(str))
    _sr = set(pd.read_excel(SECOND_RATER_FILE, sheet_name="labels")["key"].astype(str))
    add_row("Study 1", "Sample overlap", "Pilot audit and reliability sample", "images in both annotation subsamples",
            len(_pilot & _sr), np.nan, note=f"pilot n = {len(_pilot)}, reliability sample n = {len(_sr)}")

# Study 2: how much of the concordance of H_depth arises from the colors of the scene alone?
_ref = sun_feat["H_depth_measured"]
for col, lab in [("H_depth_midas_same_pixels", "MiDaS layers (same pixels)"), ("H_bands", "horizontal bands (depth-free)"),
                 ("H_shuffled", "shuffled depth (depth of another scene)")]:
    lo, hi = bootstrap_stat(lin_ccc, _ref, sun_feat[col])
    n_ = len(_complete(_ref, sun_feat[col])[0])
    add_row("Study 2", "Color-content baselines", "SUN RGB-D", f"Lin's CCC, H_depth measured vs {lab}", n_,
            lin_ccc(_ref, sun_feat[col]), lo, hi, note="95% bootstrap CI")
for col, lab in [("H_bands", "horizontal bands"), ("H_shuffled", "shuffled depth")]:
    n_, dd, lo, hi = paired_ccc_difference(_ref, sun_feat["H_depth_midas_same_pixels"], sun_feat[col])
    add_row("Study 2", "Color-content baselines", "SUN RGB-D", f"Delta CCC, MiDaS layers minus {lab}", n_, dd, lo, hi,
            note="paired 95% bootstrap CI; scenes with all scores defined")

# Study 3: Holm adjustment across the four Kruskal-Wallis tests of Table 13
_ok = T13[np.isfinite(T13["p"].astype(float))]
for (_, r), a in zip(_ok.iterrows(), holm(_ok["p"].astype(float).to_numpy())):
    add_row("Study 3", "Category effects, multiplicity", r["corpus"], f"Kruskal-Wallis, {r['variable']}", int(r["n"]),
            float(r["epsilon2"]), statistic=float(r["H"]), p=float(r["p"]), note="estimate = epsilon squared; Holm across the four tests")
    rows[-1]["p_holm"] = float(a)

TS7 = save_table(pd.DataFrame(rows), "TableS7_additional_analyses", folder=SUP_DIR)
# ========================= END SECTION: additional analyses =========================

# =========================== SECTION: fig1 ===========================
# ---------------------------------------------------------------
# Figure 1 - pipeline illustration
# ---------------------------------------------------------------
if not FROM_FEATURES:   # needs the image data and/or the models
    from matplotlib.colors import ListedColormap

    def _verified_aadb_example():
        # Representative verified interior: the image whose H_depth is closest to the subset median.
        try:
            d = aadb_feat[aadb_feat["is_interior_verified"]]
        except Exception:
            return None
        if not len(d):
            return None
        member = d.loc[(d["H_depth"] - d["H_depth"].median()).abs().idxmin(), "member"]
        img, _ = to_srgb_image(Image.open(io.BytesIO(AADB_ZIP.read(member))))
        resize_max_side(img, CONFIG["WORK_MAX_SIDE"]).save(CACHE_DIR / "example_aadb_verified.png")
        return "example_aadb_verified.png"


    _ex = _verified_aadb_example()
    examples = [(_ex, "AADB interior (verified)") if _ex else ("example_aadb_interior.png", "AADB interior"),
                ("example_sun.png", "SUN RGB-D"), ("example_style.png", "Interior-Style")]
    examples = [(p, t) for p, t in examples if (CACHE_DIR / p).exists()]
    fig, axes = plt.subplots(len(examples), 4, figsize=(8.0, 2.1 * len(examples)), squeeze=False)
    layer_cmap = ListedColormap(["#D9D9D9", OKABE_ITO[3], OKABE_ITO[2], OKABE_ITO[0]])
    names = ["Near", "Middle", "Far"]
    for row, (fname, title) in enumerate(examples):
        rgb = np.asarray(Image.open(CACHE_DIR / fname).convert("RGB"))
        clo = normalize_closeness(midas_disparity(rgb))
        feats, maps = image_features(rgb, clo, image_rng(f"fig1|{fname}"), keep_maps=True, k_list=[CONFIG["K_LAYERS"]])
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"{title}: input", fontsize=8)
        axes[row, 1].imshow(clo, cmap="magma")
        axes[row, 1].set_title("MiDaS depth (bright = near)", fontsize=8)
        axes[row, 2].imshow(maps["layers"] + 1, cmap=layer_cmap, vmin=0, vmax=3, interpolation="nearest")
        axes[row, 2].set_title("K = 3 depth layers\n(orange near, green middle, blue far)", fontsize=7.5)
        ax = axes[row, 3]
        cols = maps["layer_colors"]
        for i, c in enumerate(cols):
            if c is None:
                continue
            srgb = np.clip(skcolor.lab2rgb(np.asarray(c, float).reshape(1, 1, 3)), 0, 1).reshape(3)
            ax.add_patch(plt.Rectangle((i * 1.1, 1.0), 1.0, 1.0, color=srgb))
            ax.text(i * 1.1 + 0.5, 0.82, names[i], ha="center", va="top", fontsize=7)
        pairs = [(0, 1), (0, 2), (1, 2)]
        txt = []
        for i, j in pairs:
            if cols[i] is not None and cols[j] is not None:
                txt.append(f"CH({names[i][0]},{names[j][0]}) = {float(ou_luo_2006(cols[i], cols[j])):.2f}")
        ax.text(1.6, 0.35, "\n".join(txt) + f"\nH_depth = {feats['H_depth']:.2f}", ha="center", va="top", fontsize=7)
        ax.set_xlim(-0.1, 3.4)
        ax.set_ylim(-0.6, 2.1)
        ax.set_title("Dominant layer colors", fontsize=8)
        for a in axes[row]:
            a.axis("off")
    fig.tight_layout()
    save_fig(fig, "Fig1_pipeline_examples")
# ========================= END SECTION: fig1 =========================

# =========================== SECTION: export ===========================
# ---------------------------------------------------------------
# Sample summary (Table 1), key results, README, archive and download
# ---------------------------------------------------------------
T1 = pd.DataFrame([
    dict(study="Study 1", source="AADB (Kong et al., 2016), Hugging Face mirror " + CONFIG["AADB_REPO"],
         subset="Interior (verified): CLIP candidates confirmed by manual annotation",
         n=int(aadb_feat["is_interior_verified"].sum()), depth="MiDaS", human_ratings="Color Harmony (5 raters, mean)"),
    dict(study="Study 1", source="AADB", subset="Interior (CLIP strict): interior-space p >= %.2f, before verification" % CONFIG["STRICT_INTERIOR_THRESHOLD"],
         n=int(aadb_feat["is_interior_strict"].sum()), depth="MiDaS", human_ratings="Color Harmony (5 raters, mean)"),
    dict(study="Study 1 (screening only)", source="AADB", subset="Indoor (broad): CLIP indoor p >= %.2f; people-centered, not analyzed" % CONFIG["INTERIOR_PROB_THRESHOLD"],
         n=int(aadb_feat["is_interior"].sum()), depth="MiDaS", human_ratings="Color Harmony (5 raters, mean)"),
    dict(study="Study 1", source="AADB", subset="General random sample", n=int(aadb_feat["in_general_sample"].sum()),
         depth="MiDaS", human_ratings="Color Harmony (5 raters, mean)"),
    dict(study="Study 2", source="SUN RGB-D V1 (Song et al., 2015)", subset="Random scenes", n=len(sun_feat),
         depth="Measured (depth_bfx) and MiDaS", human_ratings="none"),
    dict(study="Study 3", source="SUN RGB-D V1", subset="Same scenes as Study 2", n=len(sun_feat), depth="MiDaS", human_ratings="none"),
    dict(study="Study 3", source="Interior-Style (" + CONFIG["STYLE_REPO"] + ")", subset="Shuffled stream, bottom 8% cropped",
         n=len(style_feat), depth="MiDaS", human_ratings="none"),
])
save_table(T1, "Table1_study_samples")

# Files written by earlier versions under other names are removed, so that outputs/ only holds current files.
for _old in ["Fig9_within_corpus_colourfulness_depth.png", "Fig9_within_corpus_colourfulness_depth.pdf"]:
    (FIG_DIR / _old).unlink(missing_ok=True)


def _get(df, **conds):
    d = df
    for k, v in conds.items():
        d = d[d[k] == v]
    return d.iloc[0] if len(d) else None


summary = OrderedDict()
for s in SUBSETS:
    r = _get(T3, subset=s, outcome="ColorHarmony", predictor="H_depth")
    summary[f"Study1_{s}_rho_Hdepth_humanCH"] = [r["n"], r["rho"], r["ci_low"], r["ci_high"], r["p"]]
    for ctrl in ["H_global", "H_bands", "H_shuffled"]:
        r = _get(T4, subset=s, comparison=f"H_depth vs {ctrl}")
        summary[f"Study1_{s}_delta_rho_vs_{ctrl}"] = [r["delta_rho"], r["ci_low"], r["ci_high"], r["p_mrr_holm"]]
    r = _get(T5, subset=s, predictor="H_depth")
    summary[f"Study1_{s}_partial_rho_Hdepth"] = [r["partial_rho"], r["ci_low"], r["ci_high"], r["p"]]
    r = _get(T6, subset=s, model_or_comparison="Base + H_depth minus Base")
    summary[f"Study1_{s}_deltaR2_Hdepth_over_base"] = [r["mean_diff"], r["ci_low"], r["ci_high"], r["p_holm"]]
for _, r in T8.iterrows():
    summary[f"Study2_{r['measure']}"] = [r["n"], r["estimate"], r["q1_or_ci_low"], r["q3_or_ci_high"], r["p"]]
for c in CORPORA:
    r = within[("colorfulness", "depth_dispersion", c)]
    summary[f"Study3_{c}_rho_colorfulness_depthdispersion"] = [r["n"], r["rho"], r["ci_low"], r["ci_high"], r["p"]]
summary = {k: [float(x) if isinstance(x, (int, float, np.integer, np.floating)) else x for x in v] for k, v in summary.items()}
(REP_DIR / "key_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("KEY RESULTS (n, estimate, CI low, CI high, p where applicable)")
for k, v in summary.items():
    print(f"  {k}: " + ", ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in v))

README = f'''# Beyond the Color Wheel - analysis outputs

Generated by beyond_color_wheel.py on {time.strftime("%Y-%m-%d %H:%M")} ({"reanalysis from the shared per-image features, --from-features" if FROM_FEATURES else "full pipeline"}).

## Folders
- figures/          main figures (PNG at 300 dpi and vector PDF)
- tables/           main tables (CSV with comma separator and dot decimal; XLSX with native numbers)
- supplementary/    implementation checks, label matching, quality control, annotation summary, agreement by sensor
- data/             per-image features with image identifiers (AADB keys, SUN RGB-D scene paths, Interior-Style image hashes)
- reproducibility/  config.json, environment.json, pip_freeze.txt, key_results.json

## Configuration
{json.dumps(CONFIG, indent=2)}

## Data sources and methods (see the article for the full reference list)
- Kong, S., Shen, X., Lin, Z., Mech, R., & Fowlkes, C. (2016). Photo aesthetics ranking network with attributes and content adaptation. In Computer Vision - ECCV 2016 (pp. 662-679). Springer. https://doi.org/10.1007/978-3-319-46448-0_40
- Song, S., Lichtenberg, S. P., & Xiao, J. (2015). SUN RGB-D: A RGB-D scene understanding benchmark suite. In Proceedings of the IEEE CVPR (pp. 567-576). https://doi.org/10.1109/CVPR.2015.7298655
- Interior-Style corpus: Hugging Face dataset {CONFIG["STYLE_REPO"]} (see the dataset card for license and citation).
- Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2022). Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE TPAMI, 44(3), 1623-1637. https://doi.org/10.1109/TPAMI.2020.3019967
- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. In Proceedings of ICML (PMLR 139, pp. 8748-8763).
- Ou, L.-C., & Luo, M. R. (2006). A colour harmony model for two-colour combinations. Color Research & Application, 31(3), 191-204. https://doi.org/10.1002/col.20208
- Ou, L.-C., Chong, P., Luo, M. R., & Minchew, C. (2011). Additivity of colour harmony. Color Research & Application, 36(5), 355-372. https://doi.org/10.1002/col.20624
- Hasler, D., & Suesstrunk, S. E. (2003). Measuring colorfulness in natural images. Proceedings of SPIE 5007, 87-95. https://doi.org/10.1117/12.477378
'''
(OUT / "README.md").write_text(README, encoding="utf-8")
log_environment()

zip_path = BASE_DIR / "BeyondColorWheel_outputs.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in OUT.rglob("*"):
        if p.is_file() and CACHE_DIR not in p.parents and p.name != "run_log.txt":
            zf.write(p, arcname=str(p.relative_to(OUT)))
print(f"Archive: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
print(f"Finished in {(time.time() - RUN_START) / 60:.1f} min. Outputs: {OUT}")
# ========================= END SECTION: export =========================
