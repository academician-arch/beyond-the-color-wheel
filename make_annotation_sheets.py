# -*- coding: utf-8 -*-
r"""
Beyond the Color Wheel - manual interior annotation material
=============================================================
Creates everything needed to label the AADB 'Interior (strict)' candidates by hand.

Run in the project folder (e.g. D:\color_wheel) with the virtual environment active:
    python make_annotation_sheets.py

Needs the outputs of beyond_color_wheel.py (outputs/data/aadb_scene_filter_and_samples.csv) and the AADB archive
(data/aadb/AADB.zip). If you already filled TableS5_strict_interior_audit_template.xlsx, copy it into the
annotations folder first; its labels are carried over automatically.

Creates (annotations folder):
    interior_annotation.xlsx              sheet 'labels': one row per candidate (label 1 = interior space, 0 = not)
                                          sheet 'criteria': the labeling rules
    pages/page_01.png, ...                numbered images; the numbers match column 'ann_no'
    second_rater/second_rater_annotation.xlsx + second_rater/pages/
                                          blind random sample of 100 candidates for an independent second annotator
Existing Excel files that already contain labels are never overwritten (use --force to rebuild them).
The annotation pages show numbered thumbnails only, so ratings, CLIP probabilities and harmony scores stay hidden.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
SEED = 42                     # same seed as the main analysis
STRICT_THRESHOLD = 0.60       # same rule as CONFIG["STRICT_INTERIOR_THRESHOLD"] in beyond_color_wheel.py
SECOND_RATER_N = 100
PER_PAGE, COLS = 48, 6
CELL_W, CELL_H, LABEL_H, HEADER_H = 320, 240, 40, 60

SCENE_FILE = BASE / "outputs" / "data" / "aadb_scene_filter_and_samples.csv"
AADB_ZIP_FILE = BASE / "data" / "aadb" / "AADB.zip"
ANN_DIR = BASE / "annotations"
MAIN_FILE = ANN_DIR / "interior_annotation.xlsx"
SR_DIR = ANN_DIR / "second_rater"
SR_FILE = SR_DIR / "second_rater_annotation.xlsx"

CRITERIA = pd.DataFrame({
    "rule": [
        "Label 1 (interior space)",
        "Label 0 (not an interior space)",
        "Ambiguous cases",
        "Blinding",
    ],
    "criterion": [
        "The main subject of the photograph is an enclosed interior space (room, hall, corridor, library, church nave, "
        "shop or restaurant interior, etc.). Enough of the space's boundaries (floor, walls and/or ceiling) is visible to "
        "read it as a space. People may be present but must not dominate the frame.",
        "Any of: close-up of an object, furniture piece, artwork or architectural detail; a view mainly of a ceiling, "
        "dome, window or opening; vehicle or transport interiors (train, car, escalator shaft); caves and tunnels; "
        "exteriors or views to the outside; people-dominated scenes; images too dark or abstract to read the space.",
        "Ask: would an interior architect use this photograph to show the space itself? If still unsure, choose the "
        "more conservative label (0) and write a short note.",
        "Label only from the images on the pages. Do not look up ratings, scores or analysis results while labeling.",
    ],
})


def _font(size):
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _has_labels(path):
    if not path.exists():
        return False
    try:
        lab = pd.to_numeric(pd.read_excel(path, sheet_name="labels")["label"], errors="coerce")
        return bool(lab.notna().any())
    except Exception:
        return True   # unreadable or unexpected layout: treat as user data and never overwrite


def _write_excel(path, labels):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        labels.to_excel(xw, sheet_name="labels", index=False)
        CRITERIA.to_excel(xw, sheet_name="criteria", index=False)
        ws = xw.sheets["labels"]
        ws.freeze_panes = "A2"
        for col, width in zip("ABCDEFG", [9, 7, 30, 10, 14, 40, 40]):
            ws.column_dimensions[col].width = width
        wc = xw.sheets["criteria"]
        wc.column_dimensions["A"].width = 30
        wc.column_dimensions["B"].width = 120


def _render_pages(items, zf, out_dir, title):
    # items: list of (number, zip member). Writes PNG pages with numbered thumbnails.
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("page_*.png"):
        old.unlink()
    f_num, f_head = _font(26), _font(28)
    n_pages = int(np.ceil(len(items) / PER_PAGE))
    for p in range(n_pages):
        chunk = items[p * PER_PAGE:(p + 1) * PER_PAGE]
        rows = int(np.ceil(len(chunk) / COLS))
        canvas = Image.new("RGB", (COLS * CELL_W, HEADER_H + rows * (CELL_H + LABEL_H)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 14), f"{title} - page {p + 1}/{n_pages}", fill="black", font=f_head)
        for i, (num, member) in enumerate(chunk):
            r, c = divmod(i, COLS)
            x0, y0 = c * CELL_W, HEADER_H + r * (CELL_H + LABEL_H)
            draw.text((x0 + 10, y0 + 6), f"#{num}", fill="black", font=f_num)
            try:
                img = Image.open(io.BytesIO(zf.read(member)))
                if img.format == "JPEG":
                    img.draft("RGB", (CELL_W * 2, CELL_H * 2))
                img = img.convert("RGB")
                img.thumbnail((CELL_W - 12, CELL_H - 8), Image.LANCZOS)
                canvas.paste(img, (x0 + (CELL_W - img.width) // 2, y0 + LABEL_H + (CELL_H - img.height) // 2))
            except Exception as exc:
                draw.text((x0 + 10, y0 + LABEL_H + 10), f"unreadable: {exc!r}"[:40], fill="red", font=f_num)
        canvas.save(out_dir / f"page_{p + 1:02d}.png", optimize=True)
    return n_pages


def main():
    for path, hint in [(SCENE_FILE, "run beyond_color_wheel.py first"), (AADB_ZIP_FILE, "the AADB archive is downloaded by beyond_color_wheel.py")]:
        if not path.exists():
            sys.exit(f"Missing {path} ({hint}).")
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    scene = pd.read_csv(SCENE_FILE)
    cand = scene[scene["p_space"] >= STRICT_THRESHOLD].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"Strict interior candidates: {len(cand)}")

    # carry over the labels of the first 100-image audit
    audit_labels, audit_notes = {}, {}
    for path in [ANN_DIR / "TableS5_strict_interior_audit_template.xlsx",
                 BASE / "outputs" / "supplementary" / "TableS5_strict_interior_audit_template.xlsx"]:
        if path.exists():
            a = pd.read_excel(path)
            if "is_interior_space_1_or_0" in a:
                lab = pd.to_numeric(a["is_interior_space_1_or_0"], errors="coerce")
                if lab.notna().any():
                    audit_labels = {k: int(v) for k, v in zip(a["key"].astype(str), lab) if pd.notna(v)}
                    if "note" in a:
                        audit_notes = {k: str(v) for k, v in zip(a["key"].astype(str), a["note"]) if pd.notna(v)}
                    print(f"Carried over {len(audit_labels)} audit labels from {path}")
                    break
    if not audit_labels:
        print("No filled audit template found; all candidates will be labeled from scratch.")

    zf = zipfile.ZipFile(AADB_ZIP_FILE)

    # ---------------- first annotator ----------------
    if _has_labels(MAIN_FILE) and "--force" not in sys.argv:
        print(f"{MAIN_FILE} already contains labels: not overwritten (use --force to rebuild it).")
    else:
        rows, todo, n = [], [], 0
        for r in cand.itertuples(index=False):
            if r.key in audit_labels:
                rows.append(dict(ann_no=np.nan, page=np.nan, key=r.key, label=audit_labels[r.key], source="audit_100",
                                 note=audit_notes.get(r.key, ""), member=r.member))
            else:
                n += 1
                todo.append((n, r.member))
                rows.append(dict(ann_no=n, page=(n - 1) // PER_PAGE + 1, key=r.key, label=np.nan, source="full_annotation",
                                 note="", member=r.member))
        labels = pd.DataFrame(rows, columns=["ann_no", "page", "key", "label", "source", "note", "member"])
        labels = pd.concat([labels[labels.source == "full_annotation"], labels[labels.source == "audit_100"]], ignore_index=True)
        n_pages = _render_pages(todo, zf, ANN_DIR / "pages", "Interior annotation")
        _write_excel(MAIN_FILE, labels)
        print(f"First annotator: {len(todo)} images to label on {n_pages} pages | {MAIN_FILE}")

    # ---------------- blind second annotator ----------------
    if _has_labels(SR_FILE) and "--force" not in sys.argv:
        print(f"{SR_FILE} already contains labels: not overwritten.")
    else:
        sr = cand.sample(n=min(SECOND_RATER_N, len(cand)), random_state=SEED + 7).reset_index(drop=True)
        sr_labels = pd.DataFrame({"sr_no": np.arange(1, len(sr) + 1), "page": np.arange(len(sr)) // PER_PAGE + 1,
                                  "key": sr["key"], "label": np.nan, "note": ""})
        SR_DIR.mkdir(parents=True, exist_ok=True)
        n_pages = _render_pages(list(zip(sr_labels["sr_no"], sr["member"])), zf, SR_DIR / "pages", "Second annotator")
        _write_excel(SR_FILE, sr_labels)
        print(f"Second annotator: {len(sr)} images on {n_pages} pages | {SR_FILE}")
    print("Done. Open the 'criteria' sheet before labeling.")


if __name__ == "__main__":
    main()
