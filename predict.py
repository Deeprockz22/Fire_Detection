#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py
==========
Interactive interface for the Mask R-CNN Fire Detection pipeline.
Provides visual predictions, batch evaluation, dataset generation, and training.
"""

import sys
import os
import time
import subprocess
import argparse
import csv
import random
from pathlib import Path

# Disable plot popups for background tasks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Torch
try:
    import torch
    import torchvision
    from PIL import Image
    from torchvision import transforms
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# Fix encoding
if sys.platform == 'win32':
    try:
        import codecs
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass

# ============================================================================
# CONFIGURATION
# ============================================================================
VERSION = "1.0.0"
BASE_DIR = Path(__file__).parent.resolve()

INPUT_DIR   = BASE_DIR / "Input"
OUTPUT_DIR  = BASE_DIR / "Output"
CKPT_DIR    = BASE_DIR / "checkpoints"
DATA_DIR    = BASE_DIR / "Dataset"
SCRIPTS_DIR = BASE_DIR / "scripts"
LOGS_DIR    = BASE_DIR / "logs"

MODEL_PATH  = CKPT_DIR / "maskrcnn_fire_best.pt"
MANIFEST    = DATA_DIR / "manifest.csv"

DEVICE      = "cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu"
NUM_CLASSES = 2
SCORE_THR   = 0.5
MASK_THR    = 0.5

# ============================================================================
# UI UTILITIES
# ============================================================================
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def press_enter():
    try: input("\nPress Enter to continue...")
    except: pass

_R  = "\033[31m";  _Y  = "\033[33m"; _W  = "\033[97m"
_RB = "\033[91m";  _YB = "\033[93m"; _RST = "\033[0m"
_FRAMES = [
    [
        f"     {_RB}  ) {_Y} ({_RB}  ){_RST}   ",
        f"    {_Y} ( {_RB}){_Y}  (  {_RB}( {_RST}  ",
        f"   {_RB}){_Y}(   {_RB}) {_Y}( {_RB})  {_RST} ",
        f"  {_Y}(  {_RB})   {_Y}(   {_RB})  {_RST} ",
        f" {_R}  \\{_RB}|{_Y}///{_RB}|{_Y}///{_R}|/{_RST}",
        f" {_R}   \\{_RB}|{_R}/////|{_RB}\\{_R}/{_RST} ",
        f"  {_R}   \\{_RB}|||{_R}///{_RST}    ",
        f"   {_R}   \\{_RB}|{_R}/ {_RST}      ",
        f"    {_R}───┴───{_RST}    ",
    ],
    [
        f"     {_Y}(  {_RB}) {_Y}  ({_RB}){_RST}  ",
        f"    {_RB}){_Y}  ({_RB})  {_Y}(  {_RST} ",
        f"   {_Y}( {_RB}) {_Y}(  {_RB})  {_Y}({_RST} ",
        f"  {_RB})  {_Y}(   {_RB})  {_Y}(  {_RST}",
        f" {_R}  \\{_Y}|{_RB}\\\\\\{_Y}|{_RB}\\\\\\{_R}|/{_RST}",
        f" {_R}   \\{_Y}|{_R}/////|{_Y}\\{_R}/{_RST} ",
        f"  {_R}   \\{_Y}|||{_R}///{_RST}    ",
        f"   {_R}   \\{_Y}|{_R}/ {_RST}      ",
        f"    {_R}───┴───{_RST}    ",
    ],
    [
        f"    {_RB} ( {_Y}){_RB}  ( {_Y}) {_RST}  ",
        f"   {_Y} ){_RB}(  {_Y}) {_RB} ( {_Y}) {_RST}",
        f"   {_RB}( {_Y})  {_RB}(  {_Y}){_RB}( {_RST} ",
        f"  {_Y})  {_RB})   {_Y}(   {_RB})  {_RST}",
        f" {_R}  \\{_RB}|{_R}\\\\\\{_RB}|{_R}\\\\\\{_RB}|/{_RST}",
        f" {_R}   \\{_RB}|{_R}/////|{_Y}\\{_R}/{_RST} ",
        f"  {_R}   \\{_RB}|||{_R}///{_RST}    ",
        f"   {_R}   \\{_RB}|{_R}/ {_RST}      ",
        f"    {_R}───┴───{_RST}    ",
    ],
]

def fire_splash():
    try:
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()
        clear_screen()
        n = len(_FRAMES[0])
        first = True
        for _ in range(6):
            for frame in _FRAMES:
                if not first:
                    sys.stdout.write(f"\033[{n}A")
                for line in frame: print(line)
                sys.stdout.flush()
                time.sleep(0.10)
                first = False
    except: pass
    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
    clear_screen()

def print_banner():
    w = 66
    print()
    print("╔" + "═" * (w - 2) + "╗")
    print("║" + " " * (w - 2) + "║")
    title = "🔥 FIRE DETECTION TOOL: MASK R-CNN  v" + VERSION
    sub   = "Instance Segmentation for Synthetic Fire Environments"
    print("║" + title.center(w - 2) + "║")
    print("║" + sub.center(w - 2) + "║")
    print("║" + " " * (w - 2) + "║")
    print("╚" + "═" * (w - 2) + "╝")
    print()

def print_section(title):
    w = 66
    bar = "─" * (w - 2)
    print()
    print("┌" + bar + "┐")
    print("│  " + title.ljust(w - 4) + "│")
    print("└" + bar + "┘")
    print()

# ============================================================================
# VISION UTILS
# ============================================================================
def build_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

def load_trained_model():
    if not MODEL_PATH.exists():
        print(f"❌ Model checkpoint not found: {MODEL_PATH}")
        return None
    model = build_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def get_pred_mask(output, h, w, score_thr=SCORE_THR, mask_thr=MASK_THR):
    pred = np.zeros((h, w), dtype=np.uint8)
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    masks  = output["masks"].cpu().numpy()
    for score, label, soft in zip(scores, labels, masks):
        if label == 1 and score >= score_thr:
            pred = np.maximum(pred, (soft[0] >= mask_thr).astype(np.uint8))
    return pred

def overlay(img_rgb, mask, color, alpha=0.45):
    out = img_rgb.copy().astype(np.float32)
    for c, v in enumerate(color):
        out[:, :, c] = np.where(mask > 0,
                                 out[:, :, c] * (1 - alpha) + v * alpha * 255,
                                 out[:, :, c])
    return out.astype(np.uint8)

def iou_score(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return float(tp / (tp + fp + fn + 1e-8))

# ============================================================================
# ACTIONS
# ============================================================================
def action_quick_predict():
    print_section("🖼️  QUICK PREDICT")
    INPUT_DIR.mkdir(exist_ok=True)
    imgs = sorted(list(INPUT_DIR.glob("*.png")) + list(INPUT_DIR.glob("*.jpg")))
    
    if imgs:
        print("Images in Input/:")
        for i, f in enumerate(imgs, 1):
            print(f"  {i}. {f.name}")
        print()
    else:
        print("💡 Drop .png or .jpg files into Input/ first, or enter full path.\n")

    raw = input("File number or full path: ").strip().strip('"').strip("'")
    if not raw: return

    if raw.isdigit() and imgs:
        idx = int(raw) - 1
        if 0 <= idx < len(imgs):
            img_path = imgs[idx]
        else:
            print("❌ Invalid number."); return
    else:
        img_path = Path(raw)
        if not img_path.exists():
            print(f"❌ File not found: {img_path}"); return

    model = load_trained_model()
    if not model: return

    print(f"\n⚙️  Predicting fire in {img_path.name}...")
    try:
        pil_img = Image.open(img_path).convert("RGB")
        img_np  = np.array(pil_img)
        h, w    = img_np.shape[:2]
        img_t   = transforms.ToTensor()(pil_img).to(DEVICE)
        
        with torch.no_grad():
            output = model([img_t])[0]
        
        pred = get_pred_mask(output, h, w, SCORE_THR, MASK_THR)
        
        # Display
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor("#1a1a2e")
        axs[0].imshow(img_np)
        axs[0].set_title("Input", color="white", fontsize=12)
        axs[0].axis("off")
        
        axs[1].imshow(overlay(img_np, pred, (1, 0.2, 0)))
        axs[1].set_title("Prediction (Mask R-CNN)", color="white", fontsize=12)
        axs[1].axis("off")
        
        OUTPUT_DIR.mkdir(exist_ok=True)
        save_path = OUTPUT_DIR / f"{img_path.stem}_pred.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"✅ Saved plot: {save_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    press_enter()


def action_eval():
    print_section("📊 EVALUATE TEST SET")
    print("Runs evaluate_maskrcnn.py on the test set split in manifest.csv.")
    print("This will print Pixel-level and Detection-level metrics.\n")
    script = SCRIPTS_DIR / "evaluate_maskrcnn.py"
    if not script.exists():
        print("❌ Script not found."); press_enter(); return
    subprocess.run([sys.executable, str(script)])
    press_enter()


def action_visualize_grid():
    print_section("📸 GENERATE VIZ GRID")
    print("Generates a side-by-side comparison grid for test set samples.")
    model = load_trained_model()
    if not model: return
    
    if not MANIFEST.exists():
        print(f"❌ Manifest not found: {MANIFEST}")
        press_enter(); return

    print("Loading test samples...")
    entries, fire, nofire = [], [], []
    with open(MANIFEST, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                entries.append((row["image_path"], row["mask_path"]))
                
    for img_p, msk_p in entries:
        gt = np.array(Image.open(msk_p).convert("L"))
        if (gt >= 128).sum() > 0: fire.append((img_p, msk_p))
        else: nofire.append((img_p, msk_p))

    n_samples = 16
    n_fire    = min(int(n_samples * 0.75), len(fire))
    n_nofire  = min(n_samples - n_fire, len(nofire))
    selected  = random.sample(fire, n_fire) + random.sample(nofire, n_nofire)
    random.shuffle(selected)
    print(f"Selected {len(selected)} samples for grid.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(selected), 3, figsize=(12, len(selected)*3))
    fig.patch.set_facecolor("#1a1a2e")
    cols = ["Input", "Ground Truth", "Prediction"]
    for ax, lbl in zip(axes[0], cols):
        ax.set_title(lbl, color="white", fontsize=11, fontweight="bold", pad=6)

    to_tensor = transforms.ToTensor()
    with torch.no_grad():
        for row_i, (img_p, msk_p) in enumerate(selected):
            pil_img = Image.open(img_p).convert("RGB")
            img_np  = np.array(pil_img)
            gt      = (np.array(Image.open(msk_p).convert("L")) >= 128).astype(np.uint8)
            h, w    = gt.shape

            out     = model([to_tensor(pil_img).to(DEVICE)])[0]
            pred    = get_pred_mask(out, h, w)
            iou     = iou_score(pred, gt)

            imgs = [img_np, overlay(img_np, gt, (0,1,0)), overlay(img_np, pred, (1,0.2,0))]
            b_color = "#00e676" if iou >= 0.5 else ("#ff6b35" if gt.sum() > 0 else "#888")

            for col_i, show_img in enumerate(imgs):
                ax = axes[row_i, col_i]
                ax.imshow(show_img)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor(b_color)
                    spine.set_linewidth(2)
                if col_i == 2:
                    color = "#00e676" if iou >= 0.5 else "#ff6b35"
                    ax.set_xlabel(f"IoU={iou:.2f}", color=color, fontsize=9)

    out_path = OUTPUT_DIR / "test_predictions_grid.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"✅ Grid saved to {out_path}")
    press_enter()


def action_prepare():
    while True:
        clear_screen()
        print_banner()
        print("PREPARE DATASET\n")
        print("  1. 🩹 Patch FDS Files   (enables .sf dump for new FDS scenarios)")
        print("  2. 🎞️ Extract Frames   (parses .sf binary into image+label frames)")
        print("  3. 📑 Prepare Dataset  (splits into train/val/test & generates manifest)")
        print("  4. ← Back\n")
        c = input("Choose (1-4): ").strip()
        if c == '1': subprocess.run([sys.executable, str(SCRIPTS_DIR / "patch_fds_for_slcf.py")]); press_enter()
        elif c == '2': subprocess.run([sys.executable, str(SCRIPTS_DIR / "extract_frames.py")]); press_enter()
        elif c == '3': subprocess.run([sys.executable, str(SCRIPTS_DIR / "prepare_dataset.py")]); press_enter()
        elif c == '4': break


def action_train():
    print_section("🧠 TRAIN MODEL")
    print("Runs train_maskrcnn.py on the current Dataset split.")
    print("This requires a GPU for reasonable speed.\n")
    confirm = input("Start training? [y/N]: ").strip().lower()
    if confirm == 'y':
        cmd = [sys.executable, str(SCRIPTS_DIR / "train_maskrcnn.py")]
        res = input("Resume from last checkpoint if exists? [y/N]: ").strip().lower()
        if res == 'y': cmd.append("--resume")
        subprocess.run(cmd)
    press_enter()


def action_files():
    while True:
        clear_screen()
        print_banner()
        print("MANAGE FILES\n")
        print("  1. 📥 List Input/ files")
        print("  2. 📤 List Output/ files")
        print("  3. 🗂️  Open Input/ folder")
        print("  4. 🗂️  Open Output/ folder")
        print("  5. 🧹 Clear Output/ folder")
        print("  6. ← Back\n")
        c = input("Choose (1-6): ").strip()
        if c == "1":
            files = list(INPUT_DIR.glob("*.*")) if INPUT_DIR.exists() else []
            print_section("📥 INPUT FILES")
            if files:
                for f in files: print(f"  • {f.name} ({f.stat().st_size/1024:.0f} KB)")
            else: print("  (empty)")
            press_enter()
        elif c == "2":
            files = list(OUTPUT_DIR.glob("*.*")) if OUTPUT_DIR.exists() else []
            print_section("📤 OUTPUT FILES")
            if files:
                for f in files: print(f"  • {f.name} ({f.stat().st_size/1024:.0f} KB)")
            else: print("  (empty)")
            press_enter()
        elif c == "3":
            INPUT_DIR.mkdir(exist_ok=True)
            if sys.platform == "win32": os.startfile(str(INPUT_DIR))
            press_enter()
        elif c == "4":
            OUTPUT_DIR.mkdir(exist_ok=True)
            if sys.platform == "win32": os.startfile(str(OUTPUT_DIR))
            press_enter()
        elif c == "5":
            files = list(OUTPUT_DIR.glob("*.*")) if OUTPUT_DIR.exists() else []
            if files:
                conf = input(f"Delete {len(files)} files? [y/N]: ").strip().lower()
                if conf == 'y':
                    for f in files: f.unlink()
                    print("✅ Deleted.")
            else:
                print("Already empty.")
            press_enter()
        elif c == "6":
            break


def action_diagnostics():
    print_section("🔧 DIAGNOSTICS")
    print(f"Device: {DEVICE.upper()}")
    print("Packages:")
    for pkg in ["torch", "torchvision", "numpy", "matplotlib", "PIL", "pandas"]:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg:<15}")
        except ImportError:
            print(f"  ❌ {pkg:<15} ← missing")
            
    print("\nDataset & Model:")
    print(f"  {'✅' if MANIFEST.exists() else '❌'} manifest.csv")
    print(f"  {'✅' if MODEL_PATH.exists() else '❌'} maskrcnn_fire_best.pt")
    press_enter()


# ============================================================================
# MAIN MENU
# ============================================================================
def show_main_menu():
    print_banner()
    w = 66
    bar = "─" * (w - 2)
    print("┌" + bar + "┐")
    print("│" + "  MAIN MENU".ljust(w - 2) + "│")
    print("├" + bar + "┤")
    items = [
        ("1", "🖼️", "Quick Predict",   "pick an image from Input/"),
        ("2", "📊", "Evaluate Test Set", "metrics on held-out test data"),
        ("3", "📸", "Generate Viz Grid", "16-panel evaluation grid -> Output/"),
        ("4", "⚙️", "Prepare Dataset",   "patch -> extract -> manifest pipelines"),
        ("5", "🧠", "Train Model",       "fine-tune Mask R-CNN"),
        ("6", "📁", "Manage Files",      "Input, Output, browse assets"),
        ("7", "🔧", "Diagnostics",       "check packages & model weights"),
    ]
    for num, icon, name, desc in items:
        line = f"  {num}.  {icon}  {name:<18} {desc}"
        print("│" + line.ljust(w - 2) + "│")
    print("├" + bar + "┤")
    print("│" + "  0.  🚪  Exit".ljust(w - 2) + "│")
    print("└" + bar + "┘")
    print()

def main():
    if not TORCH_OK:
        print("❌ CRITICAL: PyTorch/Torchvision not installed.")
        print("   pip install torch torchvision")
        sys.exit(1)
        
    fire_splash()
    
    ACTIONS = {
        "1": action_quick_predict,
        "2": action_eval,
        "3": action_visualize_grid,
        "4": action_prepare,
        "5": action_train,
        "6": action_files,
        "7": action_diagnostics,
    }
    
    while True:
        clear_screen()
        show_main_menu()
        choice = input("Choose (1-7, 0 to exit): ").strip()
        if choice == "0":
            clear_screen()
            print("\n👋 Goodbye!\n")
            break
        elif choice in ACTIONS:
            clear_screen()
            ACTIONS[choice]()
        else:
            print("❌ Invalid choice")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_screen()
        print("\n👋 Goodbye!\n")
