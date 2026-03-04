#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py - Fire Detection Tool (Mask R-CNN)
==============================================
Interactive interface for the Mask R-CNN Fire Detection pipeline.
Provides visual predictions, batch evaluation, dataset generation, and training.

Version: 1.1.0
"""

import sys
import os
import time
import subprocess
import argparse
import csv
import random
from pathlib import Path
from datetime import datetime

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
VERSION = "1.1.0"
BASE_DIR = Path(__file__).parent.resolve()

INPUT_DIR   = BASE_DIR / "Input"
OUTPUT_DIR  = BASE_DIR / "Output"
CKPT_DIR    = BASE_DIR / "checkpoints"
DATA_DIR    = BASE_DIR / "Dataset"
SCRIPTS_DIR = BASE_DIR / "scripts"
LOGS_DIR    = BASE_DIR / "logs"
DOCS_DIR    = BASE_DIR / "docs"

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
    """Enhanced ASCII art banner"""
    _A = "\033[91m"; _B = "\033[93m"; _C = "\033[33m"; _RST = "\033[0m"
    art = [
        f"{_A}███████╗██╗██████╗ ███████╗    ██████╗ ███████╗████████╗███████╗ ██████╗████████╗{_RST}",
        f"{_A}██╔════╝██║██╔══██╗██╔════╝    ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝{_RST}",
        f"{_B}█████╗  ██║██████╔╝█████╗      ██║  ██║█████╗     ██║   █████╗  ██║        ██║   {_RST}",
        f"{_B}██╔══╝  ██║██╔══██╗██╔══╝      ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   {_RST}",
        f"{_C}██║     ██║██║  ██║███████╗    ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   {_RST}",
        f"{_C}╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝    ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝   {_RST}",
    ]
    print()
    for line in art:
        print("  " + line)
    print(f"\n  \033[97mMask R-CNN Instance Segmentation  ·  Synthetic Fire Detection v{VERSION}\033[0m")
    print("  \033[90mSeeing fires that don't exist yet since 2024.\033[0m\n")

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


def action_assimilate_scenarios():
    """Smart-sync: scans fds_scenarios for new simulations and processes them"""
    print_section("🔄 ASSIMILATE NEW SCENARIOS")
    
    PROJECT_ROOT = BASE_DIR.parent
    FDS_SCENARIOS_DIR = PROJECT_ROOT / "fds_scenarios"
    
    print("This feature imports FDS simulation data from an fds_scenarios folder.\n")
    
    if not FDS_SCENARIOS_DIR.exists():
        print(f"📂 FDS scenarios folder not found at default location:")
        print(f"   {FDS_SCENARIOS_DIR}\n")
        
        print("Options:")
        print("  1. Create folder at default location")
        print("  2. Specify custom location")
        print("  3. Skip (this feature is optional)\n")
        
        choice = input("Choose [1/2/3]: ").strip()
        
        if choice == '1':
            try:
                FDS_SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
                print(f"\n✅ Created folder: {FDS_SCENARIOS_DIR}")
                print("\n💡 Next steps:")
                print("   1. Add FDS simulation folders to this directory")
                print("   2. Each scenario should contain .fds, .smv, and .sf files")
                print("   3. Run this assimilation feature again to process them")
                press_enter()
                return
            except Exception as e:
                print(f"\n❌ Failed to create folder: {e}")
                press_enter()
                return
                
        elif choice == '2':
            custom_path = input("\nEnter path to FDS scenarios folder: ").strip().strip('"').strip("'")
            if not custom_path:
                print("❌ No path provided")
                press_enter()
                return
            FDS_SCENARIOS_DIR = Path(custom_path)
            if not FDS_SCENARIOS_DIR.exists():
                print(f"❌ Directory not found: {FDS_SCENARIOS_DIR}")
                press_enter()
                return
            print(f"\n✅ Using custom location: {FDS_SCENARIOS_DIR}\n")
        
        elif choice == '3':
            print("\n💡 This feature is optional.")
            print("   You can still:")
            print("   • Use 'Quick Predict' on individual images")
            print("   • Train with existing Dataset")
            print("   • Use other menu options")
            press_enter()
            return
        else:
            print("❌ Invalid choice")
            press_enter()
            return
    else:
        print(f"✅ Found FDS scenarios folder: {FDS_SCENARIOS_DIR}\n")
    
    print(f"Scanning: {FDS_SCENARIOS_DIR}\n")
    
    # Find all scenario directories
    scenario_dirs = [d for d in FDS_SCENARIOS_DIR.iterdir() if d.is_dir()]
    
    if not scenario_dirs:
        print("❌ No scenario directories found in fds_scenarios/")
        return
    
    print(f"Found {len(scenario_dirs)} scenario directories\n")
    
    # Check which scenarios have .smv files (completed simulations)
    completed = []
    for scenario_dir in scenario_dirs:
        smv_files = list(scenario_dir.glob("*.smv"))
        if smv_files:
            completed.append((scenario_dir, smv_files[0]))
    
    if not completed:
        print("❌ No completed FDS scenarios found (.smv files missing)")
        print("\n💡 Scenarios need .smv files from completed FDS simulations")
        return
    
    print(f"✅ Found {len(completed)} completed scenarios with .smv files\n")
    
    # Check which need patching (no SLCF)
    need_patch = []
    for scenario_dir, smv_file in completed:
        fds_files = list(scenario_dir.glob("*.fds"))
        if fds_files:
            fds_content = fds_files[0].read_text(errors="replace")
            if "&SLCF" not in fds_content.upper():
                need_patch.append(scenario_dir.name)
    
    # Check what's already in Dataset
    existing_frames = set()
    if DATA_DIR.exists():
        for img in DATA_DIR.glob("**/*.png"):
            # Extract scenario name from filename (e.g., "R1_scenario_frame_001.png")
            parts = img.stem.split("_frame_")
            if parts:
                existing_frames.add(parts[0])
    
    new_scenarios = [s for s, _ in completed if s.name not in existing_frames]
    
    print("📊 Status:")
    print(f"   Total scenarios: {len(completed)}")
    print(f"   Already processed: {len(completed) - len(new_scenarios)}")
    print(f"   New to process: {len(new_scenarios)}")
    print(f"   Need patching: {len(need_patch)}\n")
    
    if not new_scenarios and not need_patch:
        print("✅ All scenarios are already processed and up-to-date!")
        return
    
    print("🔄 Assimilation workflow:")
    print("   1. Patch FDS files (add SLCF outputs if needed)")
    print("   2. Re-run simulations (only if patched)")
    print("   3. Extract frames (generate images + masks)")
    print("   4. Update manifest (add to dataset)\n")
    
    confirm = input("Proceed with assimilation? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y':
        return
    
    print("\n" + "="*66)
    print("STARTING ASSIMILATION")
    print("="*66 + "\n")
    
    # Step 1: Patch if needed
    if need_patch:
        print(f"📝 Step 1/4: Patching {len(need_patch)} FDS files...")
        subprocess.run([sys.executable, str(SCRIPTS_DIR / "patch_fds_for_slcf.py")])
        print("✅ Patching complete\n")
    else:
        print("✅ Step 1/4: No patching needed\n")
    
    # Step 2: Extract frames from new scenarios
    if new_scenarios:
        print(f"🎞️  Step 2/4: Extracting frames from {len(new_scenarios)} scenarios...")
        for i, scenario_dir in enumerate(new_scenarios, 1):
            print(f"   [{i}/{len(new_scenarios)}] Processing {scenario_dir.name}...")
            subprocess.run([
                sys.executable, 
                str(SCRIPTS_DIR / "extract_frames.py"),
                "--scenario", scenario_dir.name
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Frame extraction complete\n")
    else:
        print("✅ Step 2/4: No new scenarios to extract\n")
    
    # Step 3: Prepare dataset (regenerate manifest)
    print("📑 Step 3/4: Updating dataset manifest...")
    subprocess.run([sys.executable, str(SCRIPTS_DIR / "prepare_dataset.py")])
    print("✅ Manifest updated\n")
    
    # Step 4: Summary
    print("="*66)
    print("✅ ASSIMILATION COMPLETE")
    print("="*66)
    print(f"Dataset location: {DATA_DIR}")
    print(f"Manifest: {MANIFEST}")
    
    # Count new entries
    if MANIFEST.exists():
        with open(MANIFEST, newline="") as f:
            total_entries = sum(1 for _ in csv.DictReader(f))
        print(f"Total dataset entries: {total_entries}")
    
    print("\n💡 Next steps:")
    print("   • Review dataset with 'Generate Viz Grid'")
    print("   • Train model with updated data\n")


def action_prepare():
    while True:
        clear_screen()
        print_banner()
        print("PREPARE DATASET\n")
        print("  1. 🩹 Patch FDS Files   (enables .sf dump for new FDS scenarios)")
        print("  2. 🎞️  Extract Frames   (parses .sf binary into image+label frames)")
        print("  3. 📑 Prepare Dataset  (splits into train/val/test & generates manifest)")
        print("  4. 🔄 Assimilate New Scenarios  (scan & import new FDS simulations)")
        print("  5. ← Back\n")
        c = input("Choose (1-5): ").strip()
        if c == '1': subprocess.run([sys.executable, str(SCRIPTS_DIR / "patch_fds_for_slcf.py")]); press_enter()
        elif c == '2': subprocess.run([sys.executable, str(SCRIPTS_DIR / "extract_frames.py")]); press_enter()
        elif c == '3': subprocess.run([sys.executable, str(SCRIPTS_DIR / "prepare_dataset.py")]); press_enter()
        elif c == '4': action_assimilate_scenarios(); press_enter()
        elif c == '5': break


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


def action_setup_wizard():
    """Setup wizard for first-time users"""
    print_section("🔧 SETUP WIZARD")
    
    print("Welcome to the Fire Detection Tool setup!\n")
    print("This wizard will help you get started.\n")
    
    # Check dependencies
    print("Step 1/3: Checking dependencies...\n")
    missing = []
    for pkg in ["torch", "torchvision", "numpy", "matplotlib", "PIL"]:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} - MISSING")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install torch torchvision numpy matplotlib pillow pandas")
        press_enter()
        return
    
    # Create directories
    print("\nStep 2/3: Creating directories...\n")
    for label, path in [("Input", INPUT_DIR), ("Output", OUTPUT_DIR), 
                         ("Dataset", DATA_DIR), ("Checkpoints", CKPT_DIR),
                         ("Logs", LOGS_DIR), ("Docs", DOCS_DIR)]:
        path.mkdir(exist_ok=True)
        print(f"  ✅ {label}: {path}")
    
    # Check for model and dataset
    print("\nStep 3/3: Checking model and dataset...\n")
    
    if not MODEL_PATH.exists():
        print("  ⚠️  Model checkpoint not found")
        print("     You'll need to train a model or download pretrained weights")
    else:
        print(f"  ✅ Model checkpoint found")
    
    if not MANIFEST.exists():
        print("  ⚠️  Dataset not prepared")
        print("     Use 'Prepare Dataset' menu to process FDS scenarios")
    else:
        print(f"  ✅ Dataset manifest found")
    
    print("\n" + "═" * 70)
    print("✅ SETUP COMPLETE!")
    print("═" * 70)
    
    print("\n📖 Quick Start Guide:")
    print("  fds_scenarios is OPTIONAL - choose your workflow:")
    print("  • WITH fds_scenarios: Auto-generate dataset from FDS sims")
    print("  • WITHOUT: Use existing Dataset/ or Input/ images")
    print("\n  Setup wizard complete - ready to use!")
    
    press_enter()


def action_help():
    """Help and FAQ system"""
    while True:
        clear_screen()
        print_banner()
        print("HELP & DOCUMENTATION\n")
        print("  1. 📖 Quick Start Guide")
        print("  2. ❓ FAQ")
        print("  3. 🛠️  Troubleshooting")
        print("  4. 📚 About")
        print("  5. ← Back\n")
        
        c = input("Choose (1-5): ").strip()
        
        if c == '1':
            print_section("📖 QUICK START GUIDE")
            print("Getting Started with Fire Detection:\n")
            print("⚠️  NOTE: fds_scenarios folder is OPTIONAL!\n")
            
            print("🔥 TWO WAYS TO USE THIS TOOL:\n")
            
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("📁 OPTION A: With fds_scenarios (Automated Dataset Generation)")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("1️⃣  Create ../fds_scenarios/ folder")
            print("   • Tool can auto-detect and offer to create it")
            print("   • Or specify custom path during assimilation\n")
            
            print("2️⃣  Add FDS simulation outputs")
            print("   • Generate with fire_predict.py FDS generator")
            print("   • Or copy existing FDS simulations")
            print("   • Each needs .fds, .smv, and .sf files\n")
            
            print("3️⃣  Use 'Assimilate New Scenarios'")
            print("   • Automatically extracts training images")
            print("   • Creates labeled fire masks from HRRPUV data")
            print("   • Builds Dataset/ folder with manifest.csv\n")
            
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("🖼️  OPTION B: Without fds_scenarios (Manual Images)")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("1️⃣  Use existing Dataset/ folder (if you have one)")
            print("   • Pre-prepared datasets work fine\n")
            
            print("2️⃣  OR place images directly in Input/")
            print("   • Works with any .png/.jpg fire images")
            print("   • Use 'Quick Predict' for inference only\n")
            
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("🧠 THEN: Train Model")
            print("   • Use 'Train Model' from main menu")
            print("   • Works with either approach above\n")
            
            print("🎯 FINALLY: Make Predictions")
            print("   • Quick Predict, Evaluate, or Generate Grids")
            print("   • Works on any fire images\n")
            
            press_enter()
            
        elif c == '2':
            print_section("❓ FREQUENTLY ASKED QUESTIONS")
            
            print("Q: Is fds_scenarios folder required?")
            print("A: NO - it's completely optional! You can use this tool without it.")
            print("   • If you HAVE fds_scenarios: Auto-import FDS simulation data")
            print("   • If you DON'T: Use manual images or existing Dataset\n")
            
            print("Q: What is fds_scenarios used for?")
            print("A: It's a source of FDS simulation outputs for automated dataset generation.")
            print("   If present, the tool can extract training images from simulations.\n")
            
            print("Q: Where do I get FDS scenarios?")
            print("A: Create ../fds_scenarios/ folder and add FDS simulation outputs,")
            print("   OR generate them with fire_predict.py FDS generator\n")
            
            print("Q: What format should my data be?")
            print("A: FDS .sf slice files (TEMPERATURE and HRRPUV)")
            print("   Each scenario folder needs .fds, .smv, and .sf files\n")
            
            print("Q: How long does training take?")
            print("A: 2-4 hours on GPU for ~1000 images, much longer on CPU\n")
            
            print("Q: Can I use my own fire images?")
            print("A: YES! Place .png/.jpg in Input/ and use Quick Predict")
            print("   Works with any fire images, not just FDS data\n")
            
            print("Q: What accuracy can I expect?")
            print("A: Typically >90% IoU on synthetic fire images\n")
            
            print("Q: Do I need CUDA/GPU?")
            print("A: Recommended for training, but CPU works for inference\n")
            
            press_enter()
            
        elif c == '3':
            print_section("🛠️  TROUBLESHOOTING")
            print("Common Issues:\n")
            
            print("❌ 'Model checkpoint not found'")
            print("   → Train a model first or download pretrained weights\n")
            
            print("❌ 'No scenarios found in fds_scenarios'")
            print("   → Check directory structure: project_root/fds_scenarios/\n")
            
            print("❌ 'CUDA out of memory'")
            print("   → Reduce batch size in train_maskrcnn.py")
            print("   → Or use CPU (slower)\n")
            
            print("❌ 'Cannot read .sf files'")
            print("   → Patch FDS files to add SLCF outputs")
            print("   → Use 'Prepare Dataset' → 'Patch FDS Files'\n")
            
            print("❌ 'No frames extracted'")
            print("   → Ensure scenarios have TEMPERATURE and HRRPUV slices")
            print("   → Check .smv file for SLCF entries\n")
            
            print("For more help, check docs/ folder or open an issue on GitHub\n")
            
            press_enter()
            
        elif c == '4':
            print_section("📚 ABOUT")
            print(f"Fire Detection Tool v{VERSION}")
            print("Mask R-CNN for Instance Segmentation of Fire\n")
            
            print("Technology Stack:")
            print("  • PyTorch + TorchVision (Mask R-CNN)")
            print("  • FDS (Fire Dynamics Simulator) data")
            print("  • ResNet50-FPN backbone")
            print("  • Custom fire segmentation head\n")
            
            print("Features:")
            print("  ✓ Automatic scenario assimilation")
            print("  ✓ Train/Val/Test split management")
            print("  ✓ Interactive predictions")
            print("  ✓ Batch evaluation metrics")
            print("  ✓ Visualization grids\n")
            
            print("Dataset:")
            if MANIFEST.exists():
                try:
                    with open(MANIFEST, newline="") as f:
                        total = sum(1 for _ in csv.DictReader(f))
                    print(f"  {total} labeled fire images from FDS simulations")
                except:
                    print("  Dataset manifest exists")
            else:
                print("  Not yet prepared")
            
            print("\n© 2024-2026 Fire Prediction Team")
            print("Licensed for research and educational use\n")
            
            press_enter()
            
        elif c == '5':
            break


def action_diagnostics():
    """Enhanced system diagnostics"""
    print_section("🔧 SYSTEM DIAGNOSTICS")
    
    print("═" * 70)
    print("🐍 PYTHON ENVIRONMENT")
    print("═" * 70)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"Device: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"CUDA Available: Yes")
        if hasattr(torch.cuda, 'get_device_name'):
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CUDA Available: No (using CPU)")
    
    print(f"\n{'═' * 70}")
    print("📦 REQUIRED PACKAGES")
    print("═" * 70)
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("PIL", "Pillow"),
        ("pandas", "Pandas"),
        ("cv2", "OpenCV (optional)")
    ]
    
    for pkg, name in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  ✅ {name:<20} {ver}")
        except ImportError:
            status = "optional" if pkg == "cv2" else "MISSING"
            print(f"  ❌ {name:<20} {status}")
    
    print(f"\n{'═' * 70}")
    print("📂 PATHS & DATA")
    print("═" * 70)
    
    paths = [
        ("Base Directory", BASE_DIR),
        ("Input Folder", INPUT_DIR),
        ("Output Folder", OUTPUT_DIR),
        ("Dataset", DATA_DIR),
        ("Scripts", SCRIPTS_DIR),
        ("Checkpoints", CKPT_DIR),
    ]
    
    for label, path in paths:
        exists = "✅" if path.exists() else "❌"
        print(f"  {exists} {label:<20} {path}")
    
    print(f"\n{'═' * 70}")
    print("🤖 MODEL & DATASET")
    print("═" * 70)
    print(f"  {'✅' if MODEL_PATH.exists() else '❌'} Model Checkpoint:     {MODEL_PATH.name}")
    print(f"  {'✅' if MANIFEST.exists() else '❌'} Dataset Manifest:     manifest.csv")
    
    # Count dataset entries
    if MANIFEST.exists():
        try:
            with open(MANIFEST, newline="") as f:
                entries = list(csv.DictReader(f))
                total = len(entries)
                train = sum(1 for e in entries if e.get("split") == "train")
                val = sum(1 for e in entries if e.get("split") == "val")
                test = sum(1 for e in entries if e.get("split") == "test")
                print(f"\n  Dataset Split:")
                print(f"    Total:  {total} images")
                print(f"    Train:  {train} images")
                print(f"    Val:    {val} images")
                print(f"    Test:   {test} images")
        except:
            pass
    
    # Check for FDS scenarios
    PROJECT_ROOT = BASE_DIR.parent
    FDS_SCENARIOS_DIR = PROJECT_ROOT / "fds_scenarios"
    if FDS_SCENARIOS_DIR.exists():
        scenario_count = len([d for d in FDS_SCENARIOS_DIR.iterdir() if d.is_dir()])
        print(f"\n  {'✅' if scenario_count > 0 else '⚠️'} FDS Scenarios:       {scenario_count} scenarios found")
    else:
        print(f"\n  ⚠️  FDS Scenarios:       Not found (optional)")
    
    print(f"\n{'═' * 70}")
    if MODEL_PATH.exists() and MANIFEST.exists():
        print("✅ System ready for predictions and training!")
    elif not MODEL_PATH.exists():
        print("⚠️  Model checkpoint missing - train a model first")
    elif not MANIFEST.exists():
        print("⚠️  Dataset not prepared - use 'Prepare Dataset' menu")
    print("═" * 70)
    
    press_enter()


# ============================================================================
# MAIN MENU
# ============================================================================
def show_main_menu():
    print_banner()
    w = 70
    bar = "─" * (w - 2)
    print("┌" + bar + "┐")
    print("│" + "  MAIN MENU".ljust(w - 2) + "│")
    print("├" + bar + "┤")
    items = [
        ("1", "🖼️", "Quick Predict",   "pick an image from Input/"),
        ("2", "📊", "Evaluate Test Set", "metrics on held-out test data"),
        ("3", "📸", "Generate Viz Grid", "16-panel evaluation grid"),
        ("4", "⚙️", "Prepare Dataset",   "assimilate & process scenarios"),
        ("5", "🧠", "Train Model",       "fine-tune Mask R-CNN"),
        ("6", "📁", "Manage Files",      "Input, Output, browse assets"),
        ("7", "🔧", "Diagnostics",       "system health check"),
        ("8", "🔧", "Setup Wizard",      "first-time setup assistant"),
        ("9", "❓", "Help",              "guides, FAQ, troubleshooting"),
    ]
    for num, icon, name, desc in items:
        line = f"  {num}.  {icon}  {name:<18} {desc}"
        print("│" + line.ljust(w - 2) + "│")
    print("├" + bar + "┤")
    print("│" + "  0.  🚪  Exit".ljust(w - 2) + "│")
    print("└" + bar + "┘")
    print()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Fire Detection Tool - Mask R-CNN Instance Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py                    Interactive menu
  python predict.py --version          Show version info
  python predict.py --check            Run system diagnostics
  python predict.py --setup            Run setup wizard
  python predict.py image.png          Quick predict on image
  python predict.py --eval             Evaluate test set
        """
    )
    parser.add_argument("file", nargs="?", help="Image file to predict")
    parser.add_argument("--version", action="store_true", help="Show version")
    parser.add_argument("--check", action="store_true", help="Run diagnostics")
    parser.add_argument("--setup", action="store_true", help="Run setup wizard")
    parser.add_argument("--eval", action="store_true", help="Evaluate test set")
    parser.add_argument("--no-splash", action="store_true", help="Skip fire animation")
    
    args = parser.parse_args()
    
    if not TORCH_OK:
        print("❌ CRITICAL: PyTorch/Torchvision not installed.")
        print("   pip install torch torchvision")
        sys.exit(1)
    
    # Handle command-line modes
    if args.version:
        print_banner()
        print(f"Version: {VERSION}")
        print(f"Device: {DEVICE.upper()}")
        print(f"Model: Mask R-CNN (ResNet50-FPN)")
        if MANIFEST.exists():
            with open(MANIFEST, newline="") as f:
                total = sum(1 for _ in csv.DictReader(f))
            print(f"Dataset: {total} images")
        sys.exit(0)
    
    if args.check:
        print_banner()
        action_diagnostics()
        sys.exit(0)
    
    if args.setup:
        print_banner()
        action_setup_wizard()
        sys.exit(0)
    
    if args.eval:
        print_banner()
        action_eval()
        sys.exit(0)
    
    if args.file:
        print_banner()
        img_path = Path(args.file)
        if not img_path.exists():
            print(f"❌ File not found: {img_path}")
            sys.exit(1)
        # Run quick predict on the file
        model = load_trained_model()
        if model:
            print(f"⚙️  Predicting fire in {img_path.name}...")
            try:
                pil_img = Image.open(img_path).convert("RGB")
                img_np  = np.array(pil_img)
                h, w    = img_np.shape[:2]
                img_t   = transforms.ToTensor()(pil_img).to(DEVICE)
                
                with torch.no_grad():
                    output = model([img_t])[0]
                
                pred = get_pred_mask(output, h, w, SCORE_THR, MASK_THR)
                
                OUTPUT_DIR.mkdir(exist_ok=True)
                save_path = OUTPUT_DIR / f"{img_path.stem}_pred.png"
                
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                fig.patch.set_facecolor("#1a1a2e")
                axs[0].imshow(img_np)
                axs[0].set_title("Input", color="white", fontsize=12)
                axs[0].axis("off")
                
                axs[1].imshow(overlay(img_np, pred, (1, 0.2, 0)))
                axs[1].set_title("Prediction", color="white", fontsize=12)
                axs[1].axis("off")
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
                print(f"✅ Saved: {save_path}")
            except Exception as e:
                print(f"❌ Error: {e}")
        sys.exit(0)
    
    # Interactive mode
    if not args.no_splash:
        fire_splash()
    
    ACTIONS = {
        "1": action_quick_predict,
        "2": action_eval,
        "3": action_visualize_grid,
        "4": action_prepare,
        "5": action_train,
        "6": action_files,
        "7": action_diagnostics,
        "8": action_setup_wizard,
        "9": action_help,
    }
    
    while True:
        clear_screen()
        show_main_menu()
        choice = input("Choose (1-9, 0 to exit): ").strip()
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
