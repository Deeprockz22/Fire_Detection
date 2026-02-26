"""
extract_frames.py  (v3 — Fixed SF Binary Parser)
==================================================
Generates paired RGB images (temperature field) and binary fire masks (HRRPUV
thresholded) from FDS simulation slice (.sf) files.

No Smokeview GUI required — reads FDS binary output files directly in Python.

Confirmed FDS .sf binary format:
  Record 1: 30-char quantity name  (e.g. b'TEMPERATURE')
  Record 2: 30-char short label    (e.g. b'temp')
  Record 3: 30-char units string   (e.g. b'C')
  Record 4: 24 bytes = 6 x int32   (i1, i2, j1, j2, k1, k2)
  Then repeating:
    Record 5: 4 bytes = float32 time
    Record 6: ni*nj*nk * 4 bytes = float32 data (Fortran order)

Usage
-----
    python extract_frames.py
    python extract_frames.py --scenario A1_small_room_opening_0
    python extract_frames.py --scenario A1_small_room_opening_0 --show_info
"""

import argparse
import struct
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.cm as cm
from PIL import Image

# ==============================================================================
# CONFIGURATION
# ==============================================================================
FDS_SCENARIOS_DIR = Path(r"D:\FDS\Small_project\fds_scenarios")
OUTPUT_DIR        = Path(r"D:\FDS\Small_project\Fire Detection\Dataset")

INPUT_QUANTITY    = "TEMPERATURE"   # used as the model input image
MASK_QUANTITY     = "HRRPUV"        # used as the binary fire mask

HRRPUV_THRESHOLD  = 50.0            # kW/m^3 — above this = fire
TEMP_MIN          = 20.0            # °C colormap min
TEMP_MAX          = 900.0           # °C colormap max

IMG_WIDTH         = 640
IMG_HEIGHT        = 480
NUM_FRAMES        = 10              # frames to extract per scenario
# ==============================================================================


# ── SMV Parser ────────────────────────────────────────────────────────────────

def parse_smv(smv_path: Path) -> list:
    """Return list of slice dicts {filename, quantity, units} from .smv file."""
    slices = []
    lines  = smv_path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("SLCF"):
            if i + 3 < len(lines):
                slices.append({
                    "filename": lines[i + 1].strip(),
                    "quantity": lines[i + 2].strip(),
                    "units":    lines[i + 3].strip(),
                })
            i += 4
        else:
            i += 1
    return slices


def find_slice(slices: list, quantity: str):
    """Find the first slice whose quantity contains the given keyword."""
    for s in slices:
        if quantity.upper() in s["quantity"].upper():
            return s
    return None


# ── SF Binary Parser ──────────────────────────────────────────────────────────

def read_sf_file(sf_path: Path):
    """
    Parse an FDS binary slice file (.sf).

    Returns
    -------
    times  : np.ndarray [N]     simulation times (s)
    frames : np.ndarray [N,H,W] 2D field values per timestep
    """
    raw    = sf_path.read_bytes()
    offset = 0

    def read_record():
        nonlocal offset
        rec_len = struct.unpack_from("<i", raw, offset)[0]
        offset += 4
        data = raw[offset: offset + rec_len]
        offset += rec_len
        offset += 4  # trailing Fortran marker
        return data

    # 3 string header records
    _ = read_record()   # quantity name
    _ = read_record()   # short label
    _ = read_record()   # units

    # Grid dims: 6 x int32 (24 bytes)
    dims_raw          = read_record()
    i1, i2, j1, j2, k1, k2 = struct.unpack_from("<6i", dims_raw[:24])
    ni = i2 - i1 + 1
    nj = j2 - j1 + 1
    nk = k2 - k1 + 1
    n_exp = ni * nj * nk

    times  = []
    frames = []

    while offset + 8 < len(raw):
        # Time record
        t_raw = read_record()
        if len(t_raw) < 4:
            break
        t = struct.unpack_from("<f", t_raw)[0]

        # Data record
        d_raw = read_record()
        if len(d_raw) < n_exp * 4:
            break

        arr3d = np.frombuffer(d_raw[:n_exp * 4], dtype="<f4").reshape(
            (ni, nj, nk), order="F"
        )
        arr2d = arr3d.squeeze()   # remove singleton axis (the slice normal)
        times.append(t)
        frames.append(arr2d)

    if not frames:
        return np.array([]), None

    return np.array(times, dtype=np.float32), np.stack(frames, axis=0)


# ── Image Generators ──────────────────────────────────────────────────────────

def save_temperature_image(arr2d: np.ndarray, out_path: Path):
    """Save a temperature 2D array as a false-colour PNG using 'hot' colormap."""
    norm   = np.clip((arr2d - TEMP_MIN) / (TEMP_MAX - TEMP_MIN), 0, 1)
    cmap   = cm.get_cmap("hot")
    rgba   = cmap(norm)
    rgb    = (rgba[:, :, :3] * 255).astype(np.uint8)
    img    = Image.fromarray(rgb).resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_mask_image(arr2d: np.ndarray, out_path: Path):
    """Save HRRPUV as a binary mask PNG (255=fire, 0=background)."""
    binary = (arr2d >= HRRPUV_THRESHOLD).astype(np.uint8) * 255
    img    = Image.fromarray(binary, mode="L").resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


# ── Scenario Processing ───────────────────────────────────────────────────────

def process_scenario(scenario_dir: Path, output_base: Path, show_info: bool = False) -> bool:
    smv_files = list(scenario_dir.glob("*.smv"))
    if not smv_files:
        print(f"  [SKIP] No .smv in {scenario_dir.name}")
        return False

    slices = parse_smv(smv_files[0])
    if show_info:
        print(f"  Quantities: {[s['quantity'] for s in slices]}")

    temp_rec = find_slice(slices, INPUT_QUANTITY)
    mask_rec = find_slice(slices, MASK_QUANTITY)

    if temp_rec is None or mask_rec is None:
        missing = []
        if temp_rec is None: missing.append(INPUT_QUANTITY)
        if mask_rec is None: missing.append(MASK_QUANTITY)
        print(f"  [SKIP] Missing slices: {missing} in {scenario_dir.name}")
        return False

    temp_sf = scenario_dir / temp_rec["filename"]
    mask_sf = scenario_dir / mask_rec["filename"]

    if not temp_sf.exists() or not mask_sf.exists():
        print(f"  [SKIP] SF file(s) missing for {scenario_dir.name}")
        return False

    temp_times, temp_data = read_sf_file(temp_sf)
    mask_times, mask_data = read_sf_file(mask_sf)

    if temp_data is None or mask_data is None or temp_data.shape[0] == 0:
        print(f"  [SKIP] Empty data in {scenario_dir.name}")
        return False

    n_frames   = min(temp_data.shape[0], mask_data.shape[0])
    frame_idxs = np.linspace(0, n_frames - 1, NUM_FRAMES, dtype=int)

    sname    = scenario_dir.name
    img_dir  = output_base / "images" / sname
    mask_dir = output_base / "masks"  / sname

    for fi in frame_idxs:
        t    = temp_times[fi]
        stem = f"frame_{fi:04d}_t{t:.1f}s.png"
        save_temperature_image(temp_data[fi], img_dir  / stem)
        save_mask_image(        mask_data[fi], mask_dir / stem)

    print(f"  ✓  {sname}: {len(frame_idxs)} frame pairs saved")
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract fire frames from FDS .sf slice files")
    parser.add_argument("--scenarios_dir", type=Path, default=FDS_SCENARIOS_DIR)
    parser.add_argument("--output_dir",    type=Path, default=OUTPUT_DIR)
    parser.add_argument("--scenario",      type=str,  default=None)
    parser.add_argument("--show_info",     action="store_true")
    args = parser.parse_args()

    if args.scenario:
        dirs = [args.scenarios_dir / args.scenario]
    else:
        dirs = sorted([d for d in args.scenarios_dir.iterdir() if d.is_dir()])

    print(f"Processing {len(dirs)} scenario(s) → {args.output_dir}")
    print("=" * 60)
    ok = sum(process_scenario(d, args.output_dir, args.show_info) for d in dirs)
    print(f"\nDone. {ok}/{len(dirs)} scenarios exported.")
    print(f"Next step: run prepare_dataset.py")


if __name__ == "__main__":
    main()
