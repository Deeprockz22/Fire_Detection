"""
patch_fds_for_slcf.py
=====================
Patches R/S/W FDS scenario files to add TEMPERATURE and HRRPUV SLCF
midplane slice outputs, then re-runs each simulation so extract_frames.py
can read the .sf output files.

This is needed because these scenario series originally used SMOKF3D (3D
volumetric compressed outputs) instead of SLCF (2D slice files), which the
SF binary parser in extract_frames.py cannot read.

Usage
-----
    python patch_fds_for_slcf.py --dry_run      # preview changes only
    python patch_fds_for_slcf.py                # patch + re-run all
    python patch_fds_for_slcf.py --scenario R1_n-heptane_medium_op50_sz72
"""

import argparse
import subprocess
import shutil
from pathlib import Path

# ==============================================================================
FDS_SCENARIOS_DIR = Path(r"D:\FDS\Small_project\fds_scenarios")
FDS_EXE = r"D:\FDS\FDS6\bin\fds_local.bat"

# Prefixes of scenarios that need patching (SMOKF3D series)
TARGET_PREFIXES = ("R", "S", "W", "EXTREME", "VALIDATION")

# SLCF lines to inject (immediately before &TAIL)
# Rooms are centred at origin so Y=0.0 is always the midplane
SLCF_LINES = [
    "&SLCF PBY=0.0, QUANTITY='TEMPERATURE' /",
    "&SLCF PBY=0.0, QUANTITY='HRRPUV' /",
]
# ==============================================================================


def needs_patching(fds_path: Path) -> bool:
    """Return True if the FDS file does not yet have a &SLCF slice output defined."""
    content = fds_path.read_text(errors="replace")
    # Check specifically for the &SLCF record — not DEVC or other records with TEMPERATURE
    return "&SLCF" not in content.upper()


def patch_fds_file(fds_path: Path, dry_run: bool = False) -> bool:
    """
    Insert SLCF lines before the &TAIL line, operating line-by-line
    so Windows CRLF endings do not corrupt the output.
    """
    lines = fds_path.read_text(errors="replace").splitlines(keepends=True)

    tail_idx = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("&TAIL"):
            tail_idx = i
            break

    slcf_block = [s + "\n" for s in SLCF_LINES]

    if tail_idx is None:
        new_lines = lines + ["\n"] + slcf_block + ["&TAIL /\n"]
    else:
        new_lines = lines[:tail_idx] + slcf_block + lines[tail_idx:]

    if not dry_run:
        backup = fds_path.with_suffix(".fds.orig")
        if not backup.exists():
            shutil.copy2(fds_path, backup)
        fds_path.write_text("".join(new_lines))

    return True


def run_fds(scenario_dir: Path, fds_file: Path) -> bool:
    """Run FDS on the patched scenario and wait for completion."""
    cmd = [FDS_EXE, fds_file.name]
    print(f"    Running FDS: {fds_file.name}")
    result = subprocess.run(cmd, cwd=scenario_dir, capture_output=True,
                            text=True, timeout=3600, shell=True)
    # Windows MPI always returns 3221225781 on exit — check for .sf files instead
    sf_files = list(scenario_dir.glob("*.sf"))
    if sf_files:
        print(f"    OK — {len(sf_files)} .sf files created")
        return True
    print(f"    FAILED — no .sf files found (exit code {result.returncode})")
    return False


def process_scenario(scenario_dir: Path, dry_run: bool = False) -> bool:
    fds_files = list(scenario_dir.glob("*.fds"))
    # Skip backup files
    fds_files = [f for f in fds_files if not f.name.endswith(".orig")]
    if not fds_files:
        print(f"  [SKIP] No .fds in {scenario_dir.name}")
        return False

    fds_path = fds_files[0]

    if not needs_patching(fds_path):
        print(f"  [SKIP] Already has SLCF: {scenario_dir.name}")
        return False

    print(f"  Patching: {scenario_dir.name}")
    patch_fds_file(fds_path, dry_run=dry_run)

    if dry_run:
        print(f"    [DRY RUN] Would re-run FDS")
        return True

    rc = run_fds(scenario_dir, fds_path)
    return rc


def main():
    parser = argparse.ArgumentParser(description="Patch FDS files with SLCF outputs and re-run simulations")
    parser.add_argument("--scenarios_dir", type=Path, default=FDS_SCENARIOS_DIR)
    parser.add_argument("--scenario",      type=str,  default=None,
                        help="Process one named scenario for testing")
    parser.add_argument("--dry_run",       action="store_true",
                        help="Show what would change without writing or running anything")
    args = parser.parse_args()

    if args.scenario:
        dirs_to_patch = [args.scenarios_dir / args.scenario]
    else:
        all_dirs = sorted([d for d in args.scenarios_dir.iterdir() if d.is_dir()])
        dirs_to_patch = [
            d for d in all_dirs
            if any(d.name.startswith(p) for p in TARGET_PREFIXES)
        ]

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Found {len(dirs_to_patch)} scenarios to patch.")
    print("=" * 60)

    ok = 0
    for d in dirs_to_patch:
        if process_scenario(d, dry_run=args.dry_run):
            ok += 1

    print(f"\nDone. {ok}/{len(dirs_to_patch)} scenarios patched and re-run.")
    if not args.dry_run:
        print("Next step: re-run extract_frames.py to generate the new frames.")


if __name__ == "__main__":
    main()
