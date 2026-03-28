"""
laplace_hamming_binarization.py
================================
Laplace-Hamming frequency-domain binarization of Scanco HR-pQCT AIM files.

This script replicates Scanco's IPL Laplace-Hamming bone segmentation workflow
in Python, producing binary bone masks equivalent to IPL's seg_gauss / LH
pipeline. It is part of the BQRL open-source HR-pQCT analysis pipeline.

Pipeline
--------
1. Load grayscale AIM file (Scanco XtremeCT II, 60.7 µm isotropic)
2. Apply Laplace-Hamming frequency-domain sharpening filter
3. Binarize using IPL-equivalent intensity threshold
4. Remove isolated islands < 70 voxels (6-connectivity)
5. Restrict to periosteal (filled bone) mask

Outputs
-------
<basename>_LH_Binary.nii.gz   Binarized bone mask (within periosteal ROI)
<basename>_LH_preview.png     2-panel axial preview image

Dependencies
------------
    numpy, scipy, SimpleITK, itk (with ScancoImageIO), matplotlib

Usage
-----
Edit the PATHS section below and run in an IDE (e.g. Spyder), or pass
arguments on the command line:

    python laplace_hamming_binarization.py  \\
        scan.AIM output_dir/               \\
        bone_mask.nii.gz                   \\
        cort_mask.nii.gz trab_mask.nii.gz

References
----------
Filter parameters and binarization approach:
    Sadoughi S., Subramanian A., Ramil G., Burghardt A.J., Kazakia G.J. (2023).
    A Laplace-Hamming Binarization Approach for Second-Generation HR-pQCT
    Rescues Fine Feature Segmentation. J Bone Miner Res, 38(7), 1006-1014.
    https://doi.org/10.1002/jbmr.4819

IPL threshold calibration:
    Scanco Medical IPL reference values (seg_gauss, low_th = 1170 mgHA/cm³).
    Constants recalibrated from 3-point IPL intensity mapping (min/max/mean)
    to match the open-source pipeline's native unit space.

Authors
-------
    Yihua Zhu - Kazakia Lab, UCSF Musculoskeletal Quantitative Imaging Research
    https://github.com/zhuyihua1234
"""

import os
import argparse
import numpy as np
import itk
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")            # change to "TkAgg" for interactive display
import matplotlib.pyplot as plt  # (e.g. Spyder / Jupyter)
from scipy.ndimage import label


# ══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION  —  edit paths here for IDE / Spyder use
# ══════════════════════════════════════════════════════════════════════════════

INPUT_AIM_PATH        = r"D:\Research\Kazakia_Lab\PFJOA\XCT_masks\PFJ016_R\C0002398_version1.AIM"
OUTPUT_DIR            = r"D:\Research\Kazakia_Lab\PFJOA\XCT_masks\PFJ016_R\ormir outputs"

# Output of the auto-contour step
FILLED_BONE_MASK_PATH = r"D:\Research\Kazakia_Lab\PFJOA\XCT_masks\PFJ016_R\ormir outputs\C0002398_version1_PRX_mask.nii.gz"



# ══════════════════════════════════════════════════════════════════════════════
# ITK → SimpleITK CONVERSION UTILITY
# Inlined from ormir_xct.util.sitk_itk to avoid the ormir_xct dependency.
# ══════════════════════════════════════════════════════════════════════════════

def itk_sitk(itk_image):
    """Convert an ITK image to a SimpleITK image, preserving spatial metadata."""
    import itk
    array  = itk.array_view_from_image(itk_image)
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image.SetSpacing(   [float(s) for s in itk_image.GetSpacing()])
    sitk_image.SetOrigin(    [float(o) for o in itk_image.GetOrigin()])
    sitk_image.SetDirection( itk.array_from_matrix(itk_image.GetDirection()).flatten().tolist())
    return sitk_image


# ══════════════════════════════════════════════════════════════════════════════
# FILTER PARAMETERS
# Matched to Scanco IPL defaults for XtremeCT II at 60.7 µm.
# Do not change unless working with a different scanner / protocol.
# ══════════════════════════════════════════════════════════════════════════════

EL_SIZE_MM      = np.array([0.0607, 0.0607, 0.0607])  # voxel size (mm)
LP_CUT_OFF_FREQ = 0.3    # low-pass cutoff as fraction of max physical frequency
LAPLACE_EPS     = 0.45   # Laplacian sharpening epsilon
AMPLIFICATION   = 1.0    # overall filter gain
HP_CUT_OFF_FREQ = 0.0    # high-pass cutoff (0 = disabled)
HAMMING_AMP     = 1.0    # Hamming window amplitude


# ══════════════════════════════════════════════════════════════════════════════
# BINARIZATION THRESHOLD
#
# Equivalent to IPL seg_gauss lower threshold = 1170 mgHA/cm³.
#
# The IPL intensity pipeline:
#   scaled = clip(A * filtered + B, None, 200000) * (32768 / 200000)
#   bone   = scaled >= LH_THRESHOLD
#
# A and B are calibrated via a 3-point IPL intensity mapping (min/max/mean)
# derived from the IPL reference output for this scanner and protocol:
#   IPL min = -173102.125,  max = 396732.9375,  mean = 66682.539
#
# To recalibrate for a different scanner or protocol, record the filtered_ipl
# statistics from an IPL reference run and solve:
#   A = (ipl_max - ipl_min) / (filtered_max - filtered_min)
#   B = ipl_min - A * filtered_min
# ══════════════════════════════════════════════════════════════════════════════

IPL_SCALE_A   = 77.7911       # calibrated slope
IPL_SCALE_B   = -1359190.17   # calibrated intercept
IPL_FLOAT_MAX = 200000.0      # clip ceiling (IPL D3P_Std_FloatNormMax_M)
INT16_MAX     = 32768.0       # int16 scale denominator
LH_THRESHOLD  = 15564         # bone threshold in scaled IPL units (unchanged)


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

CC_MIN_VOXELS = 70            # remove connected components smaller than this
CC_STRUCT_6   = np.array(     # 6-connectivity structuring element
    [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
     [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
     [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_lh_binarization(
    input_aim_path,
    output_dir,
    filled_bone_mask_path,
    save_preview=True,
):
    """
    Run the full Laplace-Hamming binarization pipeline on one AIM file.

    Parameters
    ----------
    input_aim_path        : str   Path to the Scanco AIM grayscale image.
    output_dir            : str   Directory for output NIfTI files.
    filled_bone_mask_path : str   Periosteal (filled bone) mask NIfTI.
    save_preview          : bool  Save a 5-panel PNG preview (default True).

    Returns
    -------
    dict with key 'lh_binary' — boolean numpy array (Z, Y, X).
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_aim_path))[0]

    out_binary = os.path.join(output_dir, basename + "_LH_Binary.nii.gz")
    out_png    = os.path.join(output_dir, basename + "_LH_preview.png")

    # ── Step 1: Load AIM ─────────────────────────────────────────────────────
    print(f"[1/7] Loading: {input_aim_path}")
    image_type = itk.Image[itk.ctype("signed short"), 3]
    reader     = itk.ImageFileReader[image_type].New()
    image_io   = itk.ScancoImageIO.New()
    reader.SetImageIO(image_io)
    reader.SetFileName(input_aim_path)
    reader.Update()

    sitk_image = itk_sitk(reader.GetOutput())
    pixels     = sitk.GetArrayFromImage(sitk_image).astype(np.float64)  # (Z,Y,X)
    print(f"    shape  : {pixels.shape}")
    print(f"    int16  : [{pixels.min():.0f}, {pixels.max():.0f}]")

    # Offset to uint16 space — matches the unit space the IPL calibration
    # constants were derived from (Scanco native int16 + 32768).
    pixels_u16 = pixels + 32768.0

    # ── Step 2: Laplace-Hamming filter ───────────────────────────────────────
    print("[2/7] Applying Laplace-Hamming filter...")
    fft  = np.fft.fftshift(np.fft.fftn(pixels_u16.astype(np.complex128)))
    dim  = np.array(pixels_u16.shape[::-1])   # (X, Y, Z)
    phys = dim * EL_SIZE_MM

    max_freq = 1.0 / np.min(EL_SIZE_MM)
    lp_freq2 = (max_freq * LP_CUT_OFF_FREQ) ** 2
    hp_freq2 = (max_freq * HP_CUT_OFF_FREQ) ** 2
    origin   = dim // 2

    posz, posy, posx = np.mgrid[0:dim[2], 0:dim[1], 0:dim[0]]
    freq2 = (
        ((posx - origin[0]) / phys[0]) ** 2 +
        ((posy - origin[1]) / phys[1]) ** 2 +
        ((posz - origin[2]) / phys[2]) ** 2
    )

    band   = (freq2 <= lp_freq2) & (freq2 >= hp_freq2)
    kernel = (
        AMPLIFICATION
        * (1.0 + LAPLACE_EPS * (freq2 - 1.0))
        * (1.0 + (HAMMING_AMP / 2.0)
           * (np.cos(np.pi * np.sqrt(freq2 / lp_freq2)) - 1.0))
    )

    fft_filt       = np.zeros_like(fft)
    fft_filt[band] = fft[band] * kernel[band]
    filtered       = np.real(np.fft.ifftn(np.fft.ifftshift(fft_filt)))
    print(f"    filtered: [{filtered.min():.1f}, {filtered.max():.1f}]")

    # ── Step 3: Binarize (IPL intensity pipeline) ─────────────────────────────
    print(f"[3/7] Binarizing (LH_THRESHOLD = {LH_THRESHOLD})...")
    ipl    = np.clip(IPL_SCALE_A * filtered + IPL_SCALE_B, None, IPL_FLOAT_MAX)
    scaled = ipl * (INT16_MAX / IPL_FLOAT_MAX)
    binary = scaled >= LH_THRESHOLD
    print(f"    foreground before CC: {binary.sum():,}")

    # ── Step 4: Connected-components cleanup ──────────────────────────────────
    print(f"[4/7] Connected-components (6-conn, min {CC_MIN_VOXELS} voxels)...")
    labeled, n_feat = label(binary, structure=CC_STRUCT_6)
    sizes           = np.bincount(labeled.ravel())
    small           = np.where(sizes < CC_MIN_VOXELS)[0]
    small           = small[small != 0]
    binary_cc       = binary & ~np.isin(labeled, small)
    print(f"    components : {n_feat:,}")
    print(f"    removed    : {binary.sum() - binary_cc.sum():,}")
    print(f"    foreground : {binary_cc.sum():,}")

    # ── Step 5: Periosteal mask ────────────────────────────────────────────────
    print("[5/7] Applying periosteal (filled bone) mask...")
    bone_arr  = _load_mask(filled_bone_mask_path)
    _check_shape(binary_cc, bone_arr, "bone mask")
    lh_binary = binary_cc & bone_arr
    print(f"    LH_Binary  : {lh_binary.sum():,}")

    # ── Step 6: Save output ───────────────────────────────────────────────────
    print("[6/6] Saving output...")
    _save_mask(lh_binary, sitk_image, out_binary)

    if save_preview:
        _save_preview(pixels, lh_binary, basename, out_png)

    print("Done.")
    return {"lh_binary": lh_binary}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_mask(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(bool)


def _check_shape(ref, arr, name):
    if arr.shape != ref.shape:
        raise ValueError(
            f"Shape mismatch: reference {ref.shape} vs {name} {arr.shape}"
        )


def _save_mask(arr, ref_image, path):
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.CopyInformation(ref_image)
    sitk.WriteImage(out, path)
    print(f"    → {path}")


def _save_preview(pixels, lh_binary, basename, out_png):
    """Save a 2-panel axial PNG at the slice with the most foreground voxels."""
    counts = lh_binary.sum(axis=(1, 2))
    z      = int(np.argmax(counts)) if counts.max() > 0 else pixels.shape[0] // 2

    pos  = pixels[pixels > 0]
    vmin = float(np.percentile(pos,  1)) if pos.size else 0.0
    vmax = float(np.percentile(pos, 99)) if pos.size else float(pixels.max())

    def blend(gray, mask, rgb, alpha=0.5):
        g   = np.clip((gray - vmin) / (vmax - vmin + 1e-9), 0, 1)
        out = np.stack([g, g, g], axis=-1).copy()
        for c, v in enumerate(rgb):
            out[mask, c] = out[mask, c] * (1 - alpha) + v * alpha
        return out

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"{basename}  —  axial slice {z} / {pixels.shape[0]}", fontsize=13)

    axes[0].imshow(pixels[z], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Raw image")
    axes[0].axis("off")

    axes[1].imshow(blend(pixels[z], lh_binary[z], (1.0, 0.2, 0.0)))
    axes[1].set_title("LH Binary")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"    → {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Laplace-Hamming binarization of Scanco HR-pQCT AIM files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("aim",        help="Input AIM grayscale file")
    p.add_argument("output_dir", help="Output directory")
    p.add_argument("bone_mask",  help="Filled bone (periosteal) mask NIfTI")

    p.add_argument("--no-preview", action="store_true",
                   help="Skip saving the preview PNG")
    return p.parse_args()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Command-line mode
        args = _parse_args()
        run_lh_binarization(
            input_aim_path        = args.aim,
            output_dir            = args.output_dir,
            filled_bone_mask_path = args.bone_mask,
            save_preview          = not args.no_preview,
        )
    else:
        # IDE / Spyder mode — uses hardcoded paths at top of file
        run_lh_binarization(
            input_aim_path        = INPUT_AIM_PATH,
            output_dir            = OUTPUT_DIR,
            filled_bone_mask_path = FILLED_BONE_MASK_PATH,
        )
