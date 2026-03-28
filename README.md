# Laplace-Hamming Binarization for HR-pQCT

A Python implementation of the Laplace-Hamming (LH) frequency-domain binarization approach for second-generation Scanco HR-pQCT (XtremeCT II) images. This script produces binary bone masks that improve upon the manufacturer's standard Gaussian-based segmentation by better preserving fine trabecular and cortical features.

This implementation is part of the Bone Quality Research Lab (BQRL) open-source HR-pQCT analysis pipeline at UCSF.

---

## Background

The standard Scanco segmentation applies a Gaussian smoothing filter followed by a fixed density threshold. While straightforward, this approach tends to miss fine bone features — thin trabeculae and small cortical pores — particularly on second-generation scanners where the improved spatial resolution makes these structures resolvable. The Laplace-Hamming approach addresses this by applying a frequency-domain filter that combines low-pass smoothing with Laplacian edge sharpening, resulting in more complete segmentation of fine structures.

The filter parameters and binarization threshold used here follow those developed and validated by Sadoughi et al. (2023) specifically for the Scanco XtremeCT II at 60.7 µm isotropic voxel size.

---

## Pipeline

```
AIM grayscale image
        │
        ▼
[1] Load AIM via ITK ScancoImageIO
        │
        ▼
[2] Apply Laplace-Hamming frequency-domain filter
    (low-pass cutoff = 0.3, Laplacian ε = 0.45, Hamming amplitude = 1.0)
        │
        ▼
[3] Binarize using IPL-equivalent intensity threshold
    (equivalent to seg_gauss lower threshold = 1170 mgHA/cm³)
        │
        ▼
[4] Remove isolated islands < 70 voxels (6-connectivity)
        │
        ▼
[5] Restrict to periosteal (filled bone) mask
        │
        ▼
LH_Binary.nii.gz  +  LH_preview.png
```

---

## Requirements

| Package | Version tested | Notes |
|---|---|---|
| Python | ≥ 3.9 | |
| numpy | ≥ 1.23 | |
| scipy | ≥ 1.9 | |
| SimpleITK | ≥ 2.2 | |
| itk | ≥ 5.3 | Must include `ScancoImageIO` |
| matplotlib | ≥ 3.6 | |

The `itk` package must be built with `ScancoImageIO` support to read `.AIM` files. This is available in the ORMIR conda environment:

```bash
conda create -n ORMIR python=3.10
conda activate ORMIR
pip install itk-scancoimageio SimpleITK scipy matplotlib
```

---

## Inputs

| Input | Description |
|---|---|
| `scan.AIM` | Grayscale HR-pQCT image exported from Scanco XtremeCT II |
| `bone_mask.nii.gz` | Periosteal (filled bone) mask from the auto-contouring step |

The periosteal mask is used to restrict the LH binary output to the bone ROI, excluding any segmented tissue outside the periosteal envelope.

---

## Outputs

| File | Description |
|---|---|
| `<basename>_LH_Binary.nii.gz` | Binary bone mask (uint8, 0/1) in the periosteal ROI |
| `<basename>_LH_preview.png` | 2-panel axial preview: raw image and LH binary overlay |

All outputs are saved as NIfTI-1 (`.nii.gz`) with the spatial metadata (voxel spacing, origin, direction) copied from the input AIM file.

---

## Usage

### IDE / Spyder

Edit the `USER CONFIGURATION` section at the top of `laplace_hamming_binarization.py`:

```python
INPUT_AIM_PATH        = r"path/to/scan.AIM"
OUTPUT_DIR            = r"path/to/output/"
FILLED_BONE_MASK_PATH = r"path/to/bone_mask.nii.gz"
```

Then run the script directly.

### Command line

```bash
python laplace_hamming_binarization.py \
    scan.AIM \
    output_dir/ \
    bone_mask.nii.gz

# Skip preview PNG:
python laplace_hamming_binarization.py \
    scan.AIM output_dir/ bone_mask.nii.gz --no-preview
```

### As a module

```python
from laplace_hamming_binarization import run_lh_binarization

result = run_lh_binarization(
    input_aim_path        = "scan.AIM",
    output_dir            = "output/",
    filled_bone_mask_path = "bone_mask.nii.gz",
)

lh_binary = result["lh_binary"]  # boolean numpy array, shape (Z, Y, X)
```

---

## Filter Parameters

| Parameter | Value | Description |
|---|---|---|
| `LP_CUT_OFF_FREQ` | 0.3 | Low-pass cutoff as fraction of maximum physical frequency |
| `LAPLACE_EPS` | 0.45 | Laplacian sharpening epsilon |
| `HAMMING_AMP` | 1.0 | Hamming window amplitude |
| `AMPLIFICATION` | 1.0 | Overall filter gain |
| `LH_THRESHOLD` | 15564 | Bone threshold in scaled IPL units (≡ 1170 mgHA/cm³) |
| `CC_MIN_VOXELS` | 70 | Minimum connected component size (voxels) |

Parameters follow Sadoughi et al. (2023). Do not change these values unless working with a different scanner model or acquisition protocol.

### Recalibrating the IPL intensity constants

The constants `IPL_SCALE_A` and `IPL_SCALE_B` map the filtered image into Scanco's IPL intensity space so the threshold of 15564 applies correctly. They are calibrated from a 3-point IPL reference (min, max, mean of the filtered image in IPL units) using:

```
A = (IPL_max - IPL_min) / (filtered_max - filtered_min)
B = IPL_min - A × filtered_min
```

If you are using a different scanner or protocol, obtain the IPL reference statistics from a Scanco IPL run on a representative scan and substitute them into the equations above.

---

## Validation

This implementation was validated against the Scanco IPL reference pipeline on 21 patellofemoral joint HR-pQCT scans (Scanco XtremeCT II, 60.7 µm isotropic). Binary mask agreement was assessed using the Dice similarity coefficient:

**Dice score: 99.5% ± 0.16%**

This indicates near-identical segmentation between the open-source Python pipeline and the IPL reference, confirming that the Laplace-Hamming filter parameters and binarization threshold are faithfully reproduced.

---

## References

Sadoughi S., Subramanian A., Ramil G., Burghardt A.J., Kazakia G.J. (2023). A Laplace-Hamming Binarization Approach for Second-Generation HR-pQCT Rescues Fine Feature Segmentation. *Journal of Bone and Mineral Research*, 38(7), 1006–1014. https://doi.org/10.1002/jbmr.4819

Sadoughi S., Subramanian A., Ramil G., Zhou M., Burghardt A.J., Kazakia G.J. (2024). HR-pQCT Cross-Calibration Using Standard vs. Laplace-Hamming Binarization Approach. *JBMR Plus*, 8(10), ziae116. https://doi.org/10.1093/jbmrpl/ziae116

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

Kazakia Lab — Bone Quality Research Lab  
Department of Radiology and Biomedical Imaging  
University of California, San Francisco  
