# FITS Quick Viewer
 
A fast, lightweight quick-look viewer for astrophotography FITS files. Open a linear 32-bit sub, see it debayered and auto-stretched, read the header, and get instant quality metrics (HFR, FWHM, eccentricity, star count, noise, gradient…) with a **GOOD / FAIR / POOR** verdict, so you can cull bad subs in seconds.
 
Single file, no build step.
 
## Features
 
- **Opens** 32-bit float, 16-bit and 32-bit integer FITS: mono, one-shot-colour (Bayer) and RGB cubes, plus `.fz` compressed files
- **Automatic debayering** from `BAYERPAT`, `ROWORDER` and Bayer offset keywords, with a manual override (RGGB / BGGR / GRBG / GBRG / Mono)
- **Auto-stretch** in the style of PixInsight's STF (median/MAD based), with linked/unlinked channels and an adjustable target background
- **FITS header table** with a live search filter
- **Quality metrics**
  - HFR (median and spread), FWHM (px and arcsec), eccentricity
  - Star count and saturated star count
  - Background level, noise, star SNR
  - Background gradient and saturated pixel percentage
  - 3×3 map of HFR across the field to reveal tilt or field curvature
  - Verdict with the specific reasons (clouds, trailing, bad focus, gradient…)
- **Star overlay** showing exactly which stars were measured
- **BAD_ button**: renames a sub to `BAD_<name>.fits` (and back) for easy filtering, optionally jumping to the next file
- **Folder → CSV** export of all metrics, analysed in parallel, for sorting subs in a spreadsheet
- **Fast**: image on screen in about 1 s for a 24 MP frame, next file pre-analysed in the background so stepping through subs is nearly instant
## Installation
 
Requires Python 3.9+ (tkinter is included with standard Python installers; on Debian/Ubuntu: `sudo apt install python3-tk`).
 
```bash
pip install numpy scipy astropy pillow
```
or

```bash
pip install -r requirements.txt
```
 
## Usage
 
```bash
python fits_viewer.py                 # then File → Open
python fits_viewer.py path/to/sub.fits
```
 
### Controls
 
| Action | Control |
|---|---|
| Previous / next file in folder | `←` / `→` |
| Mark / unmark as BAD | `B` or `Delete` (or the toolbar button) |
| Zoom | Mouse wheel |
| Pan | Drag |
| Fit to window | `F` or double-click |
| 100 % zoom | `1` |
| Open file | `Ctrl+O` |
 
### Culling workflow
 
1. Open the first sub in a folder.
2. Step with `→`, check the verdict and the star overlay.
3. Press `B` on anything bad. It is renamed with a `BAD_` prefix and the viewer moves on.
4. Filter or move the `BAD_*` files in your file manager or stacking software. Press `B` again on a BAD_ file to undo.
For a whole-session overview, use **File → Analyse folder → CSV…** and sort by HFR, eccentricity or star count.
 
## How the metrics work
 
- **HFR**: flux-weighted mean radius of each star within a 14 px aperture (the same definition used by NINA and similar tools), after local background removal. Hot pixels and saturated stars are excluded. Values will not match other software exactly, so compare frames within the tool.
- **FWHM / eccentricity**: elliptical Gaussian fit on the 60 brightest unsaturated stars.
- **Plate scale / arcsec FWHM**: from `XPIXSZ` and `FOCALLEN`, assuming `XPIXSZ` already includes binning (as NINA and SBIG-style headers do).
- **Noise**: robust (MAD) high-pass estimate from a central crop. **Gradient**: spread of the background mesh relative to the median sky level.
- **Speed tricks**: detection runs on a 2×2-binned copy; only small patches around stars are measured at full resolution. For Bayer data the luminance is a 2×2 box average of the raw mosaic, so no debayering is needed for analysis.
### Tuning the verdict
 
Thresholds live in the `LIMITS` dict near the top of `fits_viewer.py`:
 
```python
LIMITS = dict(
    stars_poor=10, stars_warn=30,
    ecc_warn=0.55, ecc_poor=0.70,
    hfr_warn_px=4.5, fwhm_warn_arcsec=4.0,
    grad_warn=0.15, sat_warn=0.01, tilt_warn=0.25,
)
```
 
Adjust them to suit your image scale and typical seeing. The verdict is a heuristic aid, not a substitute for looking at the image.
 
## Limitations
 
- Star count is capped at 800 candidates per frame (a count near 800 means "plenty").
- Saturation is detected at 98 % of full scale. Cameras with a 14-bit ADC whose data is not scaled to 16 bits will not flag saturation.
- If a debayered image shows odd colours or a checkerboard pattern, try the other Bayer patterns in the dropdown. Capture programs differ in how they write `BAYERPAT` and `ROWORDER`.
- Very large frames can use around 1 GB of RAM while processing.
 
