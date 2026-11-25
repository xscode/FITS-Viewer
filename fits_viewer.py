#!/usr/bin/env python3
"""
FITS Quick Viewer
-----------------
Opens 32-bit float (or 16/32-bit int) linear FITS files, debayers OSC images if
needed, applies a PixInsight-style auto-stretch (STF), shows the FITS header and
computes subframe quality metrics (HFR, FWHM, eccentricity, star count, noise,
background gradient, saturation, tilt) with a simple GOOD / FAIR / POOR verdict.

Usage:   python fits_viewer.py [file.fits]
Needs:   pip install numpy scipy astropy pillow     (tkinter ships with Python)

Keys:    Left/Right = previous/next file in folder, B or Delete = mark/unmark file as BAD_
         (renames it with a BAD_ prefix), F = fit, 1 = 100 %, mouse wheel = zoom,
         drag = pan, double-click = fit
"""
import csv
import multiprocessing
import os
import queue
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from astropy.io import fits
from PIL import Image, ImageTk
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

EXTS = (".fit", ".fits", ".fts", ".fz")
R_AP = 14          # star aperture radius (px) for HFR
MAX_CAND = 800     # max star candidates measured per frame
FIT_STARS = 60     # Gaussian fits (FWHM/ecc) done on this many brightest stars
BAD_PREFIX = "BAD_"

# Thresholds used for the verdict - tweak to taste.
LIMITS = dict(
    stars_poor=10, stars_warn=30,
    ecc_warn=0.55, ecc_poor=0.70,
    hfr_warn_px=4.5, fwhm_warn_arcsec=4.0,
    grad_warn=0.15, sat_warn=0.01, tilt_warn=0.25,
)

_IDX = {"R": 0, "G": 1, "B": 2}


# --------------------------------------------------------------------------- #
#  Loading / stretching / fast debayer
# --------------------------------------------------------------------------- #
def load_fits(path):
    with fits.open(path, memmap=False) as hdul:
        hdu = next((h for h in hdul if h.data is not None and h.data.ndim >= 2), None)
        if hdu is None:
            raise ValueError("No image data found in file")
        data = np.squeeze(np.asarray(hdu.data))
        header = hdu.header.copy()
    if data.ndim == 3:
        if data.shape[0] in (3, 4):
            data = np.moveaxis(data, 0, -1)[..., :3]
        elif data.shape[2] in (3, 4):
            data = data[..., :3]
        else:
            data = data[0]
    data = np.ascontiguousarray(data, dtype=np.float32)
    return np.nan_to_num(data, copy=False), header


def full_scale(data):
    mx = float(np.max(data))
    if mx <= 1.5:
        return 1.0
    return 65535.0 if mx <= 65535.0 else mx


def effective_pattern(header, override="Auto"):
    """Return 4-char row-major Bayer pattern for the array as stored, or None."""
    if override == "Mono":
        return None
    if override != "Auto":
        return override
    pat = str(header.get("BAYERPAT", "")).strip().upper()
    if len(pat) != 4 or set(pat) - set("RGB"):
        return None
    rows = [pat[:2], pat[2:]]
    if str(header.get("ROWORDER", "")).upper() == "BOTTOM-UP":
        rows.reverse()

    def _int(*keys):
        for k in keys:
            try:
                return int(header[k])
            except (KeyError, ValueError, TypeError):
                pass
        return 0

    if _int("YBAYROFF", "BAYOFFY") % 2:
        rows.reverse()
    if _int("XBAYROFF", "BAYOFFX") % 2:
        rows = [r[::-1] for r in rows]
    return "".join(rows)


def debayer8(m, pat):
    """Bilinear demosaic of a uint8 mosaic -> (H,W,3) uint8. Works on quarter-size phase slices."""
    H, W = m.shape
    P = np.pad(m, 1, mode="reflect")
    out = np.empty((H, W, 3), np.uint8)
    for ry in (0, 1):
        for rx in (0, 1):
            nh, nw = (H - ry + 1) // 2, (W - rx + 1) // 2
            if nh <= 0 or nw <= 0:
                continue

            def s(dy, dx):
                return P[1 + ry + dy:1 + ry + dy + 2 * nh:2,
                         1 + rx + dx:1 + rx + dx + 2 * nw:2].astype(np.uint16)

            sub = out[ry::2, rx::2]
            X = pat[ry * 2 + rx]
            c = s(0, 0)
            if X == "G":
                A = pat[ry * 2 + (1 - rx)]       # colour of horizontal neighbours
                B = pat[(1 - ry) * 2 + rx]       # colour of vertical neighbours
                sub[..., 1] = c
                sub[..., _IDX[A]] = (s(0, -1) + s(0, 1) + 1) >> 1
                sub[..., _IDX[B]] = (s(-1, 0) + s(1, 0) + 1) >> 1
            else:
                O = "B" if X == "R" else "R"
                sub[..., _IDX[X]] = c
                sub[..., 1] = (s(0, -1) + s(0, 1) + s(-1, 0) + s(1, 0) + 2) >> 2
                sub[..., _IDX[O]] = (s(-1, -1) + s(-1, 1) + s(1, -1) + s(1, 1) + 2) >> 2
    return out


def mtf(m, x):
    return ((m - 1) * x) / (((2 * m - 1) * x) - m)


def _stats(s, fs, need_hi):
    s = np.asarray(s).ravel()
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) * 1.4826
    hi = float(np.percentile(s, 99.9)) if need_hi else 1.0
    return med / fs, mad / fs, hi / fs


def _lut(st, stretch, target, clip=-2.8, n=65536):
    med, mad, hi = st
    x = np.linspace(0, 1, n, dtype=np.float32)
    if stretch:
        c0 = min(max(med + clip * mad, 0.0), 1.0)
        m = mtf(target, max(med - c0, 1e-6))
        y = mtf(m, np.clip((x - c0) / max(1.0 - c0, 1e-6), 0, 1))
    else:
        y = np.clip(x / max(hi, 1e-6), 0, 1)
    return (y * 255 + 0.5).astype(np.uint8)


def _apply(a, lut, sc):
    idx = (a * sc).astype(np.int32)
    np.clip(idx, 0, 65535, out=idx)
    return lut[idx]


def render(data, fs, pat, stretch=True, linked=True, target=0.25):
    """Raw data -> uint8 display image. Stretch is done via LUT on the mosaic, then debayered."""
    sc = np.float32(65535.0 / fs)
    need_hi = not stretch

    def link(sts):
        if linked and len(sts) > 1:
            avg = tuple(float(np.mean([s[i] for s in sts])) for i in range(3))
            return [avg] * len(sts)
        return sts

    if data.ndim == 3:
        sts = link([_stats(data[::4, ::4, i], fs, need_hi) for i in range(3)])
        out = np.empty(data.shape, np.uint8)
        for i in range(3):
            out[..., i] = _apply(data[..., i], _lut(sts[i], stretch, target), sc)
        return out
    if pat is None:
        return _apply(data, _lut(_stats(data[::4, ::4], fs, need_hi), stretch, target), sc)

    samples = {0: [], 1: [], 2: []}
    for i, ch in enumerate(pat):
        samples[_IDX[ch]].append(data[(i // 2)::4, (i % 2)::4].ravel())
    sts = link([_stats(np.concatenate(samples[c]), fs, need_hi) for c in range(3)])
    luts = [_lut(st, stretch, target) for st in sts]
    mosaic = np.empty(data.shape, np.uint8)
    for i, ch in enumerate(pat):
        ry, rx = i // 2, i % 2
        mosaic[ry::2, rx::2] = _apply(data[ry::2, rx::2], luts[_IDX[ch]], sc)
    return debayer8(mosaic, pat)


# --------------------------------------------------------------------------- #
#  Quality analysis
# --------------------------------------------------------------------------- #
_YY, _XX = np.mgrid[-R_AP:R_AP + 1, -R_AP:R_AP + 1]
_YYf, _XXf = _YY.astype(np.float64), _XX.astype(np.float64)
_RING = np.hypot(_XXf, _YYf) > R_AP


def block_grid(img, box=8):
    """Smoothed grid of block medians (blocks of box x box on a 2x-subsampled image)."""
    sub = img[::2, ::2]
    h, w = sub.shape
    gh, gw = h // box, w // box
    if gh < 2 or gw < 2:
        return None
    blocks = sub[:gh * box, :gw * box].reshape(gh, box, gw, box).swapaxes(1, 2).reshape(gh, gw, -1)
    return ndi.gaussian_filter(np.median(blocks, axis=-1), 1.0)


def upsample(grid, shape):
    full = ndi.zoom(grid, (shape[0] / grid.shape[0], shape[1] / grid.shape[1]), order=1, mode="nearest")
    if full.shape != shape:
        full = np.pad(full, ((0, max(0, shape[0] - full.shape[0])), (0, max(0, shape[1] - full.shape[1]))),
                      mode="edge")[:shape[0], :shape[1]]
    return full


def box2(a):
    """2x2 box sum/4 -> output is (h-1, w-1). Neutral luminance for a Bayer mosaic."""
    return (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:]) * 0.25


def measure_star(src, x, y, noise, inv, cfa, satthr):
    R = R_AP
    if cfa:
        rp = src[y - R:y + R + 2, x - R:x + R + 2]
        p = box2(rp).astype(np.float64) * inv
        off = 0.5
    else:
        p = src[y - R:y + R + 1, x - R:x + R + 1].astype(np.float64) * inv
        off = 0.0
    sat = bool((src[y - 2:y + 3, x - 2:x + 3] >= satthr).any())
    f = p - np.median(p[_RING])
    cx = cy = 0.0
    for _ in range(3):
        w = np.clip(f, 0, None) * (np.hypot(_XXf - cx, _YYf - cy) <= 6)
        s = w.sum()
        if s <= 0:
            return None
        cx, cy = (w * _XXf).sum() / s, (w * _YYf).sum() / s
    r = np.hypot(_XXf - cx, _YYf - cy)
    fm = np.where((r <= R) & (f > noise), f, 0.0)
    flux = fm.sum()
    if flux <= 0:
        return None
    return dict(x=x + cx + off, y=y + cy + off, hfr=float((fm * r).sum() / flux), flux=float(flux),
                peak=float(f.max()), sat=sat, patch=f, cx=cx, cy=cy)


def _gauss(coords, A, x0, y0, sx, sy, th, off):
    x, y = coords
    c, s = np.cos(th), np.sin(th)
    xr = (x - x0) * c + (y - y0) * s
    yr = -(x - x0) * s + (y - y0) * c
    return (off + A * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))).ravel()


def fit_star(st):
    """Elliptical Gaussian fit -> (fwhm_px, eccentricity) or None."""
    h, R = 8, R_AP
    cx, cy = int(round(st["cx"])), int(round(st["cy"]))
    p = st["patch"][R + cy - h:R + cy + h + 1, R + cx - h:R + cx + h + 1]
    if p.shape != (2 * h + 1, 2 * h + 1):
        return None
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    p0 = [p.max(), st["cx"] - cx, st["cy"] - cy, 2.0, 2.0, 0.0, 0.0]
    try:
        popt, _ = curve_fit(_gauss, (xx, yy), p.ravel(), p0=p0, maxfev=200)   # fast LM, no bounds
    except Exception:
        return None
    sx, sy = abs(popt[3]), abs(popt[4])
    if not (0.4 <= sx <= 8 and 0.4 <= sy <= 8) or abs(popt[1]) > 3 or abs(popt[2]) > 3:
        return None
    return float(2.3548 * np.sqrt(sx * sy)), float(np.sqrt(max(0.0, 1 - (min(sx, sy) / max(sx, sy)) ** 2)))


def analyze(data, header, fs, cfa):
    """data: raw array as stored (mono / Bayer mosaic / RGB cube). cfa: True if data is a Bayer mosaic.
    Detection runs on a 2x2-binned image; measurements use full-resolution patches only."""
    inv = 1.0 / fs
    adu = 65535.0 if fs == 1.0 else fs
    src = data.mean(axis=2, dtype=np.float32) if data.ndim == 3 else data
    H, W = src.shape
    H2, W2 = H // 2, W // 2
    satthr = 0.98 * fs

    b = (src[0:2 * H2:2, 0:2 * W2:2] + src[1:2 * H2:2, 0:2 * W2:2] +
         src[0:2 * H2:2, 1:2 * W2:2] + src[1:2 * H2:2, 1:2 * W2:2]) * np.float32(0.25 * inv)
    grid = block_grid(b)
    if grid is None:
        bk = float(np.median(b[::4, ::4]))
        res, gradient = b - bk, 0.0
    else:
        res = b - upsample(grid, b.shape)
        gradient = float((np.percentile(grid, 95) - np.percentile(grid, 5)) / max(np.median(grid), 1e-7))

    # noise at full resolution, from a central crop (high-pass MAD)
    ch, cw = min(H, 1200), min(W, 1200)
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    crop = src[y0:y0 + ch, x0:x0 + cw].astype(np.float32) * np.float32(inv)
    lumc = box2(crop) if cfa else crop
    hp = lumc - ndi.uniform_filter(lumc, 15)
    noise = max(float(1.4826 * np.median(np.abs(hp - np.median(hp)))), 1e-7)

    sm = ndi.gaussian_filter(res, 0.75)
    ds = sm[::2, ::2]
    nsm = max(float(1.4826 * np.median(np.abs(ds - np.median(ds)))), 1e-7)
    peaks = (sm == ndi.maximum_filter(sm, size=5)) & (sm > 5 * nsm)
    ys, xs = np.nonzero(peaks)
    xs, ys = 2 * xs + 1, 2 * ys + 1                      # back to full-res coordinates
    m = R_AP + 3
    ok = (xs >= m) & (xs < W - m) & (ys >= m) & (ys < H - m)
    xs, ys = xs[ok], ys[ok]
    order = np.argsort(-sm[(ys - 1) // 2, (xs - 1) // 2])[:MAX_CAND]

    stars, n_sat, seen = [], 0, set()
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        cx_, cy_ = x // 6, y // 6
        if any((cx_ + dx, cy_ + dy) in seen for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            continue
        st = measure_star(src, x, y, noise, inv, cfa, satthr)
        if st is None:
            continue
        seen.add((cx_, cy_))
        if st["sat"]:
            n_sat += 1
        elif 0.6 <= st["hfr"] <= R_AP * 0.8:             # reject hot pixels / junk
            stars.append(st)

    stars.sort(key=lambda s: -s["peak"])
    for i, st in enumerate(stars):
        r = fit_star(st) if i < FIT_STARS else None
        st["fwhm"], st["ecc"] = r if r else (np.nan, np.nan)

    sub = src[::2, ::2]
    M = dict(n_stars=len(stars), n_sat=n_sat, noise=noise * adu,
             bg=float(np.median(b[::2, ::2])) * adu,
             sat_frac=float(np.count_nonzero(sub >= satthr)) / sub.size,
             gradient=gradient, hfr=np.nan, hfr_std=np.nan, fwhm_px=np.nan, fwhm_as=None, ecc=np.nan,
             snr=np.nan, tilt=np.nan, regions=None)

    if stars:
        hfrs = np.array([s["hfr"] for s in stars])
        M["hfr"] = float(np.median(hfrs))
        M["hfr_std"] = float(1.4826 * np.median(np.abs(hfrs - np.median(hfrs))))
        fw = np.array([s["fwhm"] for s in stars if np.isfinite(s["fwhm"])])
        ec = np.array([s["ecc"] for s in stars if np.isfinite(s["ecc"])])
        if len(fw):
            M["fwhm_px"] = float(np.median(fw))
        if len(ec):
            M["ecc"] = float(np.median(ec))
        M["snr"] = float(np.median([s["peak"] for s in stars]) / noise)

        reg = np.full((3, 3), np.nan)
        for gy in range(3):
            for gx in range(3):
                v = [s["hfr"] for s in stars if int(3 * s["y"] / H) == gy and int(3 * s["x"] / W) == gx]
                if len(v) >= 3:
                    reg[gy, gx] = np.median(v)
        M["regions"] = reg
        if np.isfinite(reg).sum() >= 4:
            M["tilt"] = float((np.nanmax(reg) - np.nanmin(reg)) / M["hfr"])

    # plate scale from header (assumes XPIXSZ already includes binning, as NINA/SBIG do)
    try:
        scale = 206.265 * float(header["XPIXSZ"]) / float(header["FOCALLEN"])
        M["scale"] = scale
        if np.isfinite(M["fwhm_px"]):
            M["fwhm_as"] = M["fwhm_px"] * scale
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        M["scale"] = None

    L, flags = LIMITS, []
    if M["n_stars"] < L["stars_poor"]:
        flags.append(("bad", f"Very few stars ({M['n_stars']}) - clouds, fog, dew or bad focus?"))
    elif M["n_stars"] < L["stars_warn"]:
        flags.append(("warn", f"Low star count ({M['n_stars']})"))
    if np.isfinite(M["ecc"]):
        if M["ecc"] > L["ecc_poor"]:
            flags.append(("bad", f"Stars strongly elongated (ecc {M['ecc']:.2f}) - guiding/wind/tilt"))
        elif M["ecc"] > L["ecc_warn"]:
            flags.append(("warn", f"Stars somewhat elongated (ecc {M['ecc']:.2f})"))
    if M["fwhm_as"] is not None and M["fwhm_as"] > L["fwhm_warn_arcsec"]:
        flags.append(("warn", f"Large FWHM ({M['fwhm_as']:.1f}\") - seeing or focus"))
    elif M["fwhm_as"] is None and np.isfinite(M["hfr"]) and M["hfr"] > L["hfr_warn_px"]:
        flags.append(("warn", f"High HFR ({M['hfr']:.2f} px)"))
    if M["gradient"] > L["grad_warn"]:
        flags.append(("warn", f"Strong background gradient ({M['gradient']*100:.0f} % of sky level)"))
    if M["sat_frac"] > L["sat_warn"]:
        flags.append(("warn", f"{M['sat_frac']*100:.1f} % of pixels saturated"))
    if np.isfinite(M["tilt"]) and M["tilt"] > L["tilt_warn"]:
        flags.append(("warn", f"HFR varies {M['tilt']*100:.0f} % across the field - tilt/curvature"))
    M["flags"] = flags
    M["verdict"] = "POOR" if any(f[0] == "bad" for f in flags) else "FAIR" if flags else "GOOD"
    M["stars"] = [(s["x"], s["y"], s["hfr"]) for s in stars]
    return M


def format_report(M, header, pattern, shape):
    def hv(k):
        v = header.get(k)
        return None if v is None else str(v)

    lines = []
    tag = {"GOOD": "good", "FAIR": "warn", "POOR": "bad"}[M["verdict"]]
    lines.append((f"Verdict: {M['verdict']}", tag))
    for lvl, txt in M["flags"]:
        lines.append((f"  \u2022 {txt}", lvl))
    lines.append(("", None))
    h, w = shape[:2]
    lines.append((f"Size           {w} x {h}  ({'colour' if len(shape) == 3 else 'mono'})", None))
    lines.append((f"Debayered      {pattern if pattern else 'no'}", None))
    for k in ("OBJECT", "FILTER", "EXPTIME", "GAIN", "CCD-TEMP", "DATE-OBS"):
        if hv(k):
            lines.append((f"{k:<14} {hv(k)}", None))
    lines.append(("", None))
    lines.append((f"Stars measured {M['n_stars']}   (saturated skipped: {M['n_sat']})", None))
    lines.append((f"HFR            {M['hfr']:.2f} px  (\u00b1{M['hfr_std']:.2f})", None))
    fw = f"{M['fwhm_px']:.2f} px" if np.isfinite(M["fwhm_px"]) else "n/a"
    if M["fwhm_as"] is not None:
        fw += f"  =  {M['fwhm_as']:.2f}\""
    lines.append((f"FWHM           {fw}", None))
    lines.append((f"Eccentricity   {M['ecc']:.2f}   (0 = round)", None))
    if M["scale"]:
        lines.append((f"Plate scale    {M['scale']:.2f}\"/px", None))
    lines.append(("", None))
    lines.append((f"Background     {M['bg']:.0f} ADU (16-bit equiv.)", None))
    lines.append((f"Noise (MAD)    {M['noise']:.1f} ADU", None))
    lines.append((f"Star SNR       {M['snr']:.1f}   (median peak / noise)", None))
    lines.append((f"Gradient       {M['gradient']*100:.1f} % of sky level", None))
    lines.append((f"Saturated px   {M['sat_frac']*100:.3f} %", None))
    if M["regions"] is not None:
        lines.append(("", None))
        lines.append(("Median HFR by region (top row = image top):", None))
        for row in M["regions"]:
            lines.append(("   " + "   ".join("  -  " if not np.isfinite(v) else f"{v:5.2f}" for v in row), None))
        if np.isfinite(M["tilt"]):
            lines.append((f"   spread: {M['tilt']*100:.0f} % of median", None))
    return lines


# --------------------------------------------------------------------------- #
#  Compute stages (shared by the GUI, the prefetcher and the CSV exporter)
# --------------------------------------------------------------------------- #
def compute_stage1(path, opts, loaded=None):
    """Load + render. Quick (the image can be shown as soon as this returns)."""
    bayer, stretch, linked, target = opts
    data, header = loaded if loaded else load_fits(path)
    fs = full_scale(data)
    pat = effective_pattern(header, bayer) if data.ndim == 2 else None
    return dict(raw=data, header=header, fs=fs, pat=pat, arr=render(data, fs, pat, stretch, linked, target))


def compute_stage2(s1):
    """Quality analysis."""
    M = analyze(s1["raw"], s1["header"], s1["fs"], s1["pat"] is not None)
    return dict(metrics=M, report=format_report(M, s1["header"], s1["pat"], s1["raw"].shape))


def _prefetch_job(path, opts):
    s1 = compute_stage1(path, opts)
    return opts, s1, compute_stage2(s1)


CSV_COLS = ["file", "marked_bad", "verdict", "stars", "saturated_stars", "hfr_px", "hfr_mad", "fwhm_px",
            "fwhm_arcsec", "eccentricity", "bg_adu", "noise_adu", "star_snr", "gradient_pct", "sat_pct",
            "hfr_spread_pct", "OBJECT", "FILTER", "EXPTIME", "DATE-OBS"]


def _csv_worker(args):
    f, bayer = args
    name = os.path.basename(f)
    try:
        data, hd = load_fits(f)
        fs = full_scale(data)
        pat = effective_pattern(hd, bayer) if data.ndim == 2 else None
        M = analyze(data, hd, fs, pat is not None)
        return [name, name.startswith(BAD_PREFIX), M["verdict"], M["n_stars"], M["n_sat"], f"{M['hfr']:.3f}",
                f"{M['hfr_std']:.3f}", f"{M['fwhm_px']:.3f}",
                "" if M["fwhm_as"] is None else f"{M['fwhm_as']:.3f}", f"{M['ecc']:.3f}",
                f"{M['bg']:.1f}", f"{M['noise']:.2f}", f"{M['snr']:.1f}", f"{M['gradient']*100:.1f}",
                f"{M['sat_frac']*100:.3f}", f"{M['tilt']*100:.0f}" if np.isfinite(M["tilt"]) else "",
                *[hd.get(k, "") for k in ("OBJECT", "FILTER", "EXPTIME", "DATE-OBS")]]
    except Exception as e:  # noqa
        return [name, "", f"ERROR {e!r}"]


# --------------------------------------------------------------------------- #
#  GUI
# --------------------------------------------------------------------------- #
class Viewer(tk.Tk):
    def __init__(self, start=None):
        super().__init__()
        self.title("FITS Quick Viewer")
        self.geometry("1500x900")
        self.q = queue.Queue()
        self.busy = False
        self.gen = 0
        self.path, self.files, self.idx, self.last_dir = None, [], -1, 1
        self.header = self.pil = self.result = self._raw = self.pat = None
        self.fs, self.flip = 1.0, True
        self.zoom, self.ox, self.oy = 1.0, 0.0, 0.0
        self._photo = None
        self._pool = ThreadPoolExecutor(max_workers=1)   # background prefetch of the next file
        self._pre = {}

        self.v_stretch = tk.BooleanVar(value=True)
        self.v_linked = tk.BooleanVar(value=True)
        self.v_stars = tk.BooleanVar(value=True)
        self.v_advance = tk.BooleanVar(value=True)
        self.v_bayer = tk.StringVar(value="Auto")
        self.v_target = tk.DoubleVar(value=0.25)
        self.v_filter = tk.StringVar()
        self.v_status = tk.StringVar(value="Open a FITS file (Ctrl+O)")
        self.v_badbtn = tk.StringVar(value="Mark BAD (B)")

        self._build()
        self.after(50, self._poll)
        if start:
            self.after(200, lambda: self.open_path(start))

    # ---- layout -----------------------------------------------------------
    def _build(self):
        menu = tk.Menu(self)
        fm = tk.Menu(menu, tearoff=0)
        fm.add_command(label="Open…", accelerator="Ctrl+O", command=self.ask_open)
        fm.add_command(label="Analyse folder → CSV…", command=self.export_csv)
        fm.add_separator()
        fm.add_command(label="Quit", command=self.destroy)
        menu.add_cascade(label="File", menu=fm)
        self.config(menu=menu)

        bar = ttk.Frame(self, padding=4)
        bar.pack(fill="x")
        ttk.Button(bar, text="Open", command=self.ask_open).pack(side="left")
        ttk.Button(bar, text="◀", width=3, command=lambda: self.step(-1)).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="▶", width=3, command=lambda: self.step(1)).pack(side="left")
        ttk.Button(bar, textvariable=self.v_badbtn, width=15, command=self.toggle_bad).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(bar, text="then next", variable=self.v_advance).pack(side="left", padx=(2, 8))
        ttk.Checkbutton(bar, text="Auto-stretch", variable=self.v_stretch, command=self.restretch).pack(side="left")
        ttk.Checkbutton(bar, text="Linked", variable=self.v_linked, command=self.restretch).pack(side="left")
        ttk.Label(bar, text="Target bg").pack(side="left", padx=(8, 2))
        sc = ttk.Scale(bar, from_=0.10, to=0.50, variable=self.v_target, length=100)
        sc.pack(side="left")
        sc.bind("<ButtonRelease-1>", lambda e: self.restretch())
        ttk.Label(bar, text="Bayer").pack(side="left", padx=(12, 2))
        cb = ttk.Combobox(bar, textvariable=self.v_bayer, width=7, state="readonly",
                          values=["Auto", "Mono", "RGGB", "BGGR", "GRBG", "GBRG"])
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._start("rebayer"))
        ttk.Checkbutton(bar, text="Show stars", variable=self.v_stars, command=self.redraw).pack(side="left", padx=12)
        ttk.Button(bar, text="Fit", command=self.fit_redraw).pack(side="left")

        ttk.Label(self, textvariable=self.v_status, anchor="w", padding=(6, 2)).pack(side="bottom", fill="x")

        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(pw, bg="#101010", highlightthickness=0)
        pw.add(self.canvas, weight=4)
        nb = ttk.Notebook(pw)
        pw.add(nb, weight=1)

        self.txt = tk.Text(nb, width=58, wrap="none", font=("Courier New", 10), padx=8, pady=6)
        self.txt.tag_config("good", foreground="#1a9850", font=("Courier New", 12, "bold"))
        self.txt.tag_config("warn", foreground="#d98200")
        self.txt.tag_config("bad", foreground="#d73027")
        self.txt.configure(state="disabled")
        nb.add(self.txt, text="Quality")

        hf = ttk.Frame(nb)
        ttk.Entry(hf, textvariable=self.v_filter).pack(fill="x", padx=4, pady=4)
        self.v_filter.trace_add("write", lambda *a: self.fill_header())
        self.tree = ttk.Treeview(hf, columns=("key", "val", "com"), show="headings")
        for c, t, w in (("key", "Keyword", 100), ("val", "Value", 150), ("com", "Comment", 200)):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="w")
        sb = ttk.Scrollbar(hf, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        nb.add(hf, text="FITS Header")

        c = self.canvas
        c.bind("<Configure>", lambda e: self.redraw())
        c.bind("<MouseWheel>", self.on_wheel)
        c.bind("<Button-4>", self.on_wheel)
        c.bind("<Button-5>", self.on_wheel)
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<Double-Button-1>", lambda e: self.fit_redraw())
        c.bind("<Motion>", self.on_motion)
        self.bind("<Control-o>", lambda e: self.ask_open())
        for seq, fn in (("<Left>", lambda: self.step(-1)), ("<Right>", lambda: self.step(1)),
                        ("f", self.fit_redraw), ("1", lambda: self.set_zoom(1.0)),
                        ("b", self.toggle_bad), ("<Delete>", self.toggle_bad)):
            self._bindkey(seq, fn)

    def _bindkey(self, seq, fn):
        def h(e):
            if isinstance(self.focus_get(), tk.Entry):      # don't hijack typing in the header filter
                return
            fn()
        self.bind(seq, h)

    # ---- file handling ----------------------------------------------------
    def ask_open(self):
        p = filedialog.askopenfilename(filetypes=[("FITS", "*.fit *.fits *.fts *.fz"), ("All", "*.*")])
        if p:
            self.open_path(p)

    def open_path(self, path):
        d = os.path.dirname(os.path.abspath(path))
        self.files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(EXTS))
        self.path = os.path.abspath(path)
        self.idx = self.files.index(self.path) if self.path in self.files else -1
        self.last_dir = 1
        self._start("file")

    def step(self, delta):
        if self.files and not self.busy:
            self.last_dir = 1 if delta > 0 else -1
            self.idx = (self.idx + delta) % len(self.files)
            self.path = self.files[self.idx]
            self._start("file")

    def is_bad(self):
        return bool(self.path) and os.path.basename(self.path).startswith(BAD_PREFIX)

    def toggle_bad(self):
        """Prepend BAD_ to the file name (or remove it again)."""
        if not self.path or self.busy:
            return
        d, n = os.path.split(self.path)
        marking = not n.startswith(BAD_PREFIX)
        new = os.path.join(d, BAD_PREFIX + n if marking else n[len(BAD_PREFIX):])
        if os.path.exists(new):
            messagebox.showerror("Rename failed", f"{os.path.basename(new)} already exists.")
            return
        try:
            os.rename(self.path, new)
        except OSError as e:
            messagebox.showerror("Rename failed", str(e))
            return
        if 0 <= self.idx < len(self.files):
            self.files[self.idx] = new
        self.path = new
        self._refresh_title()
        self.redraw()
        if marking and self.v_advance.get() and len(self.files) > 1:
            self.step(1)
        else:
            self.v_status.set(("Marked: " if marking else "Unmarked: ") + os.path.basename(new))

    def _refresh_title(self):
        n = os.path.basename(self.path) if self.path else ""
        self.title(f"FITS Quick Viewer — {n}" + ("   [BAD]" if self.is_bad() else ""))
        self.v_badbtn.set("Unmark BAD (B)" if self.is_bad() else "Mark BAD (B)")

    # ---- background jobs --------------------------------------------------
    def _opts(self):
        return (self.v_bayer.get(), self.v_stretch.get(), self.v_linked.get(), round(self.v_target.get(), 3))

    def _start(self, kind):
        if self.busy or not self.path:
            return
        if kind != "file" and self._raw is None:
            return
        if kind != "stretch":
            self.gen += 1
        self.busy = True
        self.v_status.set(f"Loading {os.path.basename(self.path)} …")
        self._refresh_title()
        threading.Thread(target=self._job, args=(self.gen, kind, self.path, self._opts()), daemon=True).start()

    def restretch(self):
        self._start("stretch")

    def _job(self, gen, kind, path, opts):
        try:
            bayer, stretch, linked, target = opts
            if kind == "stretch":
                arr = render(self._raw, self.fs, self.pat, stretch, linked, target)
                self.q.put(("s1", gen, dict(kind=kind, arr=arr)))
                return
            fut = self._pre.pop(path, None) if kind == "file" else None
            if fut is not None:
                try:
                    copts, s1, s2 = fut.result()              # waits if the prefetch is still running
                    if copts[0] == bayer:
                        if copts[1:] != opts[1:]:
                            s1 = dict(s1, arr=render(s1["raw"], s1["fs"], s1["pat"], stretch, linked, target))
                        self.q.put(("s1", gen, dict(s1, kind=kind)))
                        self.q.put(("s2", gen, s2))
                        return
                except Exception:
                    pass
            loaded = (self._raw, self.header) if kind == "rebayer" else None
            s1 = compute_stage1(path, opts, loaded)
            self.q.put(("s1", gen, dict(s1, kind=kind)))      # image appears now ...
            self.q.put(("s2", gen, compute_stage2(s1)))       # ... metrics follow
        except Exception as e:  # noqa
            self.q.put(("err", gen, repr(e)))

    def _poll(self):
        try:
            while True:
                kind, gen, payload = self.q.get_nowait()
                if kind == "progress":
                    self.v_status.set(payload)
                elif kind == "done_csv":
                    self.v_status.set(payload)
                elif gen != self.gen:
                    continue                                   # stale result from a file we left
                elif kind == "s1":
                    self._apply_s1(payload)
                elif kind == "s2":
                    self._apply_s2(payload)
                elif kind == "err":
                    self.busy = False
                    self.v_status.set(f"Error: {payload}")
                    messagebox.showerror("Error", payload)
        except queue.Empty:
            pass
        self.after(40, self._poll)

    def _apply_s1(self, out):
        self.busy = False
        kind = out["kind"]
        if kind != "stretch":
            self._raw, self.header, self.fs, self.pat = out["raw"], out["header"], out["fs"], out["pat"]
            self.result = None
            self.flip = str(self.header.get("ROWORDER", "")).upper() != "TOP-DOWN"
            self.show_report([("Analysing …", None)])
            if kind == "file":
                self.fill_header()
        img = Image.fromarray(out["arr"])
        if self.flip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        size_changed = self.pil is None or self.pil.size != img.size
        self.pil = img
        if kind == "file" or size_changed:
            self.fit()
        self.redraw()
        if kind != "stretch":
            self.v_status.set(f"{os.path.basename(self.path)}   —   analysing …")

    def _apply_s2(self, out):
        self.result = out["metrics"]
        self.show_report(out["report"])
        self.redraw()
        r = self.result
        n = f"[{self.idx + 1}/{len(self.files)}] " if self.files else ""
        self.v_status.set(f"{n}{os.path.basename(self.path)}   —   {r['verdict']}   "
                          f"HFR {r['hfr']:.2f}px   stars {r['n_stars']}")
        self._start_prefetch()

    def _start_prefetch(self):
        if len(self.files) < 2 or self.idx < 0:
            return
        nxt = self.files[(self.idx + self.last_dir) % len(self.files)]
        if nxt == self.path or nxt in self._pre:
            return
        self._pre = {nxt: self._pool.submit(_prefetch_job, nxt, self._opts())}

    # ---- panels -----------------------------------------------------------
    def show_report(self, lines):
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        for text, tag in lines:
            self.txt.insert("end", text + "\n", tag or ())
        self.txt.configure(state="disabled")

    def fill_header(self):
        self.tree.delete(*self.tree.get_children())
        if self.header is None:
            return
        f = self.v_filter.get().strip().lower()
        for card in self.header.cards:
            row = (card.keyword, str(card.value), card.comment)
            if not f or any(f in str(x).lower() for x in row):
                self.tree.insert("", "end", values=row)

    # ---- canvas -----------------------------------------------------------
    def fit(self):
        if self.pil is None:
            return
        cw, ch = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
        W, H = self.pil.size
        self.zoom = min(cw / W, ch / H)
        self.ox = -(cw / self.zoom - W) / 2
        self.oy = -(ch / self.zoom - H) / 2

    def fit_redraw(self):
        self.fit()
        self.redraw()

    def set_zoom(self, z):
        if self.pil is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        cx, cy = self.ox + cw / 2 / self.zoom, self.oy + ch / 2 / self.zoom
        self.zoom = z
        self.ox, self.oy = cx - cw / 2 / z, cy - ch / 2 / z
        self.redraw()

    def on_wheel(self, e):
        if self.pil is None:
            return
        up = e.num == 4 or getattr(e, "delta", 0) > 0
        z = min(max(self.zoom * (1.25 if up else 0.8), 0.02), 32)
        ix, iy = self.ox + e.x / self.zoom, self.oy + e.y / self.zoom
        self.zoom = z
        self.ox, self.oy = ix - e.x / z, iy - e.y / z
        self.redraw()

    def on_press(self, e):
        self.canvas.focus_set()
        self._drag = (e.x, e.y, self.ox, self.oy)

    def on_drag(self, e):
        if self.pil is None or not hasattr(self, "_drag"):
            return
        x0, y0, ox, oy = self._drag
        self.ox = ox - (e.x - x0) / self.zoom
        self.oy = oy - (e.y - y0) / self.zoom
        self.redraw()

    def on_motion(self, e):
        if self.pil is None or self._raw is None or self.busy:
            return
        W, H = self.pil.size
        ix, iy = int(self.ox + e.x / self.zoom), int(self.oy + e.y / self.zoom)
        if 0 <= ix < W and 0 <= iy < H:
            ya = H - 1 - iy if self.flip else iy
            v = np.atleast_1d(self._raw[ya, ix])
            tag = f" ({self.pat[(ya % 2) * 2 + ix % 2]})" if self.pat else ""
            self.v_status.set(f"x={ix + 1}  y={ya + 1}   raw: " + ", ".join(f"{a:.6g}" for a in v) + tag
                              + f"    zoom {self.zoom * 100:.0f}%")

    def redraw(self):
        c = self.canvas
        c.delete("all")
        if self.pil is None:
            return
        cw, ch = c.winfo_width(), c.winfo_height()
        W, H = self.pil.size
        z = self.zoom
        x0, y0 = max(self.ox, 0), max(self.oy, 0)
        x1, y1 = min(self.ox + cw / z, W), min(self.oy + ch / z, H)
        if x1 > x0 and y1 > y0:
            box = (int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1)))
            crop = self.pil.crop(box)
            size = (max(1, round(crop.width * z)), max(1, round(crop.height * z)))
            img = crop.resize(size, Image.NEAREST) if z >= 1 else crop.resize(size, Image.BILINEAR, reducing_gap=2.0)
            self._photo = ImageTk.PhotoImage(img)
            c.create_image((box[0] - self.ox) * z, (box[1] - self.oy) * z, image=self._photo, anchor="nw")

        if self.v_stars.get() and self.result:
            n = 0
            for x, y, hfr in self.result["stars"]:
                yd = H - 1 - y if self.flip else y
                px, py = (x + 0.5 - self.ox) * z, (yd + 0.5 - self.oy) * z
                if 0 <= px <= cw and 0 <= py <= ch:
                    r = max(5, 2.0 * hfr * z)
                    c.create_oval(px - r, py - r, px + r, py + r, outline="#33ff66")
                    n += 1
                    if n > 500:
                        break
        if self.is_bad():
            c.create_text(14, 10, text="BAD", fill="#ff4040", anchor="nw", font=("Arial", 26, "bold"))

    # ---- batch export (parallel) -----------------------------------------
    def export_csv(self):
        folder = filedialog.askdirectory(initialdir=os.path.dirname(self.path) if self.path else None,
                                         title="Folder with FITS files")
        if not folder:
            return
        out = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="fits_quality.csv")
        if not out:
            return
        files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(EXTS))
        jobs = [(f, self.v_bayer.get()) for f in files]
        workers = max(1, min((os.cpu_count() or 2) - 1, 6))

        def run():
            try:
                with open(out, "w", newline="") as fh, ProcessPoolExecutor(workers) as ex:
                    w = csv.writer(fh)
                    w.writerow(CSV_COLS)
                    for i, row in enumerate(ex.map(_csv_worker, jobs), 1):
                        w.writerow(row)
                        self.q.put(("progress", 0, f"Analysed {i}/{len(jobs)} ({workers} workers)"))
                self.q.put(("done_csv", 0, f"Saved {len(jobs)} rows to {out}"))
            except Exception as e:  # noqa
                self.q.put(("done_csv", 0, f"CSV export failed: {e!r}"))

        threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    Viewer(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()