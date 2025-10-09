import csv
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional
import pandas as pd # NEW: Import pandas for CSV output

# Define the speed of light in Angstroms per second (299,792,458 m/s)
# c ≈ 2.99792458 x 10^18 A/s
SPEED_OF_LIGHT_AA_PER_S = 2.99792458e18
# Conversion factor from µJy to erg/s/cm^2/Hz is 10^-29
MUJY_TO_CGS_NU = 1e-29

# --- CONFIGURATION ---
SPECTRA_BASE_DIR = "/raid/scratch/work/Griley/GALFIND_WORK/Spectra/2D"
CSV_PATH_GLOBAL = Path("./mphys_GOODS_S_exposures.csv")
OUTPUT_DIR = Path.cwd() / "UV_SNR_plots" 
# ---------------------

# ---------- FILE DISCOVERY FUNCTION (No change) ----------

def find_prism_fits(base_dir: str):
    """Recursively find all FITS files with 'prism' in the filename."""
    fits_files = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".fits") and "prism" in f:
                fits_files.append(Path(os.path.join(root, f)))
    return fits_files

# ---------- I/O HELPERS (No change) ----------

def get_redshift_from_csv(csv_file: Path, fits_file: Path):
    """
    Look up the redshift 'z' for a given FITS filename (matches the 'file' column).
    Returns float or None.
    """
    target = Path(fits_file).name.strip()
    with open(csv_file, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("file", "").strip() == target:
                try:
                    return float(str(row.get("z", "")).strip())
                except (TypeError, ValueError):
                    return None
    return None


def read_observed_spectrum(fits_path: Path):
    """
    Read observed-frame spectrum columns from a 1D FITS table.
    Expects 'wave', 'flux', and error 'err' (or 'full_err') columns.
    Returns wave_obs (µm), flux_obs (µJy), err_obs (µJy).
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        wave_obs = np.array(data["wave"])
        flux_obs = np.array(data["flux"])
        if "err" in data.columns.names:
            err_obs = np.array(data["err"])
        elif "full_err" in data.columns.names:
            err_obs = np.array(data["full_err"])
        else:
            raise KeyError("No flux error column found (expected 'err' or 'full_err').")
    return wave_obs, flux_obs, err_obs


# ---------- TRANSFORMS (No change) ----------

def to_rest_frame(wave_obs_um, flux_obs_uJy, err_obs_uJy, z, out_units="AA", out_flux_units="cgs"):
    
    C_LIGHT_AA_PER_S = 2.99792458e18 
    lam_obs_AA  = wave_obs_um * 1e4
    lam_rest_AA = lam_obs_AA / (1.0 + z)

    if out_flux_units.lower() in ["cgs", "erg/s/cm2/aa"]:
        factor = 1e-29 * C_LIGHT_AA_PER_S
        Flam_obs = flux_obs_uJy * factor / (lam_obs_AA**2)
        Elam_obs = err_obs_uJy   * factor / (lam_obs_AA**2)
        w = lam_rest_AA
        f = (1.0 + z)**2 * Flam_obs
        e = (1.0 + z)**2 * Elam_obs

    elif out_flux_units.lower() in ["ujy", "uJy"]:
        w = lam_rest_AA if out_units.lower() in ["aa","angstrom","ang","a"] else wave_obs_um/(1+z)
        f = flux_obs_uJy
        e = err_obs_uJy
    else:
        raise ValueError("out_flux_units must be 'cgs' or 'uJy'")

    idx = np.argsort(w)
    w, f, e = w[idx], f[idx], e[idx]
    keep = np.isfinite(w) & np.isfinite(f) & np.isfinite(e) & (e >= 0)

    return w[keep], f[keep], e[keep]


# ---------- SNR CALCULATION (Per-Pixel Average, No change) ----------

def calculate_uv_snr(w_rest: np.ndarray, f_rest: np.ndarray, e_rest: np.ndarray,
                     w_min_AA: float = 1250, w_max_AA: float = 3000) -> Optional[float]:
    
    mask = (w_rest >= w_min_AA) & (w_rest <= w_max_AA)
    
    if np.sum(mask) == 0:
        print(f"Warning: No data points in the selected UV continuum range ({w_min_AA}-{w_max_AA} A).")
        return None

    f_window = f_rest[mask]
    e_window = e_rest[mask]
    snr_per_pixel = f_window / e_window
    valid_snr = snr_per_pixel[np.isfinite(snr_per_pixel)]
    
    if len(valid_snr) > 0:
        snr = np.mean(valid_snr)
        return np.abs(snr)
    else:
        return None

# ---------- NEW: QUALITY CHECK FUNCTION ----------

def passes_quality_checks(flux: np.ndarray, snr_uv: Optional[float], max_consec_nans: int = 10) -> bool:
    """
    Check if the spectrum passes the quality criteria:
    1. Reject if there are max_consec_nans or more consecutive NaNs in the flux (band gap).
    2. Reject if the average SNR in the UV range is not positive.
    """

    # --- 1. Check for max_consec_nans or more consecutive NaNs ---
    nan_mask = ~np.isfinite(flux)
    if np.any(nan_mask):
        # Calculate the difference in indices where the NaN mask changes state
        change_indices = np.where(np.concatenate(([nan_mask[0]], nan_mask[:-1] != nan_mask[1:], [True])))[0]
        
        # Calculate the length of each consecutive run. The length of NaN runs
        # starts at the odd indices of the change_indices array difference (::2).
        consec_counts = np.diff(change_indices)[::2]
        
        if len(consec_counts) > 0 and np.any(consec_counts >= max_consec_nans):
            return False

    # --- 2. Check SNR criterion ---
    # snr_uv is None means calculation failed (e.g., no data in range)
    # np.isnan(snr_uv) means mean failed (e.g., all F/E were non-finite)
    if snr_uv is None or np.isnan(snr_uv) or snr_uv <= 0:
        return False

    return True

# ---------------- PLOTTING FUNCTIONS (No change to logic) ----------------

def plot_snr_per_pixel(
    w_rest, f_rest, e_rest, z, title="S/N per Pixel Spectrum",
    save_path=None, show=True, ax=None, w_min_AA=1250, w_max_AA=3000
):
    import numpy as np
    import matplotlib.pyplot as plt

    snr_per_pixel = f_rest / e_rest
    m = np.isfinite(snr_per_pixel)
    w, snr = w_rest[m], snr_per_pixel[m]
    
    if len(w) == 0:
        return None

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        created = True
    else:
        fig = ax.figure
    
    ax.plot(w, snr, color='darkred', lw=1.0, alpha=0.7, label='S/N per pixel')
    mask_uv = (w >= w_min_AA) & (w <= w_max_AA)
    mean_snr_uv = np.mean(snr[mask_uv]) if np.sum(mask_uv) > 0 else np.nan
    ax.axvspan(w_min_AA, w_max_AA, color='lightgrey', alpha=0.3, label=f'UV Continuum Range ({w_min_AA}-{w_max_AA} Å)')

    if np.isfinite(mean_snr_uv):
        ax.axhline(mean_snr_uv, color='red', linestyle='--', lw=1.5, 
                   label=f'Mean UV SNR = {mean_snr_uv:.2f}')

    ax.set_xlabel("Rest-frame Wavelength (Å)")
    ax.set_ylabel("Signal-to-Noise Ratio (per pixel)")
    
    ax.set_title(f"{title}: {Path(save_path).stem.split('_SNR')[0] if save_path else 'Spectrum'} (z={z:.3f})", pad=15)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', frameon=True)
    
    if len(snr) > 10:
        y_lo, y_hi = np.nanpercentile(snr, [1, 99])
        y_range = y_hi - y_lo
        y_margin = 0.1 * y_range
        ax.set_ylim(y_lo - y_margin, y_hi + y_margin)

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    if save_path:
        save_path = Path(save_path) 
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    if show and created:
        plt.show()
    return ax

def plot_rest_spectrum(
    w_rest, f_rest, e_rest, z, title="Rest-frame Spectrum",
    save_path=None, show=True, ax=None, flux_units="cgs",
    min_wave_A=1200, xlim=(1000, 10000), use_percentile_ylim=True,
    smooth_window_A=30,  # set None to disable
    snr: Optional[float] = None
):
    import numpy as np
    import matplotlib.pyplot as plt

    m = np.isfinite(w_rest) & np.isfinite(f_rest) & np.isfinite(e_rest)
    if min_wave_A is not None:
        m &= (w_rest >= min_wave_A)
    w, f, e = w_rest[m], f_rest[m], e_rest[m]

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        created = True
    else:
        fig = ax.figure

    ax.fill_between(w, f - e, f + e, color="lightblue", alpha=0.4, label="±1σ uncertainty")
    ax.plot(w, f, color="dodgerblue", lw=1.0, label="Rest-frame spectrum")

    if smooth_window_A:
        step = max(1, int(smooth_window_A / np.median(np.diff(w))))
        if step > 1:
            from numpy.lib.stride_tricks import sliding_window_view
            def boxcar(y, n):
                if len(y) < n:
                    return np.full_like(y, np.nan)
                sw = sliding_window_view(y, n)
                smoothed = np.nanmean(sw, axis=1)
                pad_left = n // 2
                pad_right = n - 1 - n // 2
                return np.concatenate([np.full(pad_left, np.nan), smoothed, np.full(pad_right, np.nan)])
            
            f_s = boxcar(f, step)
            m_smooth = np.isfinite(f_s)
            ax.plot(w[m_smooth], f_s[m_smooth], color="black", lw=1.0, alpha=0.8, label=f"Smoothed (~{smooth_window_A} Å)")

    ax.set_xlabel("Rest-frame Wavelength (Å)")
    ax.set_ylabel("Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)" if flux_units.lower() in ["cgs","erg/s/cm2/aa"] else "Flux (µJy)")
    ax.set_title(f"{title} (z={z:.3f})", pad=15)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    
    if snr is not None:
        snr_text = f"UV SNR (1250-3000 Å): {snr:.2f}"
        ax.text(
            0.98, 0.95, 
            snr_text, 
            transform=ax.transAxes, 
            fontsize=10, 
            verticalalignment='top', 
            horizontalalignment='right',
            bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none', 'pad': 3}
        )

    if xlim is not None:
        ax.set_xlim(*xlim)
        
    if use_percentile_ylim and len(f) > 10:
        data_to_check = np.r_[f - e, f + e]
        lo_perc, hi_perc = np.nanpercentile(data_to_check, [1, 99])
        
        if np.isfinite(lo_perc) and np.isfinite(hi_perc) and lo_perc < hi_perc:
            data_range = hi_perc - lo_perc
            margin = 0.1 * data_range 
            y_min = lo_perc - margin
            y_max = hi_perc + margin
            
            if np.all(f >= 0) and y_min < 0:
                 y_min = 0 
            
            ax.set_ylim(y_min, y_max)
        else:
            data_min, data_max = np.nanmin(data_to_check), np.nanmax(data_to_check)
            if data_min < data_max:
                 ax.set_ylim(data_min, data_max)

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    if save_path:
        save_path = Path(save_path) 
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    if show and created:
        plt.show()
    return ax


# ---------- CONVENIENCE WRAPPER (MODIFIED to return flux for quality check) ----------

def load_and_plot_rest(
    fits_path: Path,
    csv_path: Path,
    out_png: Optional[Path] = None,
    out_snr_png: Optional[Path] = None, 
    show=True,
    out_units="AA",
    out_flux_units="uJy"
):
    z = get_redshift_from_csv(csv_path, fits_path)
    if z is None or not np.isfinite(z):
        raise ValueError(f"No redshift found in {csv_path} for file '{Path(fits_path).name}'.")

    wave_obs, flux_obs, err_obs = read_observed_spectrum(fits_path)
    w, f, e = to_rest_frame(wave_obs, flux_obs, err_obs, z, out_units=out_units, out_flux_units=out_flux_units)

    snr = calculate_uv_snr(w, f, e)
    title = f"Rest-frame Spectrum: {Path(fits_path).stem}"

    # Check for quality before plotting
    if not passes_quality_checks(f, snr):
        return w, f, e, z, snr, False # Return False status if check fails

    # 1. Plot the Flux Spectrum
    plot_rest_spectrum(
        w, f, e, z,
        title=title,
        save_path=out_png,
        show=show,
        flux_units=out_flux_units,
        snr=snr
    )
    
    # 2. Plot the SNR per pixel
    if out_snr_png is not None:
        plot_snr_per_pixel(
            w, f, e, z,
            title="S/N per Pixel Spectrum",
            save_path=out_snr_png,
            show=show
        )

    return w, f, e, z, snr, True # Return True status if successful


# -----------------------------------------------------------
## CLI-style Batch Processing (MODIFIED for Quality Check and CSV Output)
# -----------------------------------------------------------

if __name__ == "__main__":
    
    FLUX_UNITS = "cgs" 
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Searching for 'prism' FITS files in: {SPECTRA_BASE_DIR}")
    fits_files = find_prism_fits(SPECTRA_BASE_DIR)
    
    if not fits_files:
        print("No prism FITS files found. Exiting.")
        exit()
        
    print(f"Found {len(fits_files)} files to process. Outputs saved to {OUTPUT_DIR.resolve()}")
    print("-" * 30)

    total_snr = 0
    count = 0
    
    # NEW: List to store valid SNR results for CSV output
    snr_results = [] 
    
    for fits_path in fits_files:
        snr = None
        try:
            stem = fits_path.stem
            out_png = OUTPUT_DIR / f"{stem}_rest_CGS.png" 
            out_snr_png = OUTPUT_DIR / f"{stem}_SNR_per_pixel.png" 

            # Load, check quality, and plot
            w, f, e, z, snr, success = load_and_plot_rest(
                fits_path, 
                CSV_PATH_GLOBAL, 
                out_png=out_png, 
                out_snr_png=out_snr_png, 
                show=False, 
                out_units="AA", 
                out_flux_units=FLUX_UNITS
            )
            
            if success:
                total_snr += snr
                count += 1
                snr_results.append({'file': fits_path.name, 'z': z, 'uv_snr_mean': snr})
                print(f"PASS: {stem} (z={z:.3f}): SNR={snr:.2f}")
            else:
                # Failure could be due to quality check or SNR calculation failure
                reason = "Fails Quality Check (e.g., band gap or SNR<=0)"
                if snr is None or np.isnan(snr):
                    reason += " (SNR could not be calculated)"
                
                # Still record the file and its status/non-passing SNR
                snr_results.append({'file': fits_path.name, 'z': z, 'uv_snr_mean': snr, 'status': 'REJECTED'})
                print(f"FAIL: {stem} (z={z:.3f}): {reason}")


        except Exception as err:
            snr_results.append({'file': fits_path.name, 'z': 'N/A', 'uv_snr_mean': 'N/A', 'status': 'ERROR'})
            print(f"ERROR processing {fits_path.name}: {err}")
            
    print("-" * 30)
    
    # NEW: Save all results to a CSV
    results_df = pd.DataFrame(snr_results)
    results_csv_path = OUTPUT_DIR / "uv_snr_results.csv"
    results_df.to_csv(results_csv_path, index=False)

    if count > 0:
        print(f"Batch processing complete. Total files passing quality checks: {count}")
        print(f"Average UV Continuum SNR across passing batch: {(total_snr / count):.2f}")
    else:
        print("Batch processing complete. No spectra passed the quality checks.")
        
    print(f"Detailed SNR results saved to: {results_csv_path.resolve()}")