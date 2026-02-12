import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
import pandas as pd
from astropy.table import Table
import os


# ---------------------- CONFIG ----------------------
C_LIGHT = 2.99792458e18  # Å/s
UV_RANGE = (1250, 3000)  # Å
MIN_WAVELENGTH = 1200    # Å - cut for the noisy start of the spectra
PRISM_CHECK_RANGE = (1630, 1650)  # Å - range to check for NaNs
MAX_NAN_FRACTION = 0.5   # Maximum fraction of NaNs allowed in the check range
SPECTRA_DIR = "/raid/scratch/work/austind/GALFIND_WORK/Spectra/2D"
# CSV_PATH = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/mphys_GOODS_S_exposures.csv"
CATALOGUE_PATH = "/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/gdsgdn_catalogue.fits"
OUT_DIR = "/nvme/scratch/work/Griley/Masters/UV_SNRs"
LOCAL_DIR = "/nvme/scratch/work/Griley/Masters"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------- CORE FUNCTIONS ----------------------

def find_prism_fits(base_dir):
    """Recursively find all FITS files with 'prism' and one of
    'gds', 'gdn', or 'goodsn' in the filename.
    """
    fits_files = []
    required_substrings = ("gds", "gdn", "goodsn")

    for root, dirs, files in os.walk(base_dir):
        for f in files:
            fname = f.lower()
            if (
                fname.endswith(".fits")
                and "prism" in fname
                and "v4" in fname
                and any(s in fname for s in required_substrings)
            ):
                fits_files.append(os.path.join(root, f))

    return fits_files


# def get_redshift(catalogue_path, fits_name):
#     df = pd.read_csv(catalogue_path)
#     match = df[df['file'] == fits_name]
#     if match.empty:
#         raise ValueError(f"No matching entry in CSV for: {fits_name}")
#     return float(match['z'].values[0])


def get_redshift(catalogue_path, fits_name,
                 file_col="file", z_col="zrf"):
    """
    Read redshift from a FITS catalogue.

    Parameters
    ----------
    catalogue_path : str
        Path to the FITS catalogue
    fits_name : str
        Name of the spectrum FITS file (basename)
    file_col : str
        Column name containing spectrum filenames
    z_col : str
        Column name containing redshifts
    """

    cat = Table.read(catalogue_path)

    if file_col not in cat.colnames:
        raise KeyError(f"Column '{file_col}' not found in catalogue")

    if z_col not in cat.colnames:
        raise KeyError(f"Column '{z_col}' not found in catalogue")
    
    # Match by basename (important if paths differ)
    cat_basenames = [os.path.basename(str(f)) for f in cat[file_col]]

    matches = [i for i, f in enumerate(cat_basenames) if f == fits_name]

    if len(matches) == 0:
        print(f"[MISSING IN CATALOGUE] Spectrum file has no catalogue entry: {fits_name}")
        raise ValueError(f"No matching entry in catalogue for: {fits_name}\n")

    # # Match by basename (important if paths differ)
    # matches = [i for i, f in enumerate(cat[file_col])
    #            if os.path.basename(str(f)) == fits_name]

    # if len(matches) == 0:
    #     raise ValueError(f"No matching entry in catalogue for: {fits_name}")

    if len(matches) > 1:
        print(f"Warning: multiple matches for {fits_name}, using first")

    return float(cat[z_col][matches[0]])



def read_spectrum(fits_path):
    with fits.open(fits_path) as hdul:
        spec = hdul['SPEC1D'].data
    return spec['wave'], spec['flux'], spec['err']


def convert_to_rest_frame(wave_obs, flux_obs, err_obs, z):
    """
    Convert observed flux in μJy and wavelength in μm to rest-frame
    wavelength in Å and flux in μJy (λ * Fλ) units.
    """
    flux_nu = flux_obs * 1e-29         # μJy → erg/s/cm²/Hz
    err_nu = err_obs * 1e-29
    wave_A = wave_obs * 1e4            # μm → Å
    flux_lambda = flux_nu * C_LIGHT / wave_A**2
    err_lambda = err_nu * C_LIGHT / wave_A**2
    wave_rest = wave_A / (1 + z)
    flux_rest = flux_lambda * (1 + z)**2
    err_rest = err_lambda * (1 + z)**2
    return wave_rest, flux_rest, err_rest


def clean_spectrum(wave, flux, err, min_wave=MIN_WAVELENGTH):
    mask = ((wave > min_wave))
    return wave[mask], flux[mask], err[mask]


def compute_snr(flux, err):
    snr = flux / err
    valid = np.isfinite(snr)
    return snr[valid], valid


def average_snr_in_range(wave, snr, wave_range=UV_RANGE):
    mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
    snr_in_range = snr[mask]
    if len(snr_in_range) == 0:
        return np.nan
    return np.mean(snr_in_range)


def check_prism_coverage(wave, flux, check_range=PRISM_CHECK_RANGE, max_nan_frac=MAX_NAN_FRACTION):
    """
    Check if the spectrum has sufficient coverage in the prism check range.
    Returns True if the spectrum passes (has enough valid data).
    Returns False if there are too many NaNs in the check range.
    """
    # Find data points in the check range
    mask = (
    np.isfinite(wave)
    & (wave >= check_range[0])
    & (wave <= check_range[1])
    )
    
    if not np.any(mask):
        # No data in this range at all
        print(f"No data coverage in range {check_range[0]}-{check_range[1]} Å")
        return False
    
    flux_in_range = flux[mask]
    n_total = len(flux_in_range)
    n_nan = np.sum(~np.isfinite(flux_in_range))
    nan_fraction = n_nan / n_total if n_total > 0 else 1.0
    
    print(f"Range {check_range[0]}-{check_range[1]} Å: {n_nan}/{n_total} NaNs ({nan_fraction:.1%})")
    
    if nan_fraction > max_nan_frac:
        print(f"Rejected: NaN fraction {nan_fraction:.1%} exceeds threshold {max_nan_frac:.1%}")
        return False
    
    return True


def plot_spectrum(wave, flux, err, fits_name, z, out_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(wave, flux, color='dodgerblue', lw=1.0, label='Rest-frame spectrum')
    plt.fill_between(wave, flux - err, flux + err, color='lightblue', alpha=0.4, label='1σ uncertainty')
    plt.xlabel(r"Wavelength $\lambda_{\rm rest}$ [Å]", fontsize=13)
    plt.ylabel(r"Flux density $F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=13)
    plt.title(f"Rest-frame Spectrum (λ > {MIN_WAVELENGTH} Å): {fits_name}\n(z = {z:.3f})", fontsize=14)
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.xlim(wave.min(), 2000)
    plt.tight_layout()
    out_plot = os.path.join(out_dir, fits_name.replace('.fits', '_restframe.png'))
    plt.savefig(out_plot, dpi=200)
    plt.close()
    return out_plot


def plot_snr(wave, snr, avg_snr, fits_name, out_dir, wave_range=UV_RANGE):
    plt.figure(figsize=(10, 4))
    plt.plot(wave, snr, color='darkorange', lw=1.0, alpha=0.7)
    plt.axvspan(wave_range[0], wave_range[1], color='lightgrey', alpha=0.3, label='UV continuum range')
    if not np.isnan(avg_snr):
        plt.axhline(avg_snr, color='red', lw=1.0, linestyle='--', label=f'⟨SNR⟩ = {avg_snr:.2f}')
    plt.xlabel(r"Wavelength $\lambda_{\rm rest}$ [Å]", fontsize=13)
    plt.ylabel("S/N per pixel", fontsize=13)
    plt.title(f"Average SNR in UV range ({wave_range[0]}–{wave_range[1]} Å)\n{fits_name}", fontsize=14)
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_plot = os.path.join(out_dir, fits_name.replace('.fits', '_UV_SNR.png'))
    plt.savefig(out_plot, dpi=200)
    plt.close()
    return out_plot


def passes_quality_checks(flux, snr_uv, max_consec_nans=10):
    """
    Check if the spectrum passes the quality criteria:
    1. Reject if there are 10 or more consecutive NaNs in the flux (band gap).
    2. Reject if the average SNR in the UV range is not positive.
    """

    # --- 1. Check for 10+ consecutive NaNs ---
    nan_mask = ~np.isfinite(flux)
    if np.any(nan_mask):
        # Count consecutive NaNs using run-length encoding logic
        consec_counts = np.diff(np.where(np.concatenate(([nan_mask[0]],
                                                         nan_mask[:-1] != nan_mask[1:],
                                                         [True])))[0])[::2]
        if len(consec_counts) > 0 and np.any(consec_counts >= max_consec_nans):
            return False

    # --- 2. Check SNR criterion ---
    if snr_uv is None or np.isnan(snr_uv) or snr_uv <= 0:
        return False

    return True


def process_spectrum(fits_path, catalogue_path, out_dir):
    fits_name = os.path.basename(fits_path)
    z = get_redshift(catalogue_path, fits_name)
    print(f"Processing {fits_name}, z = {z:.4f}")

    wave_obs, flux_obs, err_obs = read_spectrum(fits_path)
    wave_rest, flux_rest, err_rest = convert_to_rest_frame(wave_obs, flux_obs, err_obs, z)
    wave, flux, err = clean_spectrum(wave_rest, flux_rest, err_rest)

    plot_spectrum(wave, flux, err, fits_name, z, out_dir)

    # NEW: Check prism coverage in 1630-1650 Å range FIRST
    if not check_prism_coverage(wave, flux):
        print(f"Discarded {fits_name} due to insufficient prism coverage at ≥1630 Å\n")
        return None

    snr, valid_mask = compute_snr(flux, err)
    wave_valid = wave[valid_mask]
    avg_snr_uv = average_snr_in_range(wave_valid, snr)

    # Apply the quality checks before saving to results
    if passes_quality_checks(flux, avg_snr_uv):
        # plot_snr(wave_valid, snr, avg_snr_uv, fits_name, out_dir)
        print(f"Saved {fits_name} with avg SNR (1250–3000 Å) = {avg_snr_uv:.2f}\n")
        return {"file": fits_name, "avg_snr_uv": avg_snr_uv}
    else:
        print(f"Discarded {fits_name} due to failing quality checks\n")
        return None

# ---------------------- MAIN ----------------------

if __name__ == "__main__":
    fits_files = find_prism_fits(SPECTRA_DIR)
    print(f"Found {len(fits_files)} prism FITS files to process.\n")

    results_csv = os.path.join(LOCAL_DIR, "uv_snr_summary_gdsgdn.csv")

    # Load already-processed filenames (if CSV exists)
    if os.path.exists(results_csv):
        existing_df = pd.read_csv(results_csv, usecols=["file"])
        processed = set(existing_df["file"].astype(str))
        header_written = True
        print(f"Skipping {len(processed)} already-processed spectra.\n")
    else:
        processed = set()
        header_written = False

    n_saved = 0
    n_skipped = 0

    for f in fits_files:
        fits_name = os.path.basename(f)

        if fits_name in processed:
            n_skipped += 1
            continue

        try:
            res = process_spectrum(f, CATALOGUE_PATH, OUT_DIR)
            if res is None:
                continue

            df = pd.DataFrame([res])
            df.to_csv(
                results_csv,
                mode="a",
                header=not header_written,
                index=False,
            )

            header_written = True
            processed.add(fits_name)
            n_saved += 1

        except Exception as e:
            print(f"Failed {f}: {e}")

    print(f"\nSaved filtered SNR summary to: {results_csv}")
    print(f"New spectra saved: {n_saved}")
    print(f"Spectra skipped (already in CSV): {n_skipped}")

