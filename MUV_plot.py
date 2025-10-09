import csv
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional, List, Dict, Any, Set
import pandas as pd
import astropy.units as au
from astropy.cosmology import Planck18
from scipy.integrate import simpson

# --- CONSTANTS ---
C_LIGHT_AA_PER_S = 2.99792458e18 
AB_MAG_ZP_JY = 8.90
W_UV_MIN = 1350.0 
W_UV_MAX = 1800.0
cosmo = Planck18

# --- CONFIGURATION ---
SPECTRA_BASE_DIR = "/raid/scratch/work/Griley/GALFIND_WORK/Spectra/2D"
CSV_PATH_GLOBAL = Path("./mphys_GOODS_S_exposures.csv")
OUTPUT_DIR = Path.cwd() / "MUV_plots"
# Path to the external SNR file for filtering (implicitly SNR >= 5.0)
EXTERNAL_SNR_CSV_PATH = Path("/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/specFitMSA/src/uv_snr_5plus.csv") 

# ----------------------------------------------------------------------
## FILTERING HELPERS (FINAL CORRECTION)
# ----------------------------------------------------------------------

def get_snr_filter_files(snr_csv_path: Path) -> Set[str]:
    """
    Loads the external SNR CSV and returns a set of filenames present in the file.
    This assumes the input CSV (uv_snr_5plus.csv) already contains the desired 
    SNR >= 5.0 sample and no further numerical filtering is needed.
    """
    if not snr_csv_path.exists():
        print(f"Warning: SNR filter file not found at {snr_csv_path}. Proceeding without SNR filter.")
        return None 
        
    try:
        # Load the CSV, only need the 'file' column
        df_snr = pd.read_csv(snr_csv_path, usecols=['file'])
        
        # Extract all unique filenames as a set
        valid_files = set(df_snr['file'].astype(str).values)
        
        print(f"Loaded {len(valid_files)} filenames from the external CSV. These will be used as the MUV sample (implicit SNR >= 5.0).")
        return valid_files
        
    except Exception as e:
        print(f"Error reading or filtering SNR file: {e}")
        return None

# ----------------------------------------------------------------------
## I/O & TRANSFORMATION HELPERS (Reused)
# ----------------------------------------------------------------------

def find_prism_fits(base_dir: str) -> List[Path]:
    """Recursively find all FITS files with 'prism' in the filename."""
    fits_files = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".fits") and "prism" in f:
                fits_files.append(Path(os.path.join(root, f)))
    return fits_files

def get_redshift_from_csv(csv_file: Path, fits_file: Path) -> Optional[float]:
    """Look up the redshift 'z' for a given FITS filename."""
    target = fits_file.name.strip()
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("file", "").strip() == target:
                try:
                    return float(row.get("z", "").strip())
                except (TypeError, ValueError):
                    return None
    return None

def read_observed_spectrum(fits_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read observed-frame spectrum columns from a 1D FITS table."""
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        wave_obs = np.array(data["wave"])
        flux_obs = np.array(data["flux"])
        err_obs = np.array(data["err"] if "err" in data.columns.names else data["full_err"])
    return wave_obs, flux_obs, err_obs

def get_rest_frame_spectrum(wave_obs_um: np.ndarray, flux_obs_uJy: np.ndarray, 
                            err_obs_uJy: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts observed spectrum (µm, µJy) to rest-frame (Å, erg/s/cm2/Å)."""
    lam_obs_AA  = wave_obs_um * 1e4
    lam_rest_AA = lam_obs_AA / (1.0 + z)

    factor = 1e-29 * C_LIGHT_AA_PER_S
    Flam_obs = flux_obs_uJy * factor / (lam_obs_AA**2)
    Elam_obs = err_obs_uJy   * factor / (lam_obs_AA**2)

    w = lam_rest_AA
    f = (1.0 + z)**2 * Flam_obs
    e = (1.0 + z)**2 * Elam_obs

    idx = np.argsort(w)
    w, f, e = w[idx], f[idx], e[idx]
    keep = np.isfinite(w) & np.isfinite(f) & np.isfinite(e) & (e >= 0)
    
    return w[keep], f[keep], e[keep]

# ----------------------------------------------------------------------
## MUV CALCULATION (Fixes numpy.log11 typo)
# ----------------------------------------------------------------------

def calculate_muv(w_rest: np.ndarray, f_rest_flambda: np.ndarray, z: float, 
                  w_min: float = W_UV_MIN, w_max: float = W_UV_MAX) -> Optional[float]:
    """
    Calculates M_UV (Absolute AB Magnitude) using the integrated flux method.
    """
    
    is_UV = (w_rest >= w_min) & (w_rest <= w_max)
    
    if not is_UV.any():
        return None
    
    w_window = w_rest[is_UV]
    f_window = f_rest_flambda[is_UV]
    
    valid_mask = np.isfinite(f_window)
    w_window = w_window[valid_mask]
    f_window = f_window[valid_mask]

    if len(w_window) < 2:
        return None

    integral_flam = simpson(f_window, x=w_window)
    bandpass_width = w_max - w_min
    Flam_UV_mean = integral_flam / bandpass_width
    
    if Flam_UV_mean <= 0:
        return None
        
    lambda_eff = (w_min + w_max) / 2.0
    
    F_nu_cgs = Flam_UV_mean * (lambda_eff**2 / C_LIGHT_AA_PER_S)
    F_nu_Jy = F_nu_cgs / 1e-23
    
    m_UV_AB = -2.5 * np.log10(F_nu_Jy) + AB_MAG_ZP_JY 
    
    dL_pc = cosmo.luminosity_distance(z).to(au.pc).value
    # FIX: Corrected typo from np.log11 to np.log10
    DM = 5.0 * (np.log10(dL_pc) - 1.0) 
    K = -2.5 * np.log10(1 + z)
    
    M_UV_AB = m_UV_AB - DM - K
    
    return M_UV_AB

# ----------------------------------------------------------------------
## MAIN EXECUTION AND PLOTTING
# ----------------------------------------------------------------------

def process_and_plot_muv_vs_z(base_dir: str, csv_path: Path, output_dir: Path, snr_filter_path: Path):
    """
    Main function to calculate MUV and plot MUV vs z for all prism spectra, 
    filtered to include only file names found in the external CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    fits_files = find_prism_fits(base_dir)
    muv_data = []
    
    # Get the set of filenames from the external CSV (no max SNR cut applied)
    valid_snr_files = get_snr_filter_files(snr_filter_path)
    
    total_files = len(fits_files)
    print(f"Total initial FITS files found: {total_files}")
    if valid_snr_files is not None:
         print(f"Processing {len(valid_snr_files)} files from external CSV.")

    # --- Progress Bar Integration ---
    for i, fits_path in enumerate(fits_files):
        file_name = fits_path.name
        progress = f"{i + 1}/{total_files} galaxy MUV computed"
        
        # Skip if filter is active and file name is NOT in the approved set
        if valid_snr_files is not None and file_name not in valid_snr_files:
            continue

        try:
            z = get_redshift_from_csv(csv_path, fits_path)
            
            if z is None or not np.isfinite(z):
                continue

            wave_obs, flux_obs, err_obs = read_observed_spectrum(fits_path)
            w_rest, f_rest, e_rest = get_rest_frame_spectrum(wave_obs, flux_obs, err_obs, z)
            
            muv = calculate_muv(w_rest, f_rest, z)
            
            if muv is not None and np.isfinite(muv):
                muv_data.append({'z': z, 'muv': muv, 'file': file_name})
            
        except Exception as e:
            # print(f"[{progress}] Error processing {fits_path.name}: {e}")
            pass 

    # --- Plotting ---
    if not muv_data:
        print("\nNo valid MUV data points to plot after filtering. Exiting.")
        return

    df_muv = pd.DataFrame(muv_data)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(df_muv['z'], df_muv['muv'], 
                s=30, alpha=0.7, edgecolors='k', color='dodgerblue')
    
    plt.xlabel("Redshift (z)", fontsize=14)
    plt.ylabel("Absolute UV Magnitude (M_UV) [AB mag]", fontsize=14) 
    plt.title(f"M_UV vs. Redshift (Filtered by CSV) (N={len(df_muv)})", fontsize=16)
    
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3)
    
    z_min, z_max = df_muv['z'].min(), df_muv['z'].max()
    plt.xlim(max(0.0, z_min - 0.2), z_max + 0.2)
    
    muv_min, muv_max = df_muv['muv'].min(), df_muv['muv'].max()
    y_buffer = (muv_max - muv_min) * 0.1 
    plt.ylim(muv_max + y_buffer, muv_min - y_buffer) 

    plt.tight_layout()
    
    plot_path = output_dir / "muv_vs_z_plot_filtered_final.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nPlot saved successfully to: {plot_path.resolve()}")
    
    csv_path_out = output_dir / "muv_z_results_filtered_final.csv"
    df_muv.to_csv(csv_path_out, index=False)
    print(f"MUV results saved to: {csv_path_out.resolve()}")


# ----------------------------------------------------------------------
## CLI EXECUTION
# ----------------------------------------------------------------------

if __name__ == "__main__":
    process_and_plot_muv_vs_z(
        SPECTRA_BASE_DIR, 
        CSV_PATH_GLOBAL, 
        OUTPUT_DIR,
        EXTERNAL_SNR_CSV_PATH
    )