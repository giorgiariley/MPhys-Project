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
from scipy.optimize import curve_fit

# --- CONSTANTS ---
C_LIGHT_AA_PER_S = 2.99792458e18 
AB_MAG_ZP_JY = 8.90
W_UV_MIN = 1350.0 
W_UV_MAX = 1800.0
cosmo = Planck18

# --- BETA CALCULATION CONSTANTS (Calzetti+1994 Filter Definitions) ---
# Use the explicit C94 filter boundaries provided by the user. These are REST-FRAME wavelengths.
LOWER_C94_FILT = np.array([1268., 1309., 1342., 1407., 1562., 1677., 1760., 1866., 1930., 2400.])
UPPER_C94_FILT = np.array([1284., 1316., 1371., 1515., 1583., 1740., 1833., 1890., 1950., 2580.])

# --- CONFIGURATION ---
SPECTRA_BASE_DIR = "/raid/scratch/work/Griley/GALFIND_WORK/Spectra/2D"
CSV_PATH_GLOBAL = Path("./mphys_GOODS_S_exposures.csv")
OUTPUT_DIR = Path.cwd() / "MUV_plots"
EXTERNAL_SNR_CSV_PATH = Path("/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/specFitMSA/src/uv_snr_5plus.csv") 

# ----------------------------------------------------------------------
## FILTERING HELPERS (Reused)
# ----------------------------------------------------------------------

def get_snr_filter_files(snr_csv_path: Path) -> Set[str]:
    """
    Loads the external SNR CSV and returns a set of filenames present in the file.
    """
    if not snr_csv_path.exists():
        print(f"Warning: SNR filter file not found at {snr_csv_path}. Proceeding without SNR filter.")
        return None 
        
    try:
        df_snr = pd.read_csv(snr_csv_path, usecols=['file'])
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
## MUV CALCULATION (Reused)
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
    DM = 5.0 * (np.log10(dL_pc) - 1.0) 
    K = -2.5 * np.log10(1 + z)
    
    M_UV_AB = m_UV_AB - DM - K
    
    return M_UV_AB

# ----------------------------------------------------------------------
## BETA CALCULATION FUNCTIONS (Reused)
# ----------------------------------------------------------------------

def beta_slope_power_law_func(log_w, log_a, beta):
    """Log-linear form for power law fit: log(F_lambda) = log(a) + beta * log(lambda)."""
    return log_a + beta * log_w

def sample_spectrum_C94(w_rest: np.ndarray, f_rest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Samples the median F_lambda within the 10 standard C94 rest-frame UV filters.
    """
    sampled_waves = []
    sampled_fluxes = []

    # Iterate through the 10 C94 filters
    for w_min, w_max in zip(LOWER_C94_FILT, UPPER_C94_FILT):
        mask = (w_rest >= w_min) & (w_rest <= w_max)
        
        if np.sum(mask) == 0:
            continue
            
        f_window = f_rest[mask]
        
        # Use median flux for robustness and require positive flux
        median_flux = np.median(f_window[np.isfinite(f_window) & (f_window > 0)])
        
        if median_flux > 0:
            # Use the average wavelength of the bandpass as the sampling point
            sampled_waves.append((w_min + w_max) / 2.0)
            sampled_fluxes.append(median_flux)

    return np.array(sampled_waves), np.array(sampled_fluxes)


def calculate_beta(w_rest: np.ndarray, f_rest_flambda: np.ndarray) -> Optional[float]:
    """
    Calculates the UV continuum slope (beta) by fitting a power law to fluxes 
    sampled in the C94 filter bands.
    """
    # 1. Sample the spectrum using the C94 filter windows
    sampled_waves, sampled_fluxes = sample_spectrum_C94(w_rest, f_rest_flambda)

    # 2. Check for sufficient data points
    if len(sampled_waves) < 2:
        return None

    # 3. Perform the log-linear fit: log(F_lambda) = log(a) + beta * log(lambda)
    try:
        popt, pcov = curve_fit(
            beta_slope_power_law_func, 
            np.log10(sampled_waves), 
            np.log10(sampled_fluxes),
            p0=[0, -2.0], # Initial guess for [log(a), beta]
            maxfev=5000
        )
        
        # The second parameter is beta
        beta = popt[1]
        
        if np.isfinite(beta):
             return beta
        else:
            return None

    except Exception:
        return None # Fit failed due to singular matrix, extreme values, etc.

# ----------------------------------------------------------------------
## MAIN EXECUTION AND PLOTTING
# ----------------------------------------------------------------------

def process_and_plot_beta_vs_z(base_dir: str, csv_path: Path, output_dir: Path, snr_filter_path: Path):
    """
    Main function to calculate MUV/Beta, plot results, and save data.
    """
    os.makedirs(output_dir, exist_ok=True)
    fits_files = find_prism_fits(base_dir)
    muv_beta_data = [] 
    
    valid_snr_files = get_snr_filter_files(snr_filter_path)
    total_files = len(fits_files)
    print(f"Total initial FITS files found: {total_files}")

    for i, fits_path in enumerate(fits_files):
        file_name = fits_path.name
        progress = f"{i + 1}/{total_files} computed"
        
        if valid_snr_files is not None and file_name not in valid_snr_files:
            continue

        try:
            z = get_redshift_from_csv(csv_path, fits_path)
            
            if z is None or not np.isfinite(z):
                continue

            wave_obs, flux_obs, err_obs = read_observed_spectrum(fits_path)
            w_rest, f_rest, e_rest = get_rest_frame_spectrum(wave_obs, flux_obs, err_obs, z)
            
            muv = calculate_muv(w_rest, f_rest, z)
            beta = calculate_beta(w_rest, f_rest)
            
            if muv is not None and beta is not None and np.isfinite(muv) and np.isfinite(beta):
                muv_beta_data.append({'z': z, 'muv': muv, 'beta': beta, 'file': file_name})
            
        except Exception as e:
            # print(f"[{progress}] Error processing {fits_path.name}: {e}")
            pass 

    # --- Plotting ---
    if not muv_beta_data:
        print("\nNo valid MUV/Beta data points to plot after filtering. Exiting.")
        return

    df_data = pd.DataFrame(muv_beta_data)
    
    # --------------------------------------
    # 1. Plot Beta vs. M_UV (NEW FINAL PLOT)
    # --------------------------------------
    plt.figure(figsize=(10, 7))
    plt.scatter(df_data['muv'], df_data['beta'], 
                s=30, alpha=0.7, edgecolors='k', color='darkgreen')
    
    # M_UV is on the x-axis, and we invert the x-axis as is standard for magnitude plots
    plt.xlabel("Absolute UV Magnitude (M_UV) [AB mag]", fontsize=14)
    plt.ylabel("UV Continuum Slope (Beta)", fontsize=14)
    plt.title(f"UV Slope (Beta) vs. M_UV (N={len(df_data)})", fontsize=16)
    #plt.gca().invert_xaxis() # Brighter objects (more negative MUV) should be to the left
    plt.ylim(-3.0, 1.0) # Standard Beta range
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plot_path_beta_muv = output_dir / "beta_vs_muv_plot.png"
    plt.savefig(plot_path_beta_muv, dpi=200)
    print(f"\nSaved Beta vs M_UV plot successfully to: {plot_path_beta_muv.resolve()}")
    
    # 2. Plot M_UV vs. z (Reused)
    plt.figure(figsize=(10, 7))
    plt.scatter(df_data['z'], df_data['muv'], 
                s=30, alpha=0.7, edgecolors='k', color='dodgerblue')
    plt.xlabel("Redshift (z)", fontsize=14)
    plt.ylabel("Absolute UV Magnitude (M_UV) [AB mag]", fontsize=14) 
    plt.title(f"M_UV vs. Redshift (N={len(df_data)})", fontsize=16)
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "muv_vs_z_plot_filtered_final.png", dpi=200)
    print(f"Saved MUV vs z plot to: {(output_dir / 'muv_vs_z_plot_filtered_final.png').resolve()}")

    # 3. Plot Beta vs. z (Reused)
    plt.figure(figsize=(10, 7))
    plt.scatter(df_data['z'], df_data['beta'], 
                s=30, alpha=0.7, edgecolors='k', color='firebrick')
    
    plt.xlabel("Redshift (z)", fontsize=14)
    plt.ylabel("UV Continuum Slope (Beta)", fontsize=14)
    plt.title(f"UV Slope (Beta) vs. Redshift (N={len(df_data)})", fontsize=16)
    
    plt.ylim(-3.0, 1.0) 
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plot_path_beta = output_dir / "beta_vs_z_plot.png"
    plt.savefig(plot_path_beta, dpi=200)
    print(f"Saved Beta vs z plot successfully to: {plot_path_beta.resolve()}")
    
    # Save the final combined data
    csv_path_out = output_dir / "muv_beta_z_results.csv"
    df_data.to_csv(csv_path_out, index=False)
    print(f"MUV and Beta results saved to: {csv_path_out.resolve()}")


# ----------------------------------------------------------------------
## CLI EXECUTION
# ----------------------------------------------------------------------

if __name__ == "__main__":
    process_and_plot_beta_vs_z(
        SPECTRA_BASE_DIR, 
        CSV_PATH_GLOBAL, 
        OUTPUT_DIR,
        EXTERNAL_SNR_CSV_PATH
    )