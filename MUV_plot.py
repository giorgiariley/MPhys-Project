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
## MUV CALCULATION (Modified to include error)
# ----------------------------------------------------------------------

def calculate_integral_error(w_window: np.ndarray, e_window: np.ndarray) -> float:
    """
    Calculates the error on the integrated flux (dI) using quadrature: 
    dI^2 ~ Sum (e_i * dw_i)^2, where dw_i are the wavelength steps.
    """
    if len(w_window) < 2:
        return np.nan
        
    # Approximate wavelength steps dw_i
    dw = np.diff(w_window, prepend=w_window[0], append=w_window[-1])
    dw_i = (dw[:-1] + dw[1:]) / 2.0
    
    # Simple weighted quadrature sum (dI^2)
    dI_sq = np.sum((e_window * dw_i)**2)
    return np.sqrt(dI_sq)


def calculate_muv_and_error(w_rest: np.ndarray, f_rest_flambda: np.ndarray, 
                            e_rest_flambda: np.ndarray, z: float, 
                            w_min: float = W_UV_MIN, w_max: float = W_UV_MAX) -> tuple[Optional[float], Optional[float]]:
    """
    Calculates M_UV (Absolute AB Magnitude) and its error (Delta M_UV).
    """
    
    is_UV = (w_rest >= w_min) & (w_rest <= w_max)
    
    if not is_UV.any():
        return None, None
    
    w_window = w_rest[is_UV]
    f_window = f_rest_flambda[is_UV]
    e_window = e_rest_flambda[is_UV]
    
    valid_mask = np.isfinite(f_window) & np.isfinite(e_window) & (e_window >= 0)
    w_window = w_window[valid_mask]
    f_window = f_window[valid_mask]
    e_window = e_window[valid_mask]

    if len(w_window) < 2:
        return None, None

    # --- M_UV CALCULATION ---
    integral_flam = simpson(f_window, x=w_window)
    bandpass_width = w_max - w_min
    Flam_UV_mean = integral_flam / bandpass_width
    
    if Flam_UV_mean <= 0:
        return None, None
        
    lambda_eff = (w_min + w_max) / 2.0
    
    F_nu_cgs = Flam_UV_mean * (lambda_eff**2 / C_LIGHT_AA_PER_S)
    F_nu_Jy = F_nu_cgs / 1e-23
    
    m_UV_AB = -2.5 * np.log10(F_nu_Jy) + AB_MAG_ZP_JY 
    
    dL_pc = cosmo.luminosity_distance(z).to(au.pc).value
    DM = 5.0 * (np.log10(dL_pc) - 1.0) 
    K = -2.5 * np.log10(1 + z)
    
    M_UV_AB = m_UV_AB - DM - K

    # --- Delta M_UV CALCULATION ---
    integral_flam_err = calculate_integral_error(w_window, e_window)
    Flam_UV_mean_err = integral_flam_err / bandpass_width
    
    # Magnitude error formula: Delta M = 1.0857 * (Delta F / F)
    delta_M_UV = (2.5 / np.log(10)) * (Flam_UV_mean_err / Flam_UV_mean)
    
    if not np.isfinite(delta_M_UV) or delta_M_UV < 0:
        return M_UV_AB, None

    return M_UV_AB, delta_M_UV


# ----------------------------------------------------------------------
## BETA CALCULATION FUNCTIONS (Modified to include error)
# ----------------------------------------------------------------------

def beta_slope_power_law_func(log_w, log_a, beta):
    """Log-linear form for power law fit: log(F_lambda) = log(a) + beta * log(lambda)."""
    return log_a + beta * log_w

def sample_spectrum_C94(w_rest: np.ndarray, f_rest: np.ndarray, e_rest: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Samples the median F_lambda and the error on the median (sigma/sqrt(N)) 
    within the 10 standard C94 rest-frame UV filters.
    """
    sampled_waves = []
    sampled_fluxes = []
    sampled_errors = []

    for w_min, w_max in zip(LOWER_C94_FILT, UPPER_C94_FILT):
        mask = (w_rest >= w_min) & (w_rest <= w_max)
        
        if np.sum(mask) == 0:
            continue
            
        f_window = f_rest[mask]
        e_window = e_rest[mask]
        
        valid_points = np.isfinite(f_window) & np.isfinite(e_window) & (f_window > 0)
        
        if np.sum(valid_points) < 2:
            continue

        f_window = f_window[valid_points]
        e_window = e_window[valid_points]
        N = len(f_window)

        median_flux = np.median(f_window)
        # Error on the sampled flux: use the standard error of the mean error 
        median_error = np.median(e_window) / np.sqrt(N) 
        
        if median_flux > 0:
            sampled_waves.append((w_min + w_max) / 2.0)
            sampled_fluxes.append(median_flux)
            sampled_errors.append(median_error)

    return np.array(sampled_waves), np.array(sampled_fluxes), np.array(sampled_errors)


def calculate_beta_and_error(w_rest: np.ndarray, f_rest_flambda: np.ndarray, e_rest_flambda: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """
    Calculates the UV continuum slope (beta) and its error by fitting a power law 
    using fluxes and errors sampled in the C94 filter bands.
    """
    # 1. Sample the spectrum using the C94 filter windows
    sampled_waves, sampled_fluxes, sampled_errors = sample_spectrum_C94(w_rest, f_rest_flambda, e_rest_flambda)

    # 2. Check for sufficient data points
    if len(sampled_waves) < 2:
        return None, None

    # 3. Calculate errors in log space: Delta log(F) = 1.0857 * (Delta F / F)
    valid_mask = (sampled_fluxes > 0) & (sampled_errors > 0) & np.isfinite(sampled_errors)
    if np.sum(valid_mask) < 2:
         return None, None

    sampled_waves = sampled_waves[valid_mask]
    sampled_fluxes = sampled_fluxes[valid_mask]
    sampled_errors = sampled_errors[valid_mask]
    
    # Delta log(F) = (2.5 / ln(10)) * (Delta F / F)
    log_flux_errors = (2.5 / np.log(10)) * (sampled_errors / sampled_fluxes)
    
    # 4. Perform the log-linear fit
    try:
        popt, pcov = curve_fit(
            beta_slope_power_law_func, 
            np.log10(sampled_waves), 
            np.log10(sampled_fluxes),
            sigma=log_flux_errors,       # Pass uncertainties
            absolute_sigma=True,         # Treat sigma as absolute errors
            p0=[0, -2.0], 
            maxfev=5000
        )
        
        beta = popt[1]
        
        # Calculate error on beta: sqrt(covariance matrix diagonal element pcov[1, 1])
        beta_err = np.sqrt(pcov[1, 1])
        
        if np.isfinite(beta) and np.isfinite(beta_err):
             return beta, beta_err
        else:
            return None, None

    except Exception:
        return None, None # Fit failed

# ----------------------------------------------------------------------
## MAIN EXECUTION AND PLOTTING (Modified to improve aesthetics)
# ----------------------------------------------------------------------

def process_and_plot_beta_vs_z(base_dir: str, csv_path: Path, output_dir: Path, snr_filter_path: Path):
    """
    Main function to calculate MUV/Beta/Errors, plot results, and save data.
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
            
            # Use the new functions to get values AND errors
            muv, muv_err = calculate_muv_and_error(w_rest, f_rest, e_rest, z)
            beta, beta_err = calculate_beta_and_error(w_rest, f_rest, e_rest)
            
            # Only keep points where both values AND errors are valid
            valid_muv_beta = (muv is not None and beta is not None and 
                              muv_err is not None and beta_err is not None and
                              np.isfinite(muv) and np.isfinite(beta) and 
                              np.isfinite(muv_err) and np.isfinite(beta_err))

            if valid_muv_beta:
                muv_beta_data.append({
                    'z': z, 
                    'muv': muv, 
                    'beta': beta, 
                    'muv_err': muv_err,
                    'beta_err': beta_err,
                    'file': file_name
                })
            
        except Exception as e:
            # print(f"[{progress}] Error processing {fits_path.name}: {e}")
            pass 

    # --- Plotting ---
    if not muv_beta_data:
        print("\nNo valid MUV/Beta data points to plot after filtering. Exiting.")
        return

    df_data = pd.DataFrame(muv_beta_data)
    
    # Define plotting parameters for aesthetic improvement
    z_errs = 0.0 # No redshift error available
    MARKER_COLOR = '#1b7b3b' # Darker green for contrast
    ERROR_BAR_COLOR = '#666666' # Light gray for minimal dominance
    MARKER_SIZE = 4
    CAP_SIZE = 2
    LINE_WIDTH = 0.5
    ALPHA_POINTS = 0.8
    ALPHA_ERRORS = 0.4
    
    # Set the style for better readability
    plt.style.use('default') 

    # --------------------------------------
    # 1. Plot Beta vs. M_UV (X and Y errors)
    # --------------------------------------
    plt.figure(figsize=(10, 7))
    # Use errorbar with reduced clutter and light error bar color
    plt.errorbar(df_data['muv'], df_data['beta'], 
                 xerr=df_data['muv_err'], yerr=df_data['beta_err'],
                 fmt='o', 
                 markersize=MARKER_SIZE, 
                 capsize=CAP_SIZE, 
                 elinewidth=LINE_WIDTH, # Thin error lines
                 alpha=ALPHA_ERRORS,    # Transparent error bars
                 ecolor=ERROR_BAR_COLOR, 
                 markerfacecolor=MARKER_COLOR, # Marker color
                 markeredgecolor='k', 
                 markeredgewidth=0.5,
                 zorder=1) # Error bars below points (for clarity)
    
    # Overlay the points again without error bars but with full opacity
    # This makes the centers clearer while errors fade into the background
    plt.scatter(df_data['muv'], df_data['beta'], 
                s=MARKER_SIZE*5, alpha=ALPHA_POINTS, 
                color=MARKER_COLOR, edgecolors='k', linewidths=0.5, zorder=2)
    
    plt.xlabel("Absolute UV Magnitude ($M_{UV}$) [AB mag]", fontsize=14)
    plt.ylabel("UV Continuum Slope ($\\beta$)", fontsize=14)
    plt.title(f"UV Slope vs. M_UV (N={len(df_data)})", fontsize=16)
    plt.ylim(-3.0, 1.0)
    plt.grid(alpha=0.2, linestyle='--') # Lighter grid lines
    plt.tight_layout()
    
    plot_path_beta_muv = output_dir / "beta_vs_muv_plot_improved.png"
    plt.savefig(plot_path_beta_muv, dpi=300) # Increased DPI for better image quality
    print(f"\nSaved improved Beta vs M_UV plot successfully to: {plot_path_beta_muv.resolve()}")
    
    # 2. Plot M_UV vs. z (Y error only)
    plt.figure(figsize=(10, 7))
    plt.errorbar(df_data['z'], df_data['muv'], 
                 xerr=z_errs, yerr=df_data['muv_err'],
                 fmt='o', markersize=MARKER_SIZE, capsize=CAP_SIZE, elinewidth=LINE_WIDTH,
                 alpha=ALPHA_ERRORS, ecolor=ERROR_BAR_COLOR, 
                 markerfacecolor='dodgerblue', markeredgecolor='k', markeredgewidth=0.5,
                 zorder=1)
    
    plt.scatter(df_data['z'], df_data['muv'], 
                s=MARKER_SIZE*5, alpha=ALPHA_POINTS, 
                color='dodgerblue', edgecolors='k', linewidths=0.5, zorder=2)
                 
    plt.xlabel("Redshift (z)", fontsize=14)
    plt.ylabel("Absolute UV Magnitude ($M_{UV}$) [AB mag]", fontsize=14) 
    plt.title(f"M_UV vs. Redshift (N={len(df_data)})", fontsize=16)
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / "muv_vs_z_plot_improved.png", dpi=300)
    print(f"Saved improved MUV vs z plot to: {(output_dir / 'muv_vs_z_plot_improved.png').resolve()}")

    # 3. Plot Beta vs. z (Y error only)
    plt.figure(figsize=(10, 7))
    plt.errorbar(df_data['z'], df_data['beta'], 
                 xerr=z_errs, yerr=df_data['beta_err'],
                 fmt='o', markersize=MARKER_SIZE, capsize=CAP_SIZE, elinewidth=LINE_WIDTH,
                 alpha=ALPHA_ERRORS, ecolor=ERROR_BAR_COLOR, 
                 markerfacecolor='firebrick', markeredgecolor='k', markeredgewidth=0.5,
                 zorder=1)
                 
    plt.scatter(df_data['z'], df_data['beta'], 
                s=MARKER_SIZE*5, alpha=ALPHA_POINTS, 
                color='firebrick', edgecolors='k', linewidths=0.5, zorder=2)
    
    plt.xlabel("Redshift (z)", fontsize=14)
    plt.ylabel("UV Continuum Slope ($\\beta$)", fontsize=14)
    plt.title(f"UV Slope ($\\beta$) vs. Redshift (N={len(df_data)})", fontsize=16)
    
    plt.ylim(-3.0, 1.0) 
    plt.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    
    plot_path_beta = output_dir / "beta_vs_z_plot_improved.png"
    plt.savefig(plot_path_beta, dpi=300)
    print(f"Saved improved Beta vs z plot successfully to: {plot_path_beta.resolve()}")
    
    # Save the final combined data including errors (using the original filename convention)
    csv_path_out = output_dir / "muv_beta_z_results_with_errs.csv"
    df_data.to_csv(csv_path_out, index=False)
    print(f"MUV and Beta results with errors saved to: {csv_path_out.resolve()}")


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
