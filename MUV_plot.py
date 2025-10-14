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

# NEW CONSTANT FOR PHOTOMETRIC CATALOGUE
PHOTOMETRY_CATALOGUE_PATH = "/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-South/(0.32)as/JADES-DR3-GS-South_MASTER_Sel-F277W+F356W+F444W_v13.fits"
# PHOTOMETRY_HDU constant removed as two HDUs are now required.


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

def load_photometric_data(fits_path: str) -> Optional[pd.DataFrame]:
    """
    Loads photometric Beta and M_UV data from two different HDUs (4 and 6), 
    including 16th/84th percentiles for errors.
    """
    try:
        if not Path(fits_path).exists():
            print(f"Warning: Photometric catalogue not found at {fits_path}.")
            return None
            
        # --- HDU Indices (0-based) ---
        MUV_HDU_INDEX = 4  # Corresponds to HDU 4
        BETA_HDU_INDEX = 6 # Corresponds to HDU 6

        # Open FITS file with parameters to handle byte order issues
        with fits.open(fits_path, do_not_scale_image_data=True, uint=False) as hdul:
            
            # Load Data from two different HDUs
            data_muv = hdul[MUV_HDU_INDEX].data 
            data_beta = hdul[BETA_HDU_INDEX].data 
            
            # --- Column Names ---
            MUV_COL = 'M_UV_50'
            BETA_COL = 'beta_[1250,3000]AA_0.32as' 
            
            # Use data_muv for M_UV columns and data_beta for Beta columns
            try:
                # MUV columns from HDU 4 data
                muv_50 = data_muv[MUV_COL].byteswap().newbyteorder()
                muv_16 = data_muv['M_UV_16'].byteswap().newbyteorder()
                muv_84 = data_muv['M_UV_84'].byteswap().newbyteorder()
                
                # BETA columns from HDU 6 data
                beta_50 = data_beta[BETA_COL].byteswap().newbyteorder()
                beta_16 = data_beta[BETA_COL + '_l1'].byteswap().newbyteorder()
                beta_84 = data_beta[BETA_COL + '_u1'].byteswap().newbyteorder()
            except AttributeError:
                # Fallback if byteswap() fails
                print("Warning: Byte-swapping failed, attempting to read arrays directly.")
                muv_50 = data_muv[MUV_COL]
                muv_16 = data_muv['M_UV_16']
                muv_84 = data_muv['M_UV_84']
                
                beta_50 = data_beta[BETA_COL]
                beta_16 = data_beta[BETA_COL + '_l1']
                beta_84 = data_beta[BETA_COL + '_u1']
            
            # Ensure catalogs have the same length before proceeding
            if len(muv_50) != len(beta_50):
                 print(f"Error: M_UV (HDU {MUV_HDU_INDEX+1}) and Beta (HDU {BETA_HDU_INDEX+1}) tables have different lengths. Aborting load.")
                 return None


            # Calculate ASYMMETRIC ERRORS relative to the median (50th percentile)
            # M_UV is a magnitude: brighter (more negative) is usually smaller index (16th)
            muv_err_low = muv_50 - muv_16  # Distance from median to 16th percentile (lower magnitude side)
            muv_err_high = muv_84 - muv_50 # Distance from median to 84th percentile (higher magnitude side)
            
            # Beta is a slope: lower index (16th) is lower beta
            beta_err_low = beta_16 
            beta_err_high = beta_84
            
            # Combine into a DataFrame
            df_photo = pd.DataFrame({
                'muv': muv_50, 
                'beta': beta_50,
                'muv_err_low': muv_err_low,
                'muv_err_high': muv_err_high,
                'beta_err_low': beta_err_low,
                'beta_err_high': beta_err_high,
            })
            
            # Filter out NaNs/Infs from primary values and ensure errors are sensible
            df_photo = df_photo.replace([np.inf, -np.inf], np.nan).dropna(subset=['muv', 'beta'])
            
            print(f"Loaded {len(df_photo)} valid photometric Beta/M_UV points using HDU {MUV_HDU_INDEX+1} (M_UV) and HDU {BETA_HDU_INDEX+1} (Beta).")
            return df_photo
            
    except Exception as e:
        print(f"Error loading photometric catalogue from {fits_path}: {e}")
        return None


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
## MAIN EXECUTION AND PLOTTING (Modified to improve aesthetics and add photometry)
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
    
    # NEW STEP: Load Photometric Data (now without HDU index argument)
    df_photo = load_photometric_data(PHOTOMETRY_CATALOGUE_PATH)
    
    # Define plotting parameters for aesthetic improvement
    z_errs = 0.0 # No redshift error available
    MARKER_COLOR_SPEC = '#1b7b3b' # Darker green for spectroscopic contrast
    
    # MODIFIED: Changed color to red and increased alpha for visibility
    MARKER_COLOR_PHOTO = '#dc3545' # Red for photometric contrast
    MARKER_SIZE_PHOTO = 3.0 # Slightly larger marker size for photometry
    ALPHA_POINTS_PHOTO = 0.5 # Increased opacity
    
    ERROR_BAR_COLOR = '#666666' 
    MARKER_SIZE = 4
    CAP_SIZE = 2
    LINE_WIDTH = 0.5
    ALPHA_POINTS_SPEC = 0.8
    ALPHA_ERRORS = 0.4
    
    # Set the style for better readability
    plt.style.use('default') 

    # ----------------------------------------------------
    # 1. Plot Beta vs. M_UV (Spectroscopy + Photometry Overlay)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 7))
    
    # Layer 1: Photometric error bars (asymmetric)
    if df_photo is not None and not df_photo.empty:
        # Assemble the asymmetric error arrays for errorbar plotting
        muv_errs_photo = [df_photo['muv_err_low'].values, df_photo['muv_err_high'].values]
        beta_errs_photo = [df_photo['beta_err_low'].values, df_photo['beta_err_high'].values]
        
        # Plot Photometric Error Bars
        plt.errorbar(df_photo['muv'], df_photo['beta'],
                     xerr=muv_errs_photo,
                     yerr=beta_errs_photo,
                     fmt='o', 
                     markersize=MARKER_SIZE_PHOTO,
                     capsize=1, # Very small capsize for densely packed errors
                     elinewidth=0.2, # Very thin lines
                     alpha=ALPHA_POINTS_PHOTO, # Use point alpha for transparency
                     ecolor=MARKER_COLOR_PHOTO, # Pale red lines
                     markerfacecolor= MARKER_COLOR_PHOTO, # No marker center needed
                     markeredgecolor=MARKER_COLOR_PHOTO,
                     markeredgewidth=0.3, # Thin edge around the marker
                     label='Photometry w/ Errors', 
                     zorder=0) 

    # Layer 2: Spectroscopic error bars (uncertainty)
    plt.errorbar(df_data['muv'], df_data['beta'], 
                 xerr=df_data['muv_err'], yerr=df_data['beta_err'],
                 fmt='o', 
                 markersize=MARKER_SIZE, 
                 capsize=CAP_SIZE, 
                 elinewidth=LINE_WIDTH,
                 alpha=ALPHA_ERRORS,
                 ecolor=ERROR_BAR_COLOR, 
                 markerfacecolor=MARKER_COLOR_SPEC, 
                 markeredgecolor='k', 
                 markeredgewidth=0.5,
                 label='Spectroscopy w/ Errors',
                 zorder=1) 
    
    # Layer 3: Spectroscopic points (central measurement)
    plt.scatter(df_data['muv'], df_data['beta'], 
                s=MARKER_SIZE*5, alpha=ALPHA_POINTS_SPEC, 
                color=MARKER_COLOR_SPEC, edgecolors='k', linewidths=0.5, 
                zorder=2) # Put on top of error bars
    
    # MODIFIED: Removed LaTeX formatting from xlabel
    plt.xlabel("Absolute UV Magnitude (M_UV) [AB mag]", fontsize=14)
    # MODIFIED: Removed LaTeX formatting from ylabel
    plt.ylabel("UV Continuum Slope (Beta)", fontsize=14)
    # UPDATED: Added _power to the title
    plt.title(f"UV Slope vs. M_UV (Spectro N={len(df_data)})_power", fontsize=16)
    
    # X-axis not inverted, as requested
    plt.ylim(-3.0, 1.0) 
    plt.grid(alpha=0.2, linestyle='--') 
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    
    # UPDATED: Added _power to the output filename
    plot_path_beta_muv = output_dir / "beta_vs_muv_plot_with_photo_power.png" 
    plt.savefig(plot_path_beta_muv, dpi=300) 
    print(f"\nSaved Beta vs M_UV plot with photometric overlay successfully to: {plot_path_beta_muv.resolve()}")
    
    # 2. Plot M_UV vs. z (Y error only)
    plt.figure(figsize=(10, 7))
    plt.errorbar(df_data['z'], df_data['muv'], 
                 xerr=z_errs, yerr=df_data['muv_err'],
                 fmt='o', markersize=MARKER_SIZE, capsize=CAP_SIZE, elinewidth=LINE_WIDTH,
                 alpha=ALPHA_ERRORS, ecolor=ERROR_BAR_COLOR, 
                 markerfacecolor='dodgerblue', markeredgecolor='k', markeredgewidth=0.5,
                 zorder=1)
    
    plt.scatter(df_data['z'], df_data['muv'], 
                s=MARKER_SIZE*5, alpha=ALPHA_POINTS_SPEC, 
                color='dodgerblue', edgecolors='k', linewidths=0.5, zorder=2)
                 
    plt.xlabel("Redshift (z)", fontsize=14)
    # MODIFIED: Removed LaTeX formatting from ylabel
    plt.ylabel("Absolute UV Magnitude (M_UV) [AB mag]", fontsize=14) 
    # MODIFIED: Removed LaTeX formatting from title
    plt.title(f"M_UV vs. Redshift (N={len(df_data)})", fontsize=16)
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / "muv_vs_z_plot_improved_v2.png", dpi=300)
    print(f"Saved MUV vs z plot to: {(output_dir / 'muv_vs_z_plot_improved_v2.png').resolve()}")

    # 3. Plot Beta vs. z (Y error only)
    plt.figure(figsize=(10, 7))
    plt.errorbar(df_data['z'], df_data['beta'], 
                 xerr=z_errs, yerr=df_data['beta_err'],
                 fmt='o', markersize=MARKER_SIZE, capsize=CAP_SIZE, elinewidth=LINE_WIDTH,
                 alpha=ALPHA_ERRORS, ecolor=ERROR_BAR_COLOR, 
                 markerfacecolor='firebrick', markeredgecolor='k', markeredgewidth=0.5,
                 zorder=1)
                 
    plt.scatter(df_data['z'], df_data['beta'], 
                s=MARKER_SIZE*5, alpha=ALPHA_POINTS_SPEC, 
                color='firebrick', edgecolors='k', linewidths=0.5, zorder=2)
    
    plt.xlabel("Redshift (z)", fontsize=14)
    # MODIFIED: Removed LaTeX formatting from ylabel
    plt.ylabel("UV Continuum Slope (Beta)", fontsize=14)
    # MODIFIED: Removed LaTeX formatting from title
    plt.title(f"UV Slope (Beta) vs. Redshift (N={len(df_data)})", fontsize=16)
    
    plt.ylim(-3.0, 1.0) 
    plt.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    
    plot_path_beta = output_dir / "beta_vs_z_plot_improved_v2.png"
    plt.savefig(plot_path_beta, dpi=300)
    print(f"Saved Beta vs z plot successfully to: {plot_path_beta.resolve()}")
    
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
