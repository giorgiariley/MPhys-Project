import csv
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional, List, Dict, Any
import pandas as pd
import astropy.units as au
from astropy.cosmology import Planck18
from scipy.integrate import simpson

# --- CONSTANTS ---
# Speed of light in Angstroms per second
C_LIGHT_AA_PER_S = 2.99792458e18 
# AB Magnitude Zero Point for F_nu in Jy: m_AB = -2.5 * log10(F_nu) + 8.90
AB_MAG_ZP_JY = 8.90

# MUV Wavelength Window for integration (Supervisor's values)
W_UV_MIN = 1350.0 # Angstroms
W_UV_MAX = 1800.0 # Angstroms

# Cosmological model for distance calculation
cosmo = Planck18

# --- CONFIGURATION (Reused from your script) ---
SPECTRA_BASE_DIR = "/raid/scratch/work/Griley/GALFIND_WORK/Spectra/2D"
CSV_PATH_GLOBAL = Path("./mphys_GOODS_S_exposures.csv")
OUTPUT_DIR = Path.cwd() / "MUV_plots"

# ----------------------------------------------------------------------
## I/O & TRANSFORMATION HELPERS (Modified: get_rest_frame_spectrum)
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
        wave_obs = np.array(data["wave"])       # typically in µm
        flux_obs = np.array(data["flux"])       # typically in µJy
        err_obs = np.array(data["err"] if "err" in data.columns.names else data["full_err"])
    return wave_obs, flux_obs, err_obs

def get_rest_frame_spectrum(wave_obs_um: np.ndarray, flux_obs_uJy: np.ndarray, 
                            err_obs_uJy: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts observed spectrum (µm, µJy) to rest-frame (Å, erg/s/cm2/Å).
    This output is used for the M_UV calculation.
    """
    lam_obs_AA  = wave_obs_um * 1e4
    lam_rest_AA = lam_obs_AA / (1.0 + z)

    # Fν_obs(µJy) -> Fλ_obs(cgs)
    factor = 1e-29 * C_LIGHT_AA_PER_S # µJy->cgs_nu then *c (Å/s)
    Flam_obs = flux_obs_uJy * factor / (lam_obs_AA**2)
    Elam_obs = err_obs_uJy   * factor / (lam_obs_AA**2)

    w = lam_rest_AA
    # Fλ_obs -> Fλ_rest: Fλ_rest = Fλ_obs * (1+z)^2
    f = (1.0 + z)**2 * Flam_obs
    e = (1.0 + z)**2 * Elam_obs

    # Basic cleanup: sorting and filtering non-finite points
    idx = np.argsort(w)
    w, f, e = w[idx], f[idx], e[idx]
    keep = np.isfinite(w) & np.isfinite(f) & np.isfinite(e) & (e >= 0)
    
    return w[keep], f[keep], e[keep]

# ----------------------------------------------------------------------
## MUV CALCULATION (MODIFIED TO USE INTEGRATION METHOD)
# ----------------------------------------------------------------------

def calculate_muv(w_rest: np.ndarray, f_rest_flambda: np.ndarray, z: float, 
                  w_min: float = W_UV_MIN, w_max: float = W_UV_MAX) -> Optional[float]:
    """
    Calculates M_UV (Absolute AB Magnitude) using the integrated flux method.
    Fλ is integrated over the rest-frame window [w_min, w_max].
    """
    
    # 1. Select the rest-frame UV integration window
    is_UV = (w_rest >= w_min) & (w_rest <= w_max)
    
    if not is_UV.any():
        return None
    
    w_window = w_rest[is_UV]
    f_window = f_rest_flambda[is_UV]
    
    # Check for valid data in the window
    valid_mask = np.isfinite(f_window)
    if not valid_mask.any():
        return None

    w_window = w_window[valid_mask]
    f_window = f_window[valid_mask]

    # Re-check: need at least 2 points for Simpson's Rule
    if len(w_window) < 2:
        return None

    # 2. Integrate F_lambda across the UV window (erg / s / cm^2)
    # The integral gives total flux, not flux density.
    integral_flam = simpson(f_window, x=w_window)
    
    # 3. Calculate the Mean Flux Density (Flam_UV) over the bandpass
    # Flux Density = Integrated Flux / Bandpass Width (in rest frame)
    bandpass_width = w_max - w_min
    Flam_UV_mean = integral_flam / bandpass_width
    
    # 4. Convert Mean F_lambda to F_nu in Jy (for m_AB calculation)
    # F_nu = F_lambda * (lambda_eff^2 / c)
    # Supervisor's code uses Astropy units for the conversion, which we'll mimic:
    
    # Calculate the effective wavelength (used for the Fλ -> Fν conversion factor)
    lambda_eff = (w_min + w_max) / 2.0
    
    # Conversion factor from Fλ (cgs) to Fν (cgs)
    F_nu_cgs = Flam_UV_mean * (lambda_eff**2 / C_LIGHT_AA_PER_S)

    # Conversion to Jy (1 Jy = 1e-23 erg/s/cm2/Hz)
    F_nu_Jy = F_nu_cgs / 1e-23
    
    if F_nu_Jy <= 0:
        return None
    
    # 5. Calculate m_UV (Apparent AB Magnitude)
    m_UV_AB = -2.5 * np.log10(F_nu_Jy) + AB_MAG_ZP_JY 
    
    # 6. Calculate DM (Distance Modulus) and K-correction
    dL_pc = cosmo.luminosity_distance(z).to(au.pc).value # Luminosity distance [pc]
    DM = 5.0 * (np.log10(dL_pc) - 1.0) # Distance Modulus
    
    # K-correction: K = -2.5 * log10(1+z)
    K = -2.5 * np.log10(1 + z)
    
    # 7. Calculate M_UV (Absolute AB Magnitude)
    # M_UV_AB = m_UV_AB - DM - K 
    M_UV_AB = m_UV_AB - DM - K
    
    return M_UV_AB

# ----------------------------------------------------------------------
## MAIN EXECUTION AND PLOTTING (No change to logic, just uses new MUV)
# ----------------------------------------------------------------------

def process_and_plot_muv_vs_z(base_dir: str, csv_path: Path, output_dir: Path):
    """
    Main function to calculate MUV and plot MUV vs z for all prism spectra.
    """
    os.makedirs(output_dir, exist_ok=True)
    fits_files = find_prism_fits(base_dir)
    muv_data = []
    
    total_files = len(fits_files)
    print(f"Found {total_files} files to process for MUV calculation.")
    
    # --- Progress Bar Integration ---
    for i, fits_path in enumerate(fits_files):
        file_name = fits_path.name
        progress = f"{i + 1}/{total_files} galaxy MUV computed"
        
        try:
            z = get_redshift_from_csv(csv_path, fits_path)
            
            if z is None or not np.isfinite(z):
                print(f"[{progress}] Skipping {file_name}: Redshift not found or invalid.")
                continue

            wave_obs, flux_obs, err_obs = read_observed_spectrum(fits_path)
            w_rest, f_rest, e_rest = get_rest_frame_spectrum(wave_obs, flux_obs, err_obs, z)
            
            muv = calculate_muv(w_rest, f_rest, z)
            
            if muv is not None and np.isfinite(muv):
                muv_data.append({'z': z, 'muv': muv, 'file': file_name})
                # print(f"[{progress}] Calculated MUV for {file_name} (z={z:.3f}): MUV={muv:.2f}")
            # else:
                # print(f"[{progress}] Skipping {file_name}: MUV calculation failed (non-positive flux).")
                
        except Exception as e:
            print(f"[{progress}] Error processing {fits_path.name}: {e}")
            
    # --- Plotting ---
    if not muv_data:
        print("\nNo valid MUV data points to plot. Exiting.")
        return

    df_muv = pd.DataFrame(muv_data)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(df_muv['z'], df_muv['muv'], 
                s=30, alpha=0.7, edgecolors='k', color='dodgerblue')
    
    plt.xlabel("Redshift (z)", fontsize=14)
    plt.ylabel("Absolute UV Magnitude (M_UV) [AB mag]", fontsize=14) 
    plt.title(f"M_UV vs. Redshift for Detected Spectra (N={len(df_muv)})", fontsize=16)
    
    # Invert the Y-axis
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3)
    
    z_min, z_max = df_muv['z'].min(), df_muv['z'].max()
    plt.xlim(max(0.0, z_min - 0.2), z_max + 0.2)
    
    muv_min, muv_max = df_muv['muv'].min(), df_muv['muv'].max()
    y_buffer = (muv_max - muv_min) * 0.1 
    plt.ylim(muv_max + y_buffer, muv_min - y_buffer) 

    plt.tight_layout()
    
    plot_path = output_dir / "muv_vs_z_plot_1.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nPlot saved successfully to: {plot_path.resolve()}")
    
    # Also save the MUV results to a CSV
    csv_path_out = output_dir / "muv_z_results.csv"
    df_muv.to_csv(csv_path_out, index=False)
    print(f"MUV results saved to: {csv_path_out.resolve()}")


# ----------------------------------------------------------------------
## CLI EXECUTION
# ----------------------------------------------------------------------

if __name__ == "__main__":
    process_and_plot_muv_vs_z(
        SPECTRA_BASE_DIR, 
        CSV_PATH_GLOBAL, 
        OUTPUT_DIR
    )