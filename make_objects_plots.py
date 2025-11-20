import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os

# ---------------------- CONFIG ----------------------
C_LIGHT = 2.99792458e18  # Å/s

# --- Paths ---
# Main directory containing all FITS files in subfolders
fits_dir = "/raid/scratch/work/Griley/GALFIND_WORK/Spectra/2D/"
# Master CSV with file info (e.g., redshift) for *all* exposures
master_csv_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/mphys_GOODS_S_exposures.csv"
# NEW: CSV file that lists *only* the specific prism files we want to plot
target_list_csv = "/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/specFitMSA/data/project_mphys_ultrablue/matched_exposures_prism.csv"
# Output directory for plots
out_dir = "/nvme/scratch/work/Griley/Masters/prettyplots"
os.makedirs(out_dir, exist_ok=True)


# ---------------------- FUNCTIONS ----------------------

def build_file_map(base_dir):
    """
    Recursively scans a directory and returns a dictionary mapping
    filenames to their full paths.
    """
    print(f"Scanning {base_dir} for all .spec.fits files...")
    file_map = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".spec.fits"):
                if f in file_map:
                    print(f"Warning: Duplicate filename found: {f}. Using first instance.")
                else:
                    file_map[f] = os.path.join(root, f)
    print(f"Found {len(file_map)} total FITS files.")
    return file_map


def read_spectrum(fits_path):
    """
    Extracts 1D spectrum arrays (wavelength, flux, error) and ensures Wavelength is in microns.
    """
    with fits.open(fits_path) as hdul:
        try:
            spec1d = hdul['SPEC1D'].data
        except KeyError:
            spec1d = hdul[1].data 
        wave, flux, err = spec1d['wave'], spec1d['flux'], spec1d['err']
        # Ensure wavelength is in microns (assuming > 100 means Angstroms)
        if np.nanmax(wave) > 100.0:
            wave = wave / 1e4 
    return wave, flux, err

def read_2d_spectrum(fits_path):
    """Extracts the 2D spectral image."""
    with fits.open(fits_path) as hdul:
        if 'SCI' in hdul:
            data_2d = hdul['SCI'].data
        elif 'SPEC2D' in hdul:
            data_2d = hdul['SPEC2D'].data
        else:
            data_2d = hdul[0].data if len(hdul) > 0 else None
    if data_2d is None:
        raise KeyError("Could not find 2D spectral data (expected 'SCI' or 'SPEC2D').")
    return data_2d

def clean_data(wave, flux, err, data_2d):
    """
    Creates a mask for non-finite values and applies it to all arrays.
    """
    mask = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)
    wave_clean = wave[mask]
    flux_clean = flux[mask]
    err_clean = err[mask]
    # Apply the same mask to the spectral axis (columns) of the 2D image
    # Ensure 2D data has enough columns to be masked
    if data_2d.shape[1] == len(mask):
        data_2d_clean = data_2d[:, mask]
    else:
        # This handles cases where 1D and 2D arrays might have different lengths
        print(f"Warning: 2D data shape ({data_2d.shape[1]}) does not match wave array ({len(mask)}). Using uncleaned 2D data.")
        # We must still return a 2D array, so we return the original
        # And we must return a wave array that matches, so we use the uncleaned wave
        wave_clean = wave
        data_2d_clean = data_2d
        # Re-mask flux and error to match the uncleaned wave
        flux_clean = flux[np.isfinite(wave)]
        err_clean = err[np.isfinite(wave)]
        wave_clean = wave[np.isfinite(wave)]
        
    return wave_clean, flux_clean, err_clean, data_2d_clean

def plot_2d_spectrum(ax, data_2d, wave_x_axis):
    """
    Plots the 2D spectrum using pcolormesh for precise grid alignment.
    """
    vmin, vmax = np.percentile(data_2d[np.isfinite(data_2d)], [10, 99])
    
    n_spatial = data_2d.shape[0]
    y_corners = np.arange(n_spatial + 1) - n_spatial / 2.0 # Center Y-axis

    # Create X-axis corners
    wave_midpoints = (wave_x_axis[:-1] + wave_x_axis[1:]) / 2.0
    dw_start = wave_x_axis[1] - wave_x_axis[0]
    dw_end = wave_x_axis[-1] - wave_x_axis[-2]
    x_corners = np.concatenate([
        [wave_x_axis[0] - dw_start / 2.0], 
        wave_midpoints, 
        [wave_x_axis[-1] + dw_end / 2.0]
    ])

    ax.pcolormesh(x_corners, y_corners, data_2d, cmap='magma', vmin=vmin, vmax=vmax, shading='auto')
    ax.set_ylabel("Spatial Pixel")
    ax.set_title(f"2D Spectrum (Observed Wavelength)", pad=15)

def plot_1d_spectrum(ax, wave, flux, err):
    """Plots the 1D spectrum with percentile-based y-axis zoom."""
    ax.plot(wave, flux, color='dodgerblue', lw=0.7, label='Extracted 1D Spectrum')
    ax.fill_between(wave, flux - err, flux + err, color='lightblue', alpha=0.4)
    
    combined_data = np.concatenate([flux - err, flux + err])
    finite_data = combined_data[np.isfinite(combined_data)]
    if len(finite_data) > 10:
        y_min, y_max = np.percentile(finite_data, [0.9, 99.5])
        y_range = y_max - y_min
        y_margin = 0.15 * y_range
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    ax.set_xlabel(r"Observed Wavelength [$\mu$m]")
    ax.set_ylabel(r"$f_{\nu}$ [$\mu$Jy]") 
    ax.grid(alpha=0.3)
    # Legend removed for cleaner look

# ---------------------- MAIN ----------------------

# --- Step 1: Build a map of all FITS files on disk ---
file_map = build_file_map(fits_dir)

# --- Step 2: Load the master CSV (for redshift) ---
try:
    df_master = pd.read_csv(master_csv_path)
except FileNotFoundError:
    print(f"ERROR: Master CSV not found at {master_csv_path}")
    exit()

# --- Step 3: Load the target list CSV (files to plot) ---
try:
    df_targets = pd.read_csv(target_list_csv)
    target_filenames = df_targets['file'].tolist()
    print(f"\nLoaded {len(target_filenames)} targets to plot from {target_list_csv}")
except FileNotFoundError:
    print(f"ERROR: Target list CSV not found at {target_list_csv}")
    exit()
except KeyError:
    print(f"ERROR: 'file' column not found in {target_list_csv}")
    exit()

# --- Step 4: Merge target list with master list to get redshift ---
# This gives us a DataFrame with 'file' and 'z' for only the targets we care about
df_to_plot = df_master[df_master['file'].isin(target_filenames)]
print(f"Found {len(df_to_plot)} matching targets in master CSV to get redshift.")

# --- Step 5: Loop over each target and create plot ---
print("\n--- Starting Batch Plotting ---")
processed_count = 0
for index, row in df_to_plot.iterrows():
    target_name = row['file']
    z = row['z']
    
    if not pd.notna(z):
        print(f"Skipping {target_name}: No redshift found.")
        continue
        
    # Find the full path to this file
    if target_name not in file_map:
        print(f"Skipping {target_name}: File not found in {fits_dir}")
        continue
    
    fits_path = file_map[target_name]

    try:
        # Load both 1D + 2D spectra
        wave_obs, flux_obs, err_obs = read_spectrum(fits_path)
        data_2d_original = read_2d_spectrum(fits_path)

        # Clean all arrays using a single mask
        wave_obs_c, flux_obs_c, err_obs_c, data_2d_c = clean_data(wave_obs, flux_obs, err_obs, data_2d_original)

        # --- Create the Plot ---
        fig, ax = plt.subplots(
            2, 1, figsize=(12, 7),
            gridspec_kw={'height_ratios': [1, 2]}, sharex=True
        )

        # Plot 2D with the cleaned wavelength grid
        plot_2d_spectrum(ax[0], data_2d_c, wave_obs_c)
        ax[0].text(0.98, 0.92, f"z = {z:.3f}", transform=ax[0].transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax[0].set_title(f"2D Spectrum: {target_name}", pad=15) # Add filename to title

        # Plot 1D with the cleaned wavelength grid
        plot_1d_spectrum(ax[1], wave_obs_c, flux_obs_c, err_obs_c) 

        # Finalize plot
        ax[0].set_xlim(wave_obs_c[0], wave_obs_c[-1])
        plt.setp(ax[0].get_xticklabels(), visible=False)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05) # Reduce space between plots

        out_path = os.path.join(out_dir, target_name.replace('.spec.fits', '_2D_1D_plot.png'))
        plt.savefig(out_path, dpi=300)
        plt.close(fig) # Close figure to save memory
        
        print(f"Successfully saved plot for: {target_name}")
        processed_count += 1

    except Exception as e:
        print(f"!!! FAILED to plot {target_name}: {e}")
        # Close any partially open plot
        plt.close('all')

print(f"\n--- Batch Plotting Complete ---")
print(f"Successfully processed and saved {processed_count} / {len(df_to_plot)} plots.")
