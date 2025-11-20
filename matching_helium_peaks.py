import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os
import re

# ---------------------- CONFIG ----------------------
C_LIGHT = 2.99792458e18  # Å/s
fits_dir = "/raid/scratch/work/Griley/GALFIND_WORK/Spectra/2D/"
csv_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/mphys_GOODS_S_exposures.csv"
out_dir = "/nvme/scratch/work/Griley/Masters/matching_plots"
os.makedirs(out_dir, exist_ok=True)


# ---------------------- FUNCTIONS ----------------------

def read_spectrum(fits_path):
    """Read 1D spectrum from a FITS file."""
    try:
        with fits.open(fits_path) as hdul:
            spec = hdul['SPEC1D'].data
            wave = spec['wave']   # observed wavelength [μm]
            flux = spec['flux']   # flux [μJy]
            err  = spec['err']    # error [μJy]
        return wave, flux, err
    except (KeyError, FileNotFoundError):
        print(f"Warning: Could not read SPEC1D from {fits_path}")
        return None, None, None


def find_file_recursively(base_dir, filename):
    """Find a specific file recursively."""
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def find_all_matching_files(base_dir, prefix, suffix):
    """
    Find all files with the same prefix and suffix (different gratings).
    Returns a sorted list of full paths.
    """
    matches = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.startswith(prefix) and f.endswith(suffix):
                matches.append(os.path.join(root, f))
    # Sort so 'prism' appears first, then 'medium', then others alphabetically
    matches.sort(key=lambda x: ('prism' not in x, x))
    return matches


def convert_to_rest_frame(wave_obs, flux_obs, err_obs, z):
    """Convert to rest-frame λ in Å and Fλ in erg/s/cm²/Å."""
    flux_nu = flux_obs * 1e-29  # μJy → erg/s/cm²/Hz
    err_nu  = err_obs  * 1e-29
    wave_A  = wave_obs * 1e4    # μm → Å
    flux_lambda = flux_nu * C_LIGHT / wave_A**2
    err_lambda  = err_nu  * C_LIGHT / wave_A**2
    wave_rest = wave_A / (1 + z)
    flux_rest = flux_lambda * (1 + z)**2
    err_rest  = err_lambda * (1 + z)**2
    return wave_rest, flux_rest, err_rest


def plot_single_spectrum(ax, wave, flux, err, title, z, color='dodgerblue', fill_color='lightblue'):
    """Plot a single rest-frame spectrum."""
    if wave.size == 0:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return

    ax.plot(wave, flux, color=color, lw=1.0, label='Rest-frame spectrum')
    ax.fill_between(wave, flux - err, flux + err, color=fill_color, alpha=0.4, label='1σ uncertainty')
    ax.set_xlabel(r"Wavelength $\lambda_{\rm rest}$ [Å]", fontsize=12)
    ax.set_ylabel(r"Flux density $F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=12)
    ax.set_xlim(0,3000)
    ax.set_title(title + f"\n(z = {z:.3f})", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)


# ---------------------- MAIN LOOP ----------------------

df = pd.read_csv(csv_path)

# --- NEW: Pre-filter the main DataFrame to only include high-SNR prism files ---
# This makes the main loop much faster and more targeted.
snr_csv_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/uv_snr_5plus.csv"
print(f"Loading high-SNR prism filter file from: {snr_csv_path}")
try:
    df_snr = pd.read_csv(snr_csv_path)
    high_snr_prism_files = set(df_snr['file'])
    print(f"Found {len(high_snr_prism_files)} high-SNR prism files to process.")

    # Filter the main dataframe to only loop over these specific prism files
    df = df[df['file'].isin(high_snr_prism_files)]
except FileNotFoundError:
    print(f"Warning: High-SNR file not found at {snr_csv_path}. Processing all files from main CSV.")


for idx, row in df.iterrows():
    file_base = row['file']
    z = float(row['z'])

    # The file_base from our filtered df is now guaranteed to be a high-SNR prism file.
    # We now need to find its corresponding medium/high-res counterparts.

    # Extract object identifier parts from the prism filename
    match = re.search(r'(v4_).+?(_\d+_\d+\.spec\.fits$)', file_base)
    if not match:
        print(f"Skipping {file_base}: could not parse a standard object ID.")
        continue
    
    prefix = file_base.split(match.group(1))[0] + match.group(1)
    suffix = match.group(2)

    # --- Find all spectra that correspond to this object ID ---
    matched_files = find_all_matching_files(fits_dir, prefix, suffix)
    
    # --- As requested: Check if only the prism file was found ---
    # If only one file is found, it must be the prism file we started with,
    # so there are no other spectra to compare it to.
    if len(matched_files) <= 1:
        print(f"Skipping {file_base}: Found prism spectrum, but no other grating data to compare.")
        continue
    # --- End of check ---

    # --- Convert each to rest frame and store ---
    spectra = []
    for path in matched_files:
        if not os.path.exists(path):
            print(f"Warning: File path not found: {path}")
            continue
        wave, flux, err = read_spectrum(path)
        if wave is None:
            continue
        wave_rest, flux_rest, err_rest = convert_to_rest_frame(wave, flux, err, z)
        spectra.append((wave_rest, flux_rest, err_rest, os.path.basename(path)))

    # --- Plot all spectra side-by-side ---
    ncols = len(spectra)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), sharey=True)

    if ncols == 1:
        axes = [axes]  # Make sure axes is always a list

    # Define a color mapping for different gratings
    colors = {'prism': ('black', 'lightgray'), 'g235h': ('dodgerblue', 'lightblue'), 'g395h': ('seagreen', 'lightgreen')}
    default_color = ('purple', 'plum')

    for ax, (wave, flux, err, name) in zip(axes, spectra):
        # Determine color based on grating name in the filename
        grating_key = 'prism' if 'prism' in name else ('g235h' if 'g235h' in name else ('g395h' if 'g395h' in name else 'other'))
        line_color, fill_color = colors.get(grating_key, default_color)
        plot_single_spectrum(ax, wave, flux, err, name, z, color=line_color, fill_color=fill_color)

    plt.tight_layout()

    # --- Save figure ---
    base_name = os.path.splitext(os.path.basename(file_base))[0]
    out_file = f"{base_name}_comparison_z{z:.3f}.png"
    out_path = os.path.join(out_dir, out_file)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot with {ncols} spectra: {out_path}")


    breakpoint() # Removed for autorun

