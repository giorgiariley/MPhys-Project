import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os
import astropy.units as u
from typing import Optional, Union, List

# --- GALFIND CONFIG ---
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'
from galfind.Data import morgan_version_to_dir
from galfind import Catalogue, EAZY, SED_code, Redshift_Extractor
from galfind import Catalogue_Cutouts, ID_Selector 
from galfind import galfind_logger
from galfind import Bagpipes 

# ---------------------- CONFIG ----------------------
C_LIGHT = 2.99792458e18  # Å/s

# --- Paths ---
fits_dir = "/raid/scratch/work/Griley/GALFIND_WORK/Spectra/2D/"
master_csv_path = "/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/mphys_GOODS_S_exposures.csv"
# This CSV has the list of target *spectrum* files
target_list_csv = "/nvme/scratch/work/Griley/Masters/exposure_photometry_matches_filtered_East.csv"
# This CSV maps the *spectrum* files to the *photometry* IDs
filter_id_csv_path = "/nvme/scratch/work/Griley/Masters/exposure_photometry_matches_filtered_East.csv"
filter_id_column = 'photometry_NUMBER'
filter_file_column = 'file'

out_dir = "/nvme/scratch/work/Griley/Masters/prettyplots_combined"
os.makedirs(out_dir, exist_ok=True)

# --- Galfind Parameters ---
survey = "JADES-DR3-GS-East" 
version = "v13"
instrument_names = ["ACS_WFC", "NIRCam"]
aper_diams = [0.32] * u.arcsec
forced_phot_band = ["F277W", "F356W", "F444W"]
min_flux_pc_err = 10.
cutout_filter = "F444W" 
cutout_size_arcsec = 0.96 * u.arcsec 

# --- EAZY fits (needed for Catalogue) ---
SED_fitter_arr = [EAZY({"templates": "fsps_larson", "lowz_zmax": None})]

# ---------------------- FUNCTIONS ----------------------

def build_file_map(base_dir):
    """Scans for all .spec.fits files and maps filename to full path."""
    print(f"Scanning {base_dir} for all .spec.fits files...")
    file_map = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".spec.fits"):
                file_map.setdefault(f, os.path.join(root, f))
    print(f"Found {len(file_map)} total unique FITS files.")
    return file_map

def read_spectrum(fits_path):
    """Reads 1D spectrum, returns wave (μm), flux (μJy), err (μJy)."""
    with fits.open(fits_path) as hdul:
        try:
            spec1d = hdul['SPEC1D'].data
        except KeyError:
            spec1d = hdul[1].data 
        wave, flux, err = spec1d['wave'], spec1d['flux'], spec1d['err']
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
    """Applies a single finite-value mask to all 1D and 2D arrays."""
    mask = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)
    
    # Ensure 2D data has the same spectral dimension
    if data_2d.shape[1] != len(mask):
        print(f"Warning: 2D shape ({data_2d.shape[1]}) != wave array ({len(mask)}). Truncating mask.")
        min_len = min(data_2d.shape[1], len(mask))
        mask = mask[:min_len]
        data_2d_clean = data_2d[:, mask]
        wave_clean = wave[:min_len][mask]
        flux_clean = flux[:min_len][mask]
        err_clean = err[:min_len][mask]
    else:
        data_2d_clean = data_2d[:, mask]
        wave_clean = wave[mask]
        flux_clean = flux[mask]
        err_clean = err[mask]
        
    return wave_clean, flux_clean, err_clean, data_2d_clean

def plot_2d_spectrum(ax, data_2d, wave_x_axis):
    """Plots the 2D spectrum using pcolormesh for precise grid alignment."""
    vmin, vmax = np.percentile(data_2d[np.isfinite(data_2d)], [10, 99])
    n_spatial = data_2d.shape[0]
    y_corners = np.arange(n_spatial + 1) - n_spatial / 2.0 # Center Y-axis

    wave_midpoints = (wave_x_axis[:-1] + wave_x_axis[1:]) / 2.0
    dw_start = (wave_x_axis[1] - wave_x_axis[0]) / 2.0
    dw_end = (wave_x_axis[-1] - wave_x_axis[-2]) / 2.0
    x_corners = np.concatenate([[wave_x_axis[0] - dw_start], wave_midpoints, [wave_x_axis[-1] + dw_end]])
    
    # Fallback for non-monotonic arrays
    if not np.all(np.diff(x_corners) > 0):
        x_corners = np.linspace(wave_x_axis[0], wave_x_axis[-1], len(wave_x_axis) + 1)

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
    target_filenames = set(df_targets['file']) 
    print(f"\nLoaded {len(target_filenames)} unique targets to plot from {target_list_csv}")
except FileNotFoundError:
    print(f"ERROR: Target list CSV not found at {target_list_csv}")
    exit()
except KeyError:
    print(f"ERROR: 'file' column not found in {target_list_csv}")
    exit()

# --- Step 4: Merge target list with master list to get redshift ---
df_to_plot = df_master[df_master['file'].isin(target_filenames)]
print(f"Found {len(df_to_plot)} matching targets in master CSV to get redshift.")

# --- Step 5: Load the filename-to-ID map ---
try:
    df_id_map = pd.read_csv(filter_id_csv_path)
    # Create the dictionary: keys=filename, values=photometry_NUMBER
    id_map = pd.Series(df_id_map[filter_id_column].values, index=df_id_map[filter_file_column]).to_dict()
    # Get a set of all unique IDs we will need
    all_target_ids = set(id_map.values())
    print(f"Loaded {len(id_map)} file-to-ID mappings from {filter_id_csv_path}")
    print(f"This corresponds to {len(all_target_ids)} unique photometry IDs.")
except FileNotFoundError:
    print(f"ERROR: Filter ID CSV file not found at {filter_id_csv_path}")
    exit()
except KeyError:
    print(f"ERROR: CSV must contain '{filter_file_column}' and '{filter_id_column}' columns.")
    exit()

# --- Step 6: Load and filter the FULL Galfind Catalogue (once) ---
print(f"\nLoading full galfind catalogue for {survey} {version}...")
try:
    cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names=instrument_names,
        version_to_dir_dict=morgan_version_to_dir,
        aper_diams=aper_diams,
        forced_phot_band=forced_phot_band,
        min_flux_pc_err=min_flux_pc_err,
        crops=None # Load all objects
    )
    print(f"Full catalogue loaded with {len(cat.ID)} objects.")
    
    # --- Filter cat ONCE to only the objects we need ---
    # Use the 'ID' attribute of the catalogue for matching
    catalogue_id_column_name = 'ID' 
    if not hasattr(cat, catalogue_id_column_name):
        print(f"ERROR: Catalogue object does not have attribute '{catalogue_id_column_name}'. Cannot filter.")
        exit()
        
    id_selector = ID_Selector(list(all_target_ids), "full_target_list_selector")
    cat_selected = id_selector(cat)
    print(f"Filtered catalogue to {len(cat_selected.ID)} objects of interest.")
    
    if len(cat_selected.ID) == 0:
        print("Warning: No objects from your list were found in the main galfind catalogue. Exiting.")
        exit()
        
    # --- Run SED fitters ONCE on the filtered catalogue ---
    print("Running SED fitters on filtered catalogue...")
    for SED_fitter in SED_fitter_arr:
        for aper_diam in aper_diams:
            SED_fitter(cat_selected, aper_diam, load_PDFs = True, load_SEDs = True, update = True)
    print("SED fitting complete.")

except Exception as e:
    print(f"CRITICAL ERROR loading galfind catalogue or running SED fitter: {e}")
    print("This may be due to the missing 'libsatlas.so.3' library or symlinks.")
    exit()

# --- Step 7: Load ALL cutouts for the selected galaxies (once) ---
print(f"\nLoading all {cutout_filter} cutouts for {len(cat_selected.ID)} objects...")
try:
    cutouts_collection = Catalogue_Cutouts.from_cat_filt(
        cat_selected, 
        cutout_filter, 
        cutout_size_arcsec, 
        overwrite=False # Don't regenerate FITS if it exists
    )
    
    # --- *** THIS IS THE FIX *** ---
    # Create a dictionary for fast cutout lookup
    # We map the galaxy ID (e.g., 55994) to its corresponding Cutout object
    
    cutout_dict = {}
    
    # Get the list of IDs from the *catalogue* object, which is in the correct order
    ids_for_cutouts = cat_selected.ID 
    
    # Check for length mismatch, which would be a critical bug
    if len(ids_for_cutouts) != len(cutouts_collection):
        print(f"CRITICAL ERROR: Mismatch in length between catalogue ({len(ids_for_cutouts)}) and cutouts ({len(cutouts_collection)}).")
        exit()

    # Create the dictionary by zipping the IDs and the cutouts together
    for gal_id, cutout_obj in zip(ids_for_cutouts, cutouts_collection):
        cutout_dict[gal_id] = cutout_obj
        
    print(f"Created cutout lookup dictionary with {len(cutout_dict)} entries.")
    # --- *** END OF FIX *** ---

except Exception as e:
    print(f"CRITICAL ERROR loading cutouts: {e}")
    print("This may be due to the missing 'libsatlas.so.3' library or symlinks.")
    exit()


# --- Step 8: Loop over each target spectrum and plot ---
print("\n--- Starting Batch Plotting ---")
processed_count = 0
for index, row in df_to_plot.iterrows():
    target_name = row['file'] # The spectrum filename (e.g., ...prism-clear...)
    z = row['z']
    
    if not pd.notna(z):
        print(f"Skipping {target_name}: No redshift found.")
        continue
        
    if target_name not in file_map:
        print(f"Skipping {target_name}: File not found in {fits_dir}")
        continue
    
    fits_path = file_map[target_name]
    
    # --- A: Generate the 2D/1D Spectrum Plot ---
    try:
        wave_obs, flux_obs, err_obs = read_spectrum(fits_path)
        data_2d_original = read_2d_spectrum(fits_path)
        wave_obs_c, flux_obs_c, err_obs_c, data_2d_c = clean_data(wave_obs, flux_obs, err_obs, data_2d_original)

        # Create the figure layout
        fig = plt.figure(figsize=(16, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[3, 1])
        ax_2d = fig.add_subplot(gs[0, 0])
        ax_1d = fig.add_subplot(gs[1, 0], sharex=ax_2d)
        ax_cut = fig.add_subplot(gs[:, 1]) 

        # Plot 2D Spectrum (Top-Left)
        plot_2d_spectrum(ax_2d, data_2d_c, wave_obs_c)
        ax_2d.text(0.98, 0.92, f"z = {z:.3f}", transform=ax_2d.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax_2d.set_title(f"2D Spectrum: {target_name}", pad=15)

        # Plot 1D Spectrum (Bottom-Left)
        plot_1d_spectrum(ax_1d, wave_obs_c, flux_obs_c, err_obs_c) 
        ax_2d.set_xlim(wave_obs_c[0], wave_obs_c[-1])
        plt.setp(ax_2d.get_xticklabels(), visible=False)

        # --- B: Get and Plot the Individual Cutout (Right) ---
        galaxy_id = id_map.get(target_name) # Get the photometry_NUMBER for this file
        
        if not galaxy_id or galaxy_id not in cutout_dict:
            print(f"  - Warning: No cutout found for {target_name} (ID: {galaxy_id}).")
            ax_cut.text(0.5, 0.5, f"ID {galaxy_id} not in cutout list", ha='center', va='center', fontsize=12, color='red')
            ax_cut.set_xticks([])
            ax_cut.set_yticks([])
        else:
            try:
                # Find the specific cutout from the dictionary we made earlier
                cutout_to_plot = cutout_dict[galaxy_id]
                
                # Plot *that* cutout
                cutout_to_plot.plot(ax=ax_cut, imshow_kwargs={'cmap': 'gray_r'}) 
                ax_cut.set_title(f"{cutout_filter} (ID: {galaxy_id})")
                ax_cut.set_xticks([])
                ax_cut.set_yticks([])
                print(f"  - Successfully added cutout for ID {galaxy_id}")
            
            except Exception as e:
                print(f"!!! FAILED to create cutout for ID {galaxy_id}: {e}")
                ax_cut.text(0.5, 0.5, "Cutout plot failed", ha='center', va='center', fontsize=12, color='red')
        
        # --- Save the combined plot ---
        out_path = os.path.join(out_dir, target_name.replace('.spec.fits', '_full_diagnostic.png'))
        plt.savefig(out_path, dpi=200)
        plt.close(fig) 
        
        print(f"Successfully saved combined plot for: {target_name}")
        processed_count += 1

    except Exception as e:
        print(f"!!! FAILED to create combined plot for {target_name}: {e}")
        plt.close('all')
        continue # Skip to next galaxy if spectrum plot fails


print(f"\n--- Batch Plotting Complete ---")
print(f"Successfully processed {processed_count} / {len(df_to_plot)} combined plots.")

