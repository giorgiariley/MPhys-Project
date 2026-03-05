import astropy.units as u
from typing import Optional, Union, List
import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'

from galfind.Data import morgan_version_to_dir
# Import necessary classes
from galfind import Catalogue, EAZY, SED_code, Redshift_Extractor
# ADDED Catalogue_Cutouts and ID_Selector
from galfind import Catalogue_Cutouts, ID_Selector 
from galfind import galfind_logger
from galfind import Bagpipes 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # Needed for isin

# === Parameters ===
survey = "JADES-DR3-GS-South"
version = "v13"
instrument_names = ["ACS_WFC", "NIRCam"]
aper_diams = [0.32] * u.arcsec
forced_phot_band = ["F277W", "F356W", "F444W"]
min_flux_pc_err = 10.



# NEW: Path to your CSV file containing the IDs to keep
filter_id_csv_path = "/nvme/scratch/work/Griley/Masters/exposure_photometry_matches_filtered_South.csv"
# filter_id_csv_path = '/nvme/scratch/work/Griley/Masters/subsample_photometric_ids.csv' #NEED TO PUT HERE the path to the new one
filter_id_column = 'photometry_NUMBER' # The column name in your CSV with the IDs

# === EAZY fits ===
SED_fitter_arr = [EAZY({"templates": "fsps_larson", "lowz_zmax": None})]

sample_SED_fitter_arr = [
        # Bagpipes(...) # Keep commented out as before
    ]

# === Load the FULL Catalogue first ===
print(f"Loading full catalogue for {survey} {version}...")
cat = Catalogue.pipeline(
    survey,
    version,
    instrument_names=instrument_names,
    version_to_dir_dict=morgan_version_to_dir,
    aper_diams=aper_diams,
    forced_phot_band=forced_phot_band,
    min_flux_pc_err=min_flux_pc_err,
    crops=None # Load all objects initially
)
print(f"Full catalogue loaded with {len(cat.ID)} objects.")

# === Load the list of IDs to keep ===
print(f"Loading filter IDs from: {filter_id_csv_path}")
try:
    df_filter = pd.read_csv(filter_id_csv_path)
    if filter_id_column not in df_filter.columns:
        raise KeyError(f"Column '{filter_id_column}' not found in filter CSV.")
    
    # Get unique IDs from the specified column
    selected_ids_from_csv = df_filter[filter_id_column].unique()
    print(f"Loaded {len(selected_ids_from_csv)} unique IDs to select.")
    
except FileNotFoundError:
    print(f"ERROR: Filter CSV file not found at {filter_id_csv_path}. Cannot apply ID filter.")
    exit()
except KeyError as e:
    print(f"ERROR: {e}")
    exit()

# === Apply the ID Selector ===
# IMPORTANT: Check the actual ID column name in your 'cat' object. 
# It's often 'ID' or 'NUMBER'. We'll assume 'ID' as it's used by galfind.
catalogue_id_column_name = 'ID' 

if not hasattr(cat, catalogue_id_column_name):
     print(f"ERROR: Catalogue object does not have an attribute named '{catalogue_id_column_name}'. Cannot filter by ID.")
     # You might need to inspect `cat.data.colnames` or `vars(cat)` to find the correct ID column.
     exit()

# Get the list of all IDs from the main catalogue
cat_IDs = np.array(getattr(cat, catalogue_id_column_name))

# Create the ID_Selector using the IDs from your CSV
id_selector = ID_Selector(selected_ids_from_csv, f"selected_from_{os.path.basename(filter_id_csv_path)}")

# Apply the selector to the full catalogue to get the filtered subset
print("Applying ID filter to catalogue...")
cat_selected = id_selector(cat)
print(f"Filtered catalogue contains {len(cat_selected.ID)} objects.")

if len(cat_selected.ID) == 0:
    print("Warning: No objects remaining after applying ID filter. No plots will be generated.")
    exit()

# === Run SED fitters ONLY on the selected catalogue ===
print("Running SED fitters on selected catalogue...")
for SED_fitter in SED_fitter_arr:
        for aper_diam in aper_diams:
            # Use the filtered 'cat_selected' object
            SED_fitter(cat_selected, aper_diam, load_PDFs = True, load_SEDs = True, update = True)

# sample_SED_fitter_arr is empty, so this loop does nothing, keep as is
for SED_fitter in sample_SED_fitter_arr:
        for aper_diam in aper_diams:
            # Use the filtered 'cat_selected' object
            SED_fitter(cat_selected, aper_diam, load_PDFs = False, load_SEDs = True, update = True, temp_label = 'temp')


# === Plot phot diagnostics ONLY for selected galaxies ===
# cat.plot_phot_diagnostics( ... )
# (This section is still commented out)

# === Generate and Save Individual Cutouts ===
print("Generating/loading FITS cutouts for selected objects...")

# 1. Define the filter and size for your cutouts
cutout_filter = "F444W" 
cutout_size_arcsec = 0.96 * u.arcsec 

# 2. This loads all the FITS cutout data into a list-like object
cutouts_to_plot = Catalogue_Cutouts.from_cat_filt(
    cat_selected, 
    cutout_filter, 
    cutout_size_arcsec, 
    overwrite=False # Set to True if you want to regenerate them
)
print(f"Loaded {len(cutouts_to_plot)} cutout data objects.")


# === MODIFIED SECTION: Loop and save individual plots ===
# 3. Create a directory for the individual cutouts
individual_cutout_dir = f"{survey}_{version}_individual_cutouts_{cutout_filter}_sem2"
os.makedirs(individual_cutout_dir, exist_ok=True)
print(f"Saving {len(cutouts_to_plot)} individual PNG plots to: {individual_cutout_dir}")

# 4. Loop through each cutout, plot it, and save it
for cutout in cutouts_to_plot:
    try:
        # Get the galaxy ID from the 'meta' dictionary for the filename
        # This is the safest way to get the ID for an individual cutout
        galaxy_id = cutout.meta['ID']
        
        # Create a new figure and axis for each plot
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Call the plot method of the *individual* Cutout object
        # --- THIS IS THE CHANGED LINE ---
        # Pass cmap='gray_r' to plot in reversed grayscale (black sky, white object)
        cutout.plot(ax=ax, imshow_kwargs={'cmap': 'gray_r'})
        
        # Set a title for the individual plot
        ax.set_title(f"{survey} ID {galaxy_id} - {cutout_filter}")
        
        filename = f"ID_{galaxy_id}_{cutout_filter}_cutout.png"
        output_path = os.path.join(individual_cutout_dir, filename)
        
        fig.savefig(output_path, dpi=150)
        plt.close(fig) # Close the figure to save memory

    except Exception as e:
        # Try to get ID for error message, default to 'unknown'
        try:
            gid = cutout.meta['ID']
        except:
            gid = 'unknown'
        print(f"Warning: Failed to plot cutout for ID {gid}: {e}")

print(f"Finished saving individual cutouts.")


# === OLD PLOTTER (Commented out) ===
# print("Generating a grid of image cutouts for selected objects...")
#
# # 3. This plots the grid and saves it to a file
# print(f"Plotting cutouts for {len(cutouts_to_plot)} objects...")
# output_plot_path = f"{survey}_{version}_selected_cutouts.png"
#
# # The .plot() method returns a Figure object. 
# # It does not accept 'save_path' as an argument.
# fig = cutouts_to_plot.plot(
#     show=False  # 'show=False' tells matplotlib not to open a window
# )
#
# # Save the returned figure object to the path
# fig.savefig(output_plot_path, dpi=200)
# plt.close(fig) # Close the figure to free up memory
#
# print(f"Cutout grid plot saved to: {output_plot_path}")

print("\nScript finished.")

