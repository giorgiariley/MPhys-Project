import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import numpy as np

# ---------------------- CONFIGURATION ----------------------

import matplotlib as mpl

# --- Global Matplotlib style for publication-quality plots ---
mpl.rcParams.update({
    "font.size": 20,             # Base font size
    "axes.titlesize": 20,        # Title size
    "axes.labelsize": 20,        # Axis label size
    "xtick.labelsize": 16,       # X tick label size
    "ytick.labelsize": 16,       # Y tick label size
    "legend.fontsize": 18,       # Legend text
    "figure.titlesize": 20,      # Overall figure title
    "axes.linewidth": 1.4,       # Thicker axes
    "xtick.major.width": 1.2,    # Tick line width
    "ytick.major.width": 1.2,
    "xtick.major.size": 6,       # Tick size
    "ytick.major.size": 6,
    "lines.linewidth": 1.6,      # Slightly thicker default lines
    "savefig.dpi": 300,          # High-resolution output for publication
})

# Input Files
spec_csv_path = "/nvme/scratch/work/Griley/Masters/mphys_GOODS_S_exposures.csv"

# UPDATED: List of photometric FITS paths
phot_fits_paths = [
    "/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-South/(0.32)as/JADES-DR3-GS-South_MASTER_Sel-F277W+F356W+F444W_v13.fits",
    "/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits"
]

# NEW: Filtering for GOODS-North photometry
# Based on your "up to ra 53.175 and dec -27.78" and the plot, we define a box
# to capture the outlier cluster in the top-left of the plot (high RA, high Dec).
# (Note: RA axis is inverted, so "up to 53.175" means RA > 53.175 is to the left)
gn_ra_min = 53.175 # Filter to keep RA values GREATER than this
gn_dec_min = -27.78 # Filter to keep Dec values GREATER than this

# List of CSV files containing the filenames to keep
filter_csv_files = ["/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/specFitMSA/data/project_mphys_ultrablue/matched_exposures_prism.csv"]

# Output Plot
output_plot_path = "/nvme/scratch/work/Griley/Masters/footprint_comparison_filtered_GE+GS.png"

# Column names from your prompt
spec_ra_col = 'ra'
spec_dec_col = 'dec'
spec_file_col = 'file' # Column to use for filtering

phot_ra_col = 'ALPHA_J2000'
phot_dec_col = 'DELTA_J2000'
photometry_hdu_index = 1

# ---------------------- FUNCTIONS ----------------------

def load_valid_filenames(file_list):
    """
    Reads a list of CSV files and returns a set of unique filenames
    from the 'file' column of each file.
    """
    valid_files = set()
    print("Loading filter filenames from:")
    for f_path in file_list:
        try:
            df_filter = pd.read_csv(f_path)
            if 'file' in df_filter.columns:
                valid_files.update(df_filter['file'].astype(str).tolist())
                print(f"  - Loaded {len(df_filter['file'].unique())} unique filenames from {os.path.basename(f_path)}")
            else:
                print(f"  - Warning: 'file' column not found in {os.path.basename(f_path)}")
        except FileNotFoundError:
            print(f"  - Warning: Filter file not found: {f_path}")
        except Exception as e:
            print(f"  - Warning: Error reading {f_path}: {e}")
    print(f"Total unique filenames loaded for filtering: {len(valid_files)}")
    return valid_files

# ---------------------- SCRIPT START ----------------------

# --- Load the set of filenames to keep ---
valid_filenames = load_valid_filenames(filter_csv_files)
if not valid_filenames:
    print("ERROR: No valid filenames loaded from filter CSVs. Cannot proceed.")
    exit()

print(f"\nLoading spectroscopic catalog from: {spec_csv_path}")
try:
    df_spec = pd.read_csv(spec_csv_path)
    # Check for all required columns
    required_spec_cols = [spec_ra_col, spec_dec_col, spec_file_col]
    if not all(col in df_spec.columns for col in required_spec_cols):
        raise KeyError(f"Missing one of required columns: {', '.join(required_spec_cols)}")
        
    df_spec = df_spec.dropna(subset=[spec_ra_col, spec_dec_col])
    print(f"Loaded {len(df_spec)} total spectroscopic sources.")
except FileNotFoundError:
    print(f"ERROR: Spectroscopic CSV file not found at {spec_csv_path}")
    exit()
except KeyError as e:
    print(f"ERROR: {e} not found in {spec_csv_path}")
    exit()

# --- Apply the filename filter ---
print(f"Filtering {len(df_spec)} sources based on filename lists...")
# Keep rows where the 'file' column value exists in the set loaded from filter CSVs
df_spec_filtered = df_spec[df_spec[spec_file_col].isin(valid_filenames)].copy()
n_filtered = len(df_spec_filtered)
print(f"Kept {n_filtered} spectroscopic sources after filtering.")

if n_filtered == 0:
    print("Warning: No spectroscopic sources left after filtering. Plot may be empty.")

# --- Load AND COMBINE Photometry FITS ---
print(f"\nLoading photometric catalogs...")
all_phot_dfs = []
try:
    for path in phot_fits_paths:
        field_name = "GOODS-South" if "GS-South" in path else "GOODS-North"
        print(f"  - Loading {field_name} from {os.path.basename(path)}")
        with fits.open(path) as hdul:
            if photometry_hdu_index >= len(hdul):
                 raise IndexError(f"HDU index {photometry_hdu_index} is out of bounds.")
            
            photometry_data = hdul[photometry_hdu_index].data
            df_phot_current = pd.DataFrame(photometry_data)
            
        df_phot_current = df_phot_current.dropna(subset=[phot_ra_col, phot_dec_col])
        print(f"    Loaded {len(df_phot_current)} sources from {field_name}.")

        # Apply spatial filter ONLY to GOODS-North
        if field_name == "GOODS-North":
            mask = (df_phot_current[phot_ra_col] <= gn_ra_min) & (df_phot_current[phot_dec_col] <= gn_dec_min)
            df_phot_current = df_phot_current[mask]
            print(f"    Filtered {field_name} to {len(df_phot_current)} sources (RA > {gn_ra_min}, Dec > {gn_dec_min}).")

        all_phot_dfs.append(df_phot_current)

except FileNotFoundError as e:
    print(f"ERROR: Photometry FITS file not found. {e}")
    exit()
except (IndexError, TypeError, KeyError) as e:
     print(f"ERROR reading photometry FITS table: {e}")
     exit()

# Combine all loaded photometric dataframes
df_phot_combined = pd.concat(all_phot_dfs, ignore_index=True)
print(f"Total photometric sources to plot (GS-South + filtered GS-North): {len(df_phot_combined)}")


# --- Create the Plot ---
print("Generating footprint plot...")
plt.figure(figsize=(10, 10))

# Plot all photometric sources as a faint, gray background
plt.scatter(
    df_phot_combined[phot_ra_col], 
    df_phot_combined[phot_dec_col], 
    s=1,  # Small size
    alpha=0.1, # Very transparent
    color='gray', 
    label=f"Combined Photometric Catalog (n={len(df_phot_combined)})"
)

# Plot your FILTERED spectroscopic sources on top as bright, red points
plt.scatter(
    df_spec_filtered[spec_ra_col], 
    df_spec_filtered[spec_dec_col], 
    s=5, # Slightly larger
    alpha=0.8, 
    color='red', 
    label=f"Filtered Spectroscopic Sources (n={n_filtered})"
)

# --- Finalize Plot Aesthetics ---
plt.xlabel("Right Ascension (RA) [deg]")
plt.ylabel("Declination (Dec) [deg]")
plt.title("Catalog Footprint Comparison (Filtered Spec vs. Combined GS-S+GS-E)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Invert RA axis, which is standard for sky plots
plt.gca().invert_xaxis() 

plt.tight_layout()
plt.savefig(output_plot_path, dpi=200)

print(f"\nPlot saved successfully to: {output_plot_path}")
print("Check the plot to see if the red (filtered spectroscopic) points fall within the gray (photometric) cloud.")

