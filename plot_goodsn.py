import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import numpy as np

# ---------------------- CONFIGURATION ----------------------

# Input File: ONLY the GOODS-North catalog
phot_fits_path = "/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits"

# Output Plot
output_plot_path = "/nvme/scratch/work/Griley/Masters/goods_north_footprint_check.png"

# Column names
phot_ra_col = 'ALPHA_J2000'
phot_dec_col = 'DELTA_J2000'
photometry_hdu_index = 1

# ---------------------- SCRIPT START ----------------------

print(f"Loading GOODS-North photometric catalog from: {phot_fits_path}")
try:
    with fits.open(phot_fits_path) as hdul:
        if photometry_hdu_index >= len(hdul):
             raise IndexError(f"HDU index {photometry_hdu_index} is out of bounds. File has {len(hdul)} HDUs.")
        
        photometry_data = hdul[photometry_hdu_index].data
        df_phot = pd.DataFrame(photometry_data)
        
    df_phot = df_phot.dropna(subset=[phot_ra_col, phot_dec_col])
    print(f"Loaded {len(df_phot)} total photometric sources from GOODS-North.")
except FileNotFoundError:
    print(f"ERROR: Photometry FITS file not found at {phot_fits_path}")
    exit()
except (IndexError, TypeError, KeyError) as e:
     print(f"ERROR reading photometry FITS table: {e}")
     exit()

# --- Create the Plot ---
print("Generating GOODS-North footprint plot...")
plt.figure(figsize=(10, 10))

# Plot all photometric sources
plt.scatter(
    df_phot[phot_ra_col], 
    df_phot[phot_dec_col], 
    s=1,  # Small size
    alpha=0.1, # Very transparent
    color='blue', # Use blue for clarity
    label=f"GOODS-North Photometric Catalog (n={len(df_phot)})"
)

# --- Finalize Plot Aesthetics ---
plt.xlabel("Right Ascension (RA) [deg]", fontsize=12)
plt.ylabel("Declination (Dec) [deg]", fontsize=12)
plt.title("GOODS-North Full Photometric Footprint", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Invert RA axis, which is standard for sky plots
plt.gca().invert_xaxis() 

plt.tight_layout()
plt.savefig(output_plot_path, dpi=200)

print(f"\nPlot saved successfully to: {output_plot_path}")
print("Check this plot to see the full RA/Dec distribution for GOODS-North.")
