import pandas as pd
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
import matplotlib as mpl



# --- Global Matplotlib style for publication-quality plots ---
mpl.rcParams.update({
    "font.size": 20,             # Base font size
    "axes.titlesize": 20,        # Title size
    "axes.labelsize": 20,        # Axis label size
    "xtick.labelsize": 16,       # X tick label size
    "ytick.labelsize": 16,       # Y tick label size
    "legend.fontsize": 20,       # Legend text
    "figure.titlesize": 20,      # Overall figure title
    "axes.linewidth": 1.4,       # Thicker axes
    "xtick.major.width": 1.2,    # Tick line width
    "ytick.major.width": 1.2,
    "xtick.major.size": 6,       # Tick size
    "ytick.major.size": 6,
    "lines.linewidth": 1.6,      # Slightly thicker default lines
    "savefig.dpi": 300,          # High-resolution output for publication
})


# --- Paths ---
galaxies_path = '/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/specFitMSA/data/project_mphys_ultrablue/matching_ids_sample.csv'
galaxies_df = pd.read_csv(galaxies_path)

south_cat_path = "/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-South/(0.32)as/JADES-DR3-GS-South_MASTER_Sel-F277W+F356W+F444W_v13.fits"
east_cat_path  = "/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits"

# --- Load catalogues ---
def load_objects_table(fits_path):
    with fits.open(fits_path) as hdul:
        return hdul['OBJECTS'].data

south_data = load_objects_table(south_cat_path)
east_data  = load_objects_table(east_cat_path)

# --- Compute colours and their errors for catalogues ---
def compute_colours_and_errors(data):
    # Fluxes
    F444W = data['MAG_APER_F444W']
    F410M = data['MAG_APER_F410M']
    F115W = data['MAG_APER_F115W']
    F150W = data['MAG_APER_F150W']
    # Magnitude errors
    F444W_err = data['MAGERR_APER_F444W']
    F410M_err = data['MAGERR_APER_F410M']
    F115W_err = data['MAGERR_APER_F115W']
    F150W_err = data['MAGERR_APER_F150W']
    
    # Colours
    colour1 = F410M - F444W
    colour2 = F115W - F150W
    
    # Propagated errors
    colour1_err = np.sqrt(F410M_err**2 + F444W_err**2)
    colour2_err = np.sqrt(F115W_err**2 + F150W_err**2)
    
    return colour1, colour2, colour1_err, colour2_err

# Compute for South and East
south_colour1, south_colour2, south_err1, south_err2 = compute_colours_and_errors(south_data)
east_colour1, east_colour2, east_err1, east_err2 = compute_colours_and_errors(east_data)

# --- Match sample galaxies and propagate errors ---
gal_colour1 = []
gal_colour2 = []
gal_err1 = []
gal_err2 = []

for idx, row in galaxies_df.iterrows():
    if not pd.isna(row['photo_id_south']):
        mask = south_data['NUMBER'] == row['photo_id_south']
        if np.any(mask):
            gal_colour1.append(south_colour1[mask][0])
            gal_colour2.append(south_colour2[mask][0])
            gal_err1.append(south_err1[mask][0])
            gal_err2.append(south_err2[mask][0])
    elif not pd.isna(row['photo_id_east']):
        mask = east_data['NUMBER'] == row['photo_id_east']
        if np.any(mask):
            gal_colour1.append(east_colour1[mask][0])
            gal_colour2.append(east_colour2[mask][0])
            gal_err1.append(east_err1[mask][0])
            gal_err2.append(east_err2[mask][0])

# Convert to numpy arrays
gal_colour1 = np.array(gal_colour1)
gal_colour2 = np.array(gal_colour2)
gal_err1 = np.array(gal_err1)
gal_err2 = np.array(gal_err2)

# --- Plot with error bars ---
plt.figure(figsize=(6,6))
plt.errorbar(gal_colour2, gal_colour1, xerr=gal_err2, yerr=gal_err1,
             fmt='o', markersize=6, color='blue', ecolor='gray', elinewidth=1,
             capsize=2, label='Sample Galaxies')

plt.xlabel('F115W - F150W [mag]')
plt.ylabel('F410M - F444W [mag]')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.title('JWST NIRCam Colour-Colour Diagram (Sample Galaxies) with Errors')

plt.savefig('sample_galaxies_colour_colour_errors.png', dpi=300, bbox_inches='tight')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area

# ----------------------------------------------------
# 1. CONFIGURATION & PATHS
# ----------------------------------------------------
LINKING_CSV = '/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/specFitMSA/data/project_mphys_ultrablue/matching_ids_sample.csv'

F560_CSV = "/nvme/scratch/work/Griley/Masters/miri/F560W_fluxes.csv"
F770_CSV = "/nvme/scratch/work/Griley/Masters/miri/F770W_fluxes.csv"

# We need the images just to calculate pixel area for unit conversion
F560_IMG = "/raid/scratch/data/jwst-miri/SMILES/60mas/hlsp_smiles_jwst_miri_goodss_F560W_v1.0_drz_aligned.fits"
F770_IMG = "/raid/scratch/data/jwst-miri/SMILES/60mas/hlsp_smiles_jwst_miri_goodss_F770W_v1.0_drz_aligned.fits"

SOUTH_CAT_PATH = "/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-South/(0.32)as/JADES-DR3-GS-South_MASTER_Sel-F277W+F356W+F444W_v13.fits"
EAST_CAT_PATH  = "/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits"

# ----------------------------------------------------
# 2. HELPER FUNCTIONS
# ----------------------------------------------------
def load_objects_table(path):
    with fits.open(path) as hdul:
        return hdul["OBJECTS"].data

def get_mjy_to_ujy_factor(fits_path):
    """
    Calculates the factor to convert (Sum of MJy/sr) -> uJy
    Factor = (Pixel Area in sr) * 1e12
    """
    with fits.open(fits_path) as hdul:
        w = WCS(hdul[1].header)
        
    # Calculate pixel area in deg^2, convert to steradians
    area_deg2 = proj_plane_pixel_area(w)
    area_sr = area_deg2 * (np.pi / 180.0)**2
    
    # MJy -> uJy is 1e12
    return area_sr * 1e12

def flux_to_abmag(flux, unit_zero_point_ujy=1.0):
    """
    Convert flux to AB Magnitude.
    """
    flux_ujy = flux * unit_zero_point_ujy
    mags = np.full_like(flux_ujy, np.nan, dtype=float)
    valid = flux_ujy > 0
    mags[valid] = -2.5 * np.log10(flux_ujy[valid]) + 23.90
    return mags

# ----------------------------------------------------
# 3. LOAD DATA
# ----------------------------------------------------
print("Loading JADES catalogues...")
south_data = load_objects_table(SOUTH_CAT_PATH)
east_data  = load_objects_table(EAST_CAT_PATH)

print(f"Loading linking file...")
galaxies_df = pd.read_csv(LINKING_CSV)

# Fix column name
if 'spectroscopy_id' in galaxies_df.columns:
    print("Renaming 'spectroscopy_id' to 'object_id'...")
    galaxies_df = galaxies_df.rename(columns={'spectroscopy_id': 'object_id'})

# Load MIRI CSVs
df_560 = pd.read_csv(F560_CSV)
df_770 = pd.read_csv(F770_CSV)
print(len(df_560), len(df_770))


# ----------------------------------------------------
# 5. MERGE DATA
# ----------------------------------------------------
miri_combined = pd.merge(
    df_560, df_770, 
    on=["object_id", "ra", "dec"], 
    suffixes=("_560", "_770")
)

merged_df = pd.merge(galaxies_df, miri_combined, on="object_id", how="inner")
print(len(merged_df))

# ----------------------------------------------------
# 6. EXTRACT F444W (JADES)
# ----------------------------------------------------
f444_fluxes = []

for idx, row in merged_df.iterrows():
    f444_val = np.nan
    # South
    if not pd.isna(row['photo_id_south']):
        mask = south_data['NUMBER'] == int(row['photo_id_south'])
        if np.any(mask):
            f444_val = south_data['FLUX_APER_F444W'][mask][0]
    # East
    elif not pd.isna(row['photo_id_east']):
        mask = east_data['NUMBER'] == int(row['photo_id_east'])
        if np.any(mask):
            f444_val = east_data['FLUX_APER_F444W'][mask][0]
    
    f444_fluxes.append(f444_val)

merged_df["flux_f444w_njy"] = np.array(f444_fluxes)

# ----------------------------------------------------
# 7. CALCULATE COLOURS
# ----------------------------------------------------
valid_mask = (
    (merged_df["flux_f560w"] > 0) & 
    (merged_df["flux_f770w"] > 0) & 
    (merged_df["flux_f444w_njy"] > 0)
)
plot_df = merged_df[valid_mask].copy()

# MIRI is now definitely uJy -> factor 1.0
plot_df["mag560"] = flux_to_abmag(plot_df["flux_f560w"].values, unit_zero_point_ujy=1.0)
plot_df["mag770"] = flux_to_abmag(plot_df["flux_f770w"].values, unit_zero_point_ujy=1.0)

# JADES is nJy -> factor 0.001
plot_df["mag444"] = flux_to_abmag(plot_df["flux_f444w_njy"].values, unit_zero_point_ujy=0.001)

plot_df["colour_x"] = plot_df["mag560"] - plot_df["mag770"]
plot_df["colour_y"] = plot_df["mag444"] - plot_df["mag560"]

# ----------------------------------------------------
# 8. PLOT
# ----------------------------------------------------
plt.figure(figsize=(7,6))

plt.scatter(
    plot_df["colour_x"], plot_df["colour_y"],
    s=70, edgecolor="black", facecolor="cyan", alpha=0.9, zorder=10,
    label="Our galaxies"
)

# Pop III region markers

# Vertical line: at x = -1.1, going from y = 0.85 up to 10 (or top of plot)
plt.vlines(x=-1.5, ymin=0.85, ymax=10, color="blue", linewidth=2)

# Horizontal line: at y = 0.9, going from x = -10 (or far left) up to -1.5
plt.hlines(y=0.9, xmin=-10, xmax=-1.5, color="blue", linewidth=2)
plt.text(-3, 2, "Pop III ", color="blue", fontsize=17, fontweight='bold')
plt.text(-3, 1.05, "Region", color="blue", fontsize=17, fontweight='bold')


plt.xlabel("F560W $-$ F770W [mag]")
plt.ylabel("F444W $-$ F560W [mag]")
plt.title(f"MIRI Colour–Colour Diagram ({len(plot_df)} Objects)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc='upper right')

# Reset limits to standard view
plt.xlim(-3, 2)
plt.ylim(-2, 8)

plt.savefig("miri_colour_colour_final.png", dpi=300, bbox_inches="tight")
plt.show()