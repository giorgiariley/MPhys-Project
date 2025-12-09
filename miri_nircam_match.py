import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits

# ----------------------------------------------------
# INPUT FILES
# ----------------------------------------------------
MATCHING_IDS_CSV = "/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/specFitMSA/data/project_mphys_ultrablue/matching_ids_sample.csv"

F560W_CSV = "/nvme/scratch/work/Griley/Masters/miri/F560W_fluxes.csv"
F770W_CSV = "/nvme/scratch/work/Griley/Masters/miri/F770W_fluxes.csv"

south_cat_path = "/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-South/(0.32)as/JADES-DR3-GS-South_MASTER_Sel-F277W+F356W+F444W_v13.fits"
east_cat_path  = "/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits"

# ----------------------------------------------------
# LOAD MATCHING FILE
# ----------------------------------------------------
ids_df = pd.read_csv(MATCHING_IDS_CSV)
ids_df = ids_df[["spectroscopy_id", "photo_id_south", "photo_id_east"]]

print("Loaded spectroscopy ↔ photometry table:")
print(ids_df.head())

# ----------------------------------------------------
# LOAD MIRI POSITIONS (these use spectroscopy IDs)
# ----------------------------------------------------
miri_560 = pd.read_csv(F560W_CSV)
miri_770 = pd.read_csv(F770W_CSV)

# Merge 560+770 flux tables
miri_df = pd.merge(
    miri_560, miri_770,
    on=["object_id", "ra", "dec"],
    how="inner"
)

# Rename object_id → spectroscopy_id so it matches ids_df
miri_df = miri_df.rename(columns={"object_id": "spectroscopy_id"})

# Keep only spectroscopy IDs in the matching file (the 39 sources)
miri_df = miri_df[miri_df["spectroscopy_id"].isin(ids_df["spectroscopy_id"])]

print(f"\nMIRI detections matching spectroscopy IDs: {len(miri_df)}")

# ----------------------------------------------------
# LOAD GOODS-S AND GOODS-E TABLES
# ----------------------------------------------------
def load_objects_table(path):
    with fits.open(path) as hdul:
        return hdul["OBJECTS"].data

south = load_objects_table(south_cat_path)
east  = load_objects_table(east_cat_path)

# Convenience lookups
south_df = pd.DataFrame(south)
east_df  = pd.DataFrame(east)

# ----------------------------------------------------
# COMPUTE SEPARATIONS
# ----------------------------------------------------
separations_arcsec = []

print("\nComputing separations...")

for _, row in miri_df.iterrows():

    spec_id = int(row["spectroscopy_id"])
    ra_miri = row["ra"]
    dec_miri = row["dec"]

    # Find matching photometry ID
    match_row = ids_df[ids_df["spectroscopy_id"] == spec_id].iloc[0]

    pid_s = match_row["photo_id_south"]
    pid_e = match_row["photo_id_east"]

    # Get photometric RA/Dec from GOODS-S or GOODS-E
    if pd.notna(pid_s):
        phot = south_df[south_df["NUMBER"] == pid_s]
    else:
        phot = east_df[east_df["NUMBER"] == pid_e]

    if len(phot) == 0:
        continue

    ra_nir = float(phot["ALPHA_J2000"])
    dec_nir = float(phot["DELTA_J2000"])

    # Compute sky separation
    c_miri = SkyCoord(ra_miri*u.deg, dec_miri*u.deg)
    c_nir  = SkyCoord(ra_nir*u.deg, dec_nir*u.deg)

    sep = c_miri.separation(c_nir).arcsec
    separations_arcsec.append(sep)

    print(f"ID {spec_id}: separation = {sep:.3f} arcsec")

# ----------------------------------------------------
# SUMMARY STATISTICS
# ----------------------------------------------------
separations_arcsec = np.array(separations_arcsec)

print("\n=== MATCH SEPARATION STATISTICS (arcsec) ===")
print(f"Median separation:  {np.median(separations_arcsec):.4f}")
print(f"Mean separation:    {np.mean(separations_arcsec):.4f}")
print(f"95th percentile:    {np.percentile(separations_arcsec, 95):.4f}")
print(f"Max separation:     {np.max(separations_arcsec):.4f}")
