#this script will make a csv from the high helium halpha csv file that also contains the photmetric id 
"""
Produces a subsample CSV of JWST photometric IDs from a master FITS catalogue.

Filter condition:
    filename must appear in Ha_6565_SN (col 3)
    AND in HeII_4687_SN (col 1) OR HeII_1640_SN (col 2) or both.

Output columns: SURVEY, id_phot
"""

import pandas as pd
from astropy.table import Table

# ── Paths ─────────────────────────────────────────────────────────────────────
FITS_PATH = "/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/gdsgdn_catalogue.fits"
CSV_PATH  = ("/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/specFitMSA/data/project_mphys_ultrablue/prism_subset.csv")
OUT_PATH  = "subsample_photometric_ids.csv"

# ── Load the filter CSV ────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# ha      = set(df["Ha_6565_SN"].dropna())       # col 3 — must be present
# heii_47 = set(df["HeII_4687_SN"].dropna())     # col 1
# heii_16 = set(df["HeII_1640_SN"].dropna())     # col 2

# # Condition: in Ha AND (in HeII_4687 OR HeII_1640)
# qualifying = ha & (heii_47 | heii_16)

# print(f"Ha_6565_SN entries      : {len(ha)}")
# print(f"HeII_4687_SN entries    : {len(heii_47)}")
# print(f"HeII_1640_SN entries    : {len(heii_16)}")
# print(f"Qualifying files        : {len(qualifying)}")

# ── Load the master FITS catalogue ─────────────────────────────────────────────
cat = Table.read(FITS_PATH).to_pandas()

# Normalise the 'file' column (strip whitespace/bytes if needed)
# NEW — handles bytes correctly
cat["file"] = cat["file"].apply(
    lambda x: x.decode().strip() if isinstance(x, bytes) else str(x).strip()
)

# ── Filter and export ──────────────────────────────────────────────────────────
cat["SURVEY"] = cat["SURVEY"].apply(
    lambda x: x.decode().strip() if isinstance(x, bytes) else str(x).strip()
)
qualifying = set(df["file"].dropna())
subsample = pd.merge(
    df[["file", "PROG_ID", "Index"]], # Columns to take from the CSV
    cat[["file", "SURVEY", "SURVEY_ID", "id_phot"]], # Columns to take from FITS
    on="file", 
    how="inner"
)

print(f"Rows in subsample       : {len(subsample)}")
print(subsample.head(10))

subsample.to_csv(OUT_PATH, index=False)
print(f"\nSaved to: {OUT_PATH}")