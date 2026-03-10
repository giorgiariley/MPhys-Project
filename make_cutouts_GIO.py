"""
Standalone cutout script — no galfind required.
Uses astropy Cutout2D to extract F444W stamps for objects in subsample CSV.
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.table import Table

# === Paths ===
SUBSAMPLE_CSV   = "/nvme/scratch/work/Griley/Masters/subsample_photometric_ids.csv"
CATALOGUE_FITS  = "/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/gdsgdn_catalogue.fits"
OUTPUT_DIR      = "/raid/scratch/work/Griley/GALFIND_WORK/Cutouts/v13/HighHeHaGIO"
CUTOUT_SIZE     = 0.96 * u.arcsec
FILTER          = "F444W"

# Image path template — fill in survey short name and filter
# e.g. /raid/scratch/data/jwst/JADES-DR3-GS-South/NIRCam/mosaic_1293_wispnathan/30mas/GOODS_S_South-F444W_i2dnobgnobg.fits
IMAGE_PATHS = {
    "JADES-DR3-GS-South":  "/raid/scratch/data/jwst/JADES-DR3-GS-South/NIRCam/mosaic_1293_wispnathan/30mas/GOODS_S_South-F444W_i2dnobgnobg.fits",
    "JADES-DR3-GS-East":   "/raid/scratch/data/jwst/JADES-DR3-GS-East/NIRCam/mosaic_1293_wispnathan/30mas/GOODS_S_East-F444W_i2dnobg.fits",
    "JADES-DR3-GS-North":  "/raid/scratch/data/jwst/JADES-DR3-GS-North/NIRCam/mosaic_1293_wispnathan/30mas/GOODS_S_North-F444W_i2dnobg.fits",
    "JADES-DR3-GN-Deep":   "/raid/scratch/data/jwst/JADES-DR3-GN-Deep/NIRCam/mosaic_1293_wispnathan/30mas/GOODS-N-PrimaryDeep-F444W_i2dnobg.fits",
    "JADES-DR3-GN-Medium": "/raid/scratch/data/jwst/JADES-DR3-GN-Medium/NIRCam/mosaic_1293_wispnathan/30mas/GOODS-N-PrimaryMedium-F444W_i2dnobg.fits",
}

# === Load subsample CSV ===
df = pd.read_csv(SUBSAMPLE_CSV)
df = df.dropna(subset=["SURVEY_ID"])
df["SURVEY_ID"] = df["SURVEY_ID"].astype(int)

# === Load master FITS catalogue for RA/Dec ===
print("Loading master catalogue...")
cat = Table.read(CATALOGUE_FITS)


survey_id_to_radec = {}
for row in cat:
    survey = row["SURVEY"].decode().strip() if isinstance(row["SURVEY"], bytes) else str(row["SURVEY"]).strip()
    sid = int(row["SURVEY_ID"])
    ra, dec = float(row["phot_RA"]), float(row["phot_DEC"])
    key = (survey, sid)
    if key not in survey_id_to_radec and not (np.isnan(ra) or np.isnan(dec)):
        survey_id_to_radec[key] = (ra, dec)
print(f"Loaded {len(survey_id_to_radec)} objects from master catalogue.")

# === Loop over surveys ===
for survey, image_path in IMAGE_PATHS.items():
    survey_df = df[df["SURVEY"] == survey]
    if len(survey_df) == 0:
        print(f"No objects for {survey}, skipping.")
        continue

    if image_path is None:
        print(f"WARNING: No image path set for {survey}, skipping. Please fill in IMAGE_PATHS.")
        continue

    if not os.path.exists(image_path):
        print(f"WARNING: Image not found at {image_path}, skipping.")
        continue

    print(f"\nProcessing {survey} ({len(survey_df)} objects)...")

    # Load image once per survey
    with fits.open(image_path) as hdul:
        # Try SCI extension first, fall back to primary
        if "SCI" in [h.name for h in hdul]:
            data = hdul["SCI"].data
            header = hdul["SCI"].header
        else:
            data = hdul[0].data
            header = hdul[0].header
    wcs = WCS(header)

    # Output directory per survey
    out_dir = os.path.join(OUTPUT_DIR, survey)
    os.makedirs(out_dir, exist_ok=True)

    for _, row in survey_df.iterrows():
        survey_id = int(row["SURVEY_ID"])

        if survey_id not in [k[1] for k in survey_id_to_radec if k[0] == survey]:
            print(f"  WARNING: SURVEY_ID {survey_id} not found for {survey}, skipping.")
            continue
        ra, dec = survey_id_to_radec.get((survey, survey_id))
        position = SkyCoord(ra=ra, dec=dec, unit="deg")

        try:
            cutout = Cutout2D(data, position, CUTOUT_SIZE, wcs=wcs)

            fig, ax = plt.subplots(figsize=(4, 4))
            vmin, vmax = np.nanpercentile(cutout.data, [1, 99])
            ax.imshow(cutout.data, origin="lower", cmap="gray_r",
                      vmin=vmin, vmax=vmax)
            ax.set_title(f"{survey}\nID {survey_id}", fontsize=8)
            ax.axis("off")

            filename = f"ID_{survey_id}_{FILTER}_cutout.png"
            fig.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {filename}")

        except Exception as e:
            print(f"  WARNING: Failed for ID {survey_id}: {e}")

print("\nDone.")