import astropy.units as u
import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'

from galfind.Data import morgan_version_to_dir
from galfind import Catalogue, EAZY, SED_code, Redshift_Extractor
from galfind import Catalogue_Cutouts, ID_Selector
from galfind import galfind_logger
from galfind import Bagpipes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# === Global Parameters ===
version             = "v13"
instrument_names    = ["ACS_WFC", "NIRCam"]
aper_diams          = [0.32] * u.arcsec
forced_phot_band    = ["F277W", "F356W", "F444W"]
min_flux_pc_err     = 10.
cutout_filter       = "F444W"
cutout_size_arcsec  = 0.96 * u.arcsec

filter_id_csv_path  = '/nvme/scratch/work/Griley/Masters/subsample_photometric_ids.csv'

SED_fitter_arr      = [EAZY({"templates": "fsps_larson", "lowz_zmax": None})]
sample_SED_fitter_arr = []

# === Load the full subsample CSV ===
df_filter = pd.read_csv(filter_id_csv_path)
df_filter = df_filter.dropna(subset=["id_phot"])          # drop rows with no id_phot
df_filter["id_phot"] = df_filter["id_phot"].astype(int)

unique_surveys = df_filter["SURVEY"].unique()
print(f"Found {len(unique_surveys)} unique surveys: {unique_surveys}\n")

# === Loop over each survey ===
for survey in unique_surveys:
    print(f"\n{'='*60}")
    print(f"Processing survey: {survey}")
    print(f"{'='*60}")

    # IDs belonging to this survey only
    survey_ids = df_filter.loc[df_filter["SURVEY"] == survey, "id_phot"].values
    print(f"  Objects to select: {len(survey_ids)}")

    # --- Load full catalogue for this survey ---
    print(f"  Loading full catalogue...")
    cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names=instrument_names,
        version_to_dir_dict=morgan_version_to_dir,
        aper_diams=aper_diams,
        forced_phot_band=forced_phot_band,
        min_flux_pc_err=min_flux_pc_err,
        crops=None
    )
    print(f"  Full catalogue loaded with {len(cat.ID)} objects.")

    # --- Apply ID filter ---
    id_selector = ID_Selector(survey_ids, f"selected_from_csv_{survey}")
    cat_selected = id_selector(cat)
    print(f"  Filtered catalogue: {len(cat_selected.ID)} objects.")

    if len(cat_selected.ID) == 0:
        print(f"  WARNING: No matching objects found for {survey}, skipping.")
        continue

    # --- Run SED fitters ---
    print(f"  Running SED fitters...")
    for SED_fitter in SED_fitter_arr:
        for aper_diam in aper_diams:
            SED_fitter(cat_selected, aper_diam, load_PDFs=True, load_SEDs=True, update=True)

    for SED_fitter in sample_SED_fitter_arr:
        for aper_diam in aper_diams:
            SED_fitter(cat_selected, aper_diam, load_PDFs=False, load_SEDs=True,
                       update=True, temp_label='temp')

    # --- Generate cutouts ---
    print(f"  Generating cutouts...")
    cutouts_to_plot = Catalogue_Cutouts.from_cat_filt(
        cat_selected,
        cutout_filter,
        cutout_size_arcsec,
        overwrite=False
    )
    print(f"  Loaded {len(cutouts_to_plot)} cutout objects.")

    # --- Save individual cutout PNGs ---
    individual_cutout_dir = f"{survey}_{version}_individual_cutouts_{cutout_filter}_sem2"
    os.makedirs(individual_cutout_dir, exist_ok=True)
    print(f"  Saving PNGs to: {individual_cutout_dir}")

    for cutout in cutouts_to_plot:
        try:
            galaxy_id = cutout.meta['ID']
            fig, ax = plt.subplots(figsize=(6, 6))
            cutout.plot(ax=ax, imshow_kwargs={'cmap': 'gray_r'})
            ax.set_title(f"{survey} ID {galaxy_id} - {cutout_filter}")
            filename = f"ID_{galaxy_id}_{cutout_filter}_cutout.png"
            fig.savefig(os.path.join(individual_cutout_dir, filename), dpi=150)
            plt.close(fig)
        except Exception as e:
            try:
                gid = cutout.meta['ID']
            except:
                gid = 'unknown'
            print(f"  WARNING: Failed to plot cutout for ID {gid}: {e}")

    print(f"  Done with {survey}.")

print("\nAll surveys processed.")