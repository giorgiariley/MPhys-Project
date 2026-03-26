import os 
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'

import astropy.units as u
import pandas as pd
from galfind import Catalogue, ID_Selector, Filter, PSF_Cutout, Galfit_Fitter
from galfind.Data import morgan_version_to_dir

#=== Parameters===
version = 'v13'
instrument_names = ['ACS_WFC', 'NIRCam']
aper_diams = [0.32] * u.arcsec
forced_phot_band = ["F277W", "F356W", "F444W"]
min_flux_pc_err = 10.0
filt_name = "F444W"
PSF_BASE = "/raid/scratch/work/austind/GALFIND_WORK/PSFs/NIRCam/v13"
psf_survey_map = {
    'JADES-DR3-GS-East': 'JADES-GS',
    'JADES-DR3-GS-North': 'JADES-GS',
    'JADES-DR3-GS-South': 'JADES-GS',
    'JADES-DR3-GN-Deep': 'JADES-GN',
    'JADES-DR3-GN-Medium': 'JADES-GN',
}

#=== Load and deduplicate the subsample ===
df = pd.read_csv('/nvme/scratch/work/Griley/Masters/AGN/subsample_photometric_ids.csv')
df = df.drop_duplicates(subset='SURVEY_ID')
print(f"Unique objects after deduplication: {len(df)}")

#create a galfind filter object for F444W
filt = Filter.from_filt_name(filt_name)

#loop over all surveys 
for survey in df['SURVEY'].unique():
    survey_ids = df[df['SURVEY'] == survey]['SURVEY_ID'].astype(int).tolist()
    print(f"Processing survey {survey} with {len(survey_ids)} unique objects")
    
    #load catalogue
    print(f"Loading catalogue for {survey}..." )
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

    print(f"Full catalogue: {len(cat.ID)} objects")

    #filter to subsample 
    id_selector = ID_Selector(survey_ids, f"agn_subsample_{survey}_3")    
    cat = id_selector(cat, return_copy = True)
    print(f"Filtered catalogue: {len(cat.ID)} objects")

    cat.load_sextractor_auto_mags()
    cat.load_sextractor_auto_fluxes()
    cat.load_sextractor_kron_radii()
    cat.load_sextractor_Re()

    #Load PSF for this survey
    psf_survey = psf_survey_map[survey]
    psf_path = f"{PSF_BASE}/{psf_survey}/{filt_name}/empirical/pixscl=0.03as_size=3.0as_galfind.fits"
    psf = PSF_Cutout.from_fits(fits_path=psf_path, filt=filt, unit="adu", pix_scale=0.03 * u.arcsec, size=3.0 * u.arcsec)
    # galfit_psf_fitter = Galfit_Fitter(psf, "psf")
    # galfit_psf_fitter(cat)
    # galfit_sersic_fitter = Galfit_Fitter(psf, "sersic")
    # galfit_sersic_fitter(cat)
    galfit_sersic_fixed_fitter = Galfit_Fitter(psf, "sersic", fixed_params = ["n"])
    galfit_sersic_fixed_fitter(cat)


    print(f"Finished PSF fitting for {survey}")

