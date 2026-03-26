#this code is for emasuring the fluxes in 0.2as and in 0.4as to see if things are compact ie AGN

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import SkyCircularAperture, ApertureStats

IMAGE_PATHS = {
    "JADES-DR3-GS-South":  "/raid/scratch/data/jwst/JADES-DR3-GS-South/NIRCam/mosaic_1293_wispnathan/30mas/GOODS_S_South-F444W_i2dnobgnobg.fits",
    "JADES-DR3-GS-East":   "/raid/scratch/data/jwst/JADES-DR3-GS-East/NIRCam/mosaic_1293_wispnathan/30mas/GOODS_S_East-F444W_i2dnobg.fits",
    "JADES-DR3-GS-North":  "/raid/scratch/data/jwst/JADES-DR3-GS-North/NIRCam/mosaic_1293_wispnathan/30mas/GOODS_S_North-F444W_i2dnobg.fits",
    "JADES-DR3-GN-Deep":   "/raid/scratch/data/jwst/JADES-DR3-GN-Deep/NIRCam/mosaic_1293_wispnathan/30mas/GOODS-N-PrimaryDeep-F444W_i2dnobg.fits",
    "JADES-DR3-GN-Medium": "/raid/scratch/data/jwst/JADES-DR3-GN-Medium/NIRCam/mosaic_1293_wispnathan/30mas/GOODS-N-PrimaryMedium-F444W_i2dnobg.fits",
}

df = pd.read_csv('/nvme/scratch/work/Griley/Masters/AGN/subsample_photometric_ids.csv')
df = df.drop_duplicates(subset='SURVEY_ID')

results = []
for survey, group in df.groupby('SURVEY'):
    print(f"Processing survey {survey} with {len(group)} unique objects")
    image_path = IMAGE_PATHS[survey]
    
    with fits.open(image_path) as hdul:
        # find the science extension
        print([h.name for h in hdul])
        sci = hdul['SCI'].data  # adjust if needed
        wcs = WCS(hdul['SCI'].header)
    
    for _, row in group.iterrows():
        coord = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg)
        
        ap_02 = SkyCircularAperture(coord, r=0.1*u.arcsec)  # 0.2as diameter
        ap_04 = SkyCircularAperture(coord, r=0.2*u.arcsec)  # 0.4as diameter
        
        stats_02 = ApertureStats(sci, ap_02, wcs=wcs)
        stats_04 = ApertureStats(sci, ap_04, wcs=wcs)
        
        flux_02 = stats_02.sum
        flux_04 = stats_04.sum
        ratio = flux_02 / flux_04 if flux_04 != 0 else np.nan
        
        results.append({
            'SURVEY_ID': row['SURVEY_ID'],
            'SURVEY': survey,
            'ra': row['ra'],
            'dec': row['dec'],
            'flux_02as': flux_02,
            'flux_04as': flux_04,
            'concentration': ratio,
        })

results_df = pd.DataFrame(results)
print(results_df[['SURVEY_ID', 'flux_02as', 'flux_04as', 'concentration']].head(20))
results_df.to_csv('/nvme/scratch/work/Griley/Masters/AGN/concentration.csv', index=False)