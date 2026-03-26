#this code is for emasuring the fluxes in 0.2as and in 0.4as to see if things are compact ie AGN

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import SkyCircularAperture, ApertureStats
import matplotlib.pyplot as plt
from photutils.centroids import centroid_2dg
from astropy.nddata import Cutout2D


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

        # convert catalogue position to pixel coords
        x_cat, y_cat = wcs.world_to_pixel(coord)
        # make a small cutout around the catalogue position for centroiding
        cutout = Cutout2D(sci, (x_cat, y_cat), size=20, wcs=wcs)
        # find true centroid
        x_cen, y_cen = centroid_2dg(cutout.data)
        # convert centroid back to sky coordinates
        true_coord = cutout.wcs.pixel_to_world(x_cen, y_cen)
        
        ap_02 = SkyCircularAperture(true_coord, r=0.1*u.arcsec)  # 0.2as diameter
        ap_04 = SkyCircularAperture(true_coord, r=0.2*u.arcsec)  # 0.4as diameter
        
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
results_df.to_csv('/nvme/scratch/work/Griley/Masters/AGN/concentration.csv', index=False)
agn_by_eye = [20964, 25515, 55994, 47644, 55940, 10207, 16707, 20897, 15646, 23813 ]
for agn_id in agn_by_eye:
    row = results_df[results_df['SURVEY_ID'] == agn_id]
    if len(row) > 0:
        print(f"ID {agn_id}: concentration = {row['concentration'].values[0]:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(results_df['concentration'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Concentration (flux 0.2as / flux 0.4as)')
ax.set_ylabel('N')
ax.set_title('F444W Concentration')
plt.tight_layout()
plt.savefig('/nvme/scratch/work/Griley/Masters/AGN/concentration_hist.png', dpi=150)
