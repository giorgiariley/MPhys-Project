# from astropy.table import Table
# import numpy as np
# cat = Table.read('/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/gdsgdn_catalogue.fits')

# for sid in [9235, 23469, 10207]:
#     matches = [r for r in cat if int(r['SURVEY_ID']) == sid]
#     print(f"SURVEY_ID {sid}: {len(matches)} matches")
#     for m in matches:
#         print(f"  phot_RA={m['phot_RA']}, phot_DEC={m['phot_DEC']}")


# from astropy.io import fits
# from astropy.wcs import WCS
# from astropy.nddata import Cutout2D
# from astropy.coordinates import SkyCoord
# import astropy.units as u
# import numpy as np

# checks = [
#     (6897, "JADES-DR3-GN-Medium", 189.17606599717737, 62.25632595460843),
#     (14358, "JADES-DR3-GN-Medium", 189.1366292000999, 62.22343499441153),
#     (26473, "JADES-DR3-GS-South", 53.11649946574956, -27.857289467934955),
#     (65621, "JADES-DR3-GS-South", 53.07281001716544, -27.84584280324058),
# ]

# IMAGE_PATHS = {
#     "JADES-DR3-GS-South": "/raid/scratch/data/jwst/JADES-DR3-GS-South/NIRCam/mosaic_1293_wispnathan/30mas/GOODS_S_South-F444W_i2dnobgnobg.fits",
#     "JADES-DR3-GN-Medium": "/raid/scratch/data/jwst/JADES-DR3-GN-Medium/NIRCam/mosaic_1293_wispnathan/30mas/GOODS-N-PrimaryMedium-F444W_i2dnobg.fits",
# }

# for sid, survey, ra, dec in checks:
#     with fits.open(IMAGE_PATHS[survey]) as hdul:
#         hdr = hdul["SCI"].header if "SCI" in [h.name for h in hdul] else hdul[0].header
#         data = hdul["SCI"].data if "SCI" in [h.name for h in hdul] else hdul[0].data
#     wcs = WCS(hdr)
#     position = SkyCoord(ra=ra, dec=dec, unit="deg")
#     cutout = Cutout2D(data, position, 0.96*u.arcsec, wcs=wcs)
#     print(f"ID {sid}: cutout shape={cutout.data.shape}, "
#           f"all NaN={np.all(np.isnan(cutout.data))}, "
#           f"all zero={np.all(cutout.data==0)}, "
#           f"min={np.nanmin(cutout.data):.4f}, max={np.nanmax(cutout.data):.4f}")

from astropy.table import Table
import numpy as np

cat = Table.read('/nvme/scratch/work/Griley/Masters/morf/combined_morfometryka.fits')
print(cat.colnames)
print(f"\nN rows: {len(cat)}")
print(cat['RnFit2D', 'nFit2D', 'qFit2D'][:5])
print(cat.colnames)
from astropy.table import Table

cat = Table.read('/nvme/scratch/work/Griley/Masters/morf/combined_morfometryka.fits')
print(cat['# rootname9.65'][:5])

from astropy.table import Table
import pandas as pd

cat = Table.read('/nvme/scratch/work/Griley/Masters/morf/combined_morfometryka.fits')

df = pd.DataFrame({
    'rootname': list(cat['# rootname9.65']),
    'RnFit2D':  cat['RnFit2D'],
    'nFit2D':   cat['nFit2D'],
    'qFit2D':   cat['qFit2D'],
})

df.to_csv('/nvme/scratch/work/Griley/Masters/morf/agn_morphology.csv', index=False)

print(df.head(10))
print(f"\nSaved {len(df)} rows")