from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

path = "/raid/scratch/data/jwst-miri/SMILES/60mas/hlsp_smiles_jwst_miri_goodss_F560W_v1.0_drz_aligned.fits"

hdul = fits.open(path)
w = WCS(hdul[1].header)
img = hdul[1].data
hdul.close()

fig = plt.figure(figsize=(10,10))
ax = plt.subplot(111, projection=w)

vmin = np.nanpercentile(img, 5)
vmax = np.nanpercentile(img, 99)
ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)

# ----------------------------------------------------------
# Force decimal RA/Dec labels (disable sexagesimal)
# ----------------------------------------------------------
ra = ax.coords[0]
dec = ax.coords[1]

# Display in degrees, not hourangle
ra.set_format_unit('deg')
dec.set_format_unit('deg')

# Make sure tick labels use plain decimal formatting
ra.set_major_formatter('d.ddd')   # 3 decimal places
dec.set_major_formatter('d.ddd')

ra.set_axislabel("Right Ascension (deg)")
dec.set_axislabel("Declination (deg)")

plt.tight_layout()
plt.show()
plt.savefig("miri_f560w_mosaic.png")
