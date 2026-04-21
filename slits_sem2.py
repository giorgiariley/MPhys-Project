from msaexp import msa
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS

spec_path = '/raid/scratch/work/austind/GALFIND_WORK/Spectra/2D/gds-deep-v4/gds-deep-v4_prism-clear_1210_13577.spec.fits'
MSA_METAFILE_BASE = '/raid/scratch/work/austind/GALFIND_WORK/Spectra/MSA_metafiles'
cutout_path = '/raid/scratch/work/Griley/GALFIND_WORK/Cutouts/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/3.00as/F444W/data/48772.fits'

#reading the specfit header
with fits.open(spec_path) as hdul:
    h = hdul['SCI'].header
    msametfl = h['MSAMETFL']
    msametid = int(h['MSAMETID'])
    patt_num = int(h['PATT_NUM'])

print(f"MSA Metafile: {msametfl}, MSA Metadata ID: {msametid}, Dither Point Index: {patt_num}")

metafile_path = f"{MSA_METAFILE_BASE}/{msametfl}"
print(f"Loading MSA metafile: {metafile_path}")
#load MSA metafile 
MSA_metafile = msa.MSAMetafile(metafile_path)

#get the slit regions
slits = MSA_metafile.regions_from_metafile(
    dither_point_index=patt_num,
    as_string=False,
    with_bars=True,
    msa_metadata_id=msametid,
)

#load cutout image and WCS
with fits.open(cutout_path) as hdul:
    image = hdul['SCI'].data
    wcs = WCS(hdul['SCI'].header)

fig, ax = plt.subplots(figsize=(5,5))
vmin = np.percentile(image, 10)
vmax = np.percentile(image, 99.5)
ax.imshow(image, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)

for s in slits:
    xy = np.array(s.xy[0])  # shape (4, 2) - RA/Dec corners
    # convert corners to pixel coordinates
    pixels = wcs.world_to_pixel_values(xy[:, 0], xy[:, 1])
    x_pix = np.append(pixels[0], pixels[0][0])  # close the rectangle
    y_pix = np.append(pixels[1], pixels[1][0])
    
    if s.meta['is_source']:
        color = 'magenta'
        lw = 2.0
    else:
        color = 'lightpink'
        lw = 2.0
    
    ax.plot(x_pix, y_pix, color=color, lw=lw, alpha=0.8)

ax.set_xlim(0, image.shape[1])
ax.set_ylim(0, image.shape[0])
ax.axis('off')
ax.set_title('1210_13577 | 48772')
plt.tight_layout()
plt.savefig('/nvme/scratch/work/Griley/Masters/slit_overlays/1210_13577.png', dpi=150)
plt.show()