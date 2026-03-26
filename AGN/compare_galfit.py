#this code is going to be for making a plot of sersic index against effective radius, and coloured with a colour bar on whether 
#the object has a better fit with psf or sersic, helping define whether its compact
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, vstack


surveys = ['JADES-DR3-GS-East', 'JADES-DR3-GS-North', 'JADES-DR3-GS-South', 'JADES-DR3-GN-Deep', 'JADES-DR3-GN-Medium']

BASE = '/raid/scratch/work/Griley/GALFIND_WORK/GALFIT/output/v13/ACS_WFC+NIRCam'
sersic_tables = []
psf_tables = []

for survey in surveys:
    sersic_path = f"{BASE}/{survey}/F444W/sersic/results.fits"
    psf_path = f"{BASE}/{survey}/F444W/psf/results.fits"
    sersic_tables.append(Table.read(sersic_path))
    psf_tables.append(Table.read(psf_path)) 

sersic_cat = vstack(sersic_tables)
psf_cat = vstack(psf_tables)
print(f"Total objects: {len(sersic_cat)}")
print(f"  - RFF > 0.5: {(sersic_cat['rff'] > 0.5).sum()}")
print(f"  - red_chi2 > 3: {(sersic_cat['red_chi2'] > 3).sum()}")
print(f"  - r_e < 0.1: {(sersic_cat['r_e'] < 0.1).sum()}")

mask = ((sersic_cat['rff']<=0.5) & (sersic_cat['red_chi2']<=3) & (sersic_cat['r_e'] >= 0.1))
sersic_cat = sersic_cat[mask]
psf_cat = psf_cat[mask]
print(f"Objects after quality cuts: {len(sersic_cat)}")


delta_chi2 = sersic_cat['chi2'] - psf_cat['chi2']

fig,ax = plt.subplots(figsize=(8,6))
sc = ax.scatter(
    sersic_cat['r_e'], sersic_cat['n'],
    c = delta_chi2, cmap='RdBu', vmin=-1, vmax=1,
    s = 20, alpha = 0.7,
)
plt.colorbar(sc, label=r'$\Delta \chi^2$ (Sersic - PSF)')
ax.set_xlabel(r'Effective Radius $r_e$ (pixels)')
ax.set_ylabel('Sersic Index $n$')
ax.set_xscale('log')
ax.set_title('F444W')
plt.tight_layout()
plt.savefig('/nvme/scratch/work/Griley/Masters/AGN/sersic_vs_re.png', dpi=150)
plt.show()