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
sersic_fixed_tables = []


#looping thru the different surveys
for survey in surveys:
    sersic_path = f"{BASE}/{survey}/F444W/sersic/results.fits"
    psf_path = f"{BASE}/{survey}/F444W/psf/results.fits"
    sersic_fixed_path = f"{BASE}/{survey}/F444W/sersic_fixed_n/results.fits"
    sersic_tables.append(Table.read(sersic_path))
    psf_tables.append(Table.read(psf_path)) 
    sersic_fixed_tables.append(Table.read(sersic_fixed_path))

#constructing catalogues and applying quality cuts
sersic_cat = vstack(sersic_tables)
psf_cat = vstack(psf_tables)
sersic_fixed_cat = vstack(sersic_fixed_tables)
print(f"Total objects: {len(sersic_cat)}")

mask = ((sersic_cat['rff']<=0.5) & (sersic_cat['red_chi2']<=3) & (sersic_cat['r_e'] >= 0.1))
sersic_cat = sersic_cat[mask]
psf_cat = psf_cat[mask]
sersic_fixed_cat = sersic_fixed_cat[mask]
print(f"Objects after quality cuts: {len(sersic_cat)}, with {len(psf_cat)} PSF fits")

#replacing the sersic catalogue for anything with n > 10 
replace_mask = sersic_cat['n'] > 10
print(f"Objects with n >= 10 replaced by fixed n=1: {replace_mask.sum()}")
sersic_combined = sersic_cat.copy()
sersic_combined[replace_mask] = sersic_fixed_cat[replace_mask]

#calculating which fit is better
delta_chi2 = sersic_combined['red_chi2'] - psf_cat['red_chi2']
#plot
fig,ax = plt.subplots(figsize=(8,6))
sc = ax.scatter(
    sersic_combined['r_e'], sersic_combined['n'],
    c = delta_chi2, cmap='spring', vmin = -1, vmax = 1,
    s = 20, alpha = 0.7,
)

plt.colorbar(sc, label=r'$\Delta \chi^2_{\rm red}$ (Sersic - PSF)')
ax.set_xlabel(r'Effective Radius $r_e$ (pixels)')
ax.set_ylabel('Sersic Index $n$')
ax.set_xscale('log')
ax.set_title('F444W')
plt.tight_layout()
plt.savefig('/nvme/scratch/work/Griley/Masters/AGN/sersic_vs_re.png', dpi=150)
