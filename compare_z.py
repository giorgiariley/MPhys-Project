#this script is goign to compare zrf and zgrade by plotting them 
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

CAT_FILE = "/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/gdsgdn_catalogue.fits"

zrf = Table.read(CAT_FILE)['zrf']
zgrade = Table.read(CAT_FILE)['zgrade']
file = Table.read(CAT_FILE)['file']

plt.figure(figsize=(8, 8))
plt.scatter(zrf, zgrade, alpha=0.5, edgecolor='k')
plt.plot([0, 5], [0, 5], 'r--')
plt.xlabel('zrf')
plt.ylabel('zgrade')
plt.title('Comparison of zrf and zgrade')
plt.grid()
plt.savefig("/nvme/scratch/work/Griley/Masters/zrf_vs_zgrade.png")

#now want to print number of points plotted that were not NaN in either zrf or zgrade
mask = ~np.isnan(zrf) & ~np.isnan(zgrade)
print(f"Number of points plotted: {np.sum(mask)}")

#print how many rows have the word 'prism' in the file column
prism_mask = np.array(['prism' in f for f in file])
print(f"Number of rows with 'prism' in file column: {np.sum(prism_mask)}")

v4_mask = np.array(['v4' in f for f in file])
print(f"Number of rows with 'v4' in file column: {np.sum(v4_mask)}")

mask = prism_mask & v4_mask
print(f"Number of rows with 'prism' and 'v4' in file column: {np.sum(mask)}")