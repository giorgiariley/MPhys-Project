from astropy.table import Table
import pandas as pd
import numpy as np

cat = Table.read('/nvme/scratch/work/Griley/Masters/AGN/combined_morfometryka.fits')

PSF_FWHM = 4.55059

df = pd.DataFrame({
    'rootname': list(cat['# rootname9.65']),
    'RnFit2D':  np.array(cat['RnFit2D']).byteswap().newbyteorder(),
    'nFit2D':   np.array(cat['nFit2D']).byteswap().newbyteorder(),
    'qFit2D':   np.array(cat['qFit2D']).byteswap().newbyteorder(),
})

df['AGN_candidate'] = (
    (df['RnFit2D'] < PSF_FWHM) &
    (df['nFit2D']  > 4.0)      &
    (df['qFit2D']  > 0.9)
).map({True: 'yes', False: 'no'})

df.to_csv('/nvme/scratch/work/Griley/Masters/AGN/agn_morphology.csv', index=False)

print(df[df['AGN_candidate'] == 'yes'])
print(f"\nTotal objects: {len(df)}")
print(f"AGN candidates: {(df['AGN_candidate'] == 'yes').sum()}")