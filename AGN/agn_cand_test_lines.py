import pandas as pd
import numpy as np

single = pd.read_csv('/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/specFitMSA/data/project_mphys_ultrablue/catalog-flux_prism.csv')
broad  = pd.read_csv('/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/specFitMSA/data/project_mphys_ultrablue/catalog-flux_prism_broad.csv')

single_bic = single[['file', 'Index', 'Ha_6565_redchisq', 'Ha_6565_npar', 'Ha_6565_nobs']].copy()
broad_bic  = broad[['file', 'Ha_6565_Ha_6565_2_redchisq', 'Ha_6565_Ha_6565_2_npar', 
                     'Ha_6565_Ha_6565_2_nobs', 'Ha_6565_2_fwhm', 
                     'Ha_6565_2_fwhm_siglo', 'Ha_6565_2_fwhm_sigup']].copy()

df = single_bic.merge(broad_bic, on='file', how='left')

df['BIC_single'] = (df['Ha_6565_redchisq'] * 
                   (df['Ha_6565_nobs'] - df['Ha_6565_npar']) + 
                    df['Ha_6565_npar'] * np.log(df['Ha_6565_nobs']))

df['BIC_broad']  = (df['Ha_6565_Ha_6565_2_redchisq'] * 
                   (df['Ha_6565_Ha_6565_2_nobs'] - df['Ha_6565_Ha_6565_2_npar']) + 
                    df['Ha_6565_Ha_6565_2_npar'] * np.log(df['Ha_6565_Ha_6565_2_nobs']))

df['delta_BIC'] = df['BIC_broad'] - df['BIC_single']

df['broad_AGN_candidate'] = df['delta_BIC'] < -10

df.to_csv('/nvme/scratch/work/Griley/Masters/AGN/agn_lines_candidates.csv', index=False)

print(df[['file', 'Index', 'delta_BIC', 'Ha_6565_2_fwhm', 
          'Ha_6565_2_fwhm_siglo', 'Ha_6565_2_fwhm_sigup', 
          'broad_AGN_candidate']])
print(f"\nTotal objects: {len(df)}")
print(f"Broad line AGN candidates (delta_BIC < -10): {df['broad_AGN_candidate'].sum()}")