import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

# ---------------------- CONFIGURATION ----------------------

# Input Files
exposures_csv_path = "/nvme/scratch/work/Griley/Masters/mphys_GOODS_S_exposures.csv"

# NEW: Dictionary of photometry FITS files to load, with their field names
photometry_catalogs = {
    "South": "/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-South/(0.32)as/JADES-DR3-GS-South_MASTER_Sel-F277W+F356W+F444W_v13.fits",
    "East": "/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits"
}

# List of CSV files containing the filenames to keep
filter_csv_files = [
"/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/specFitMSA/data/project_mphys_ultrablue/matched_exposures_prism.csv"]

# MODIFIED: Output path is now a base path
output_csv_base = "/nvme/scratch/work/Griley/Masters/exposure_photometry_matches_filtered"

# Matching Parameters
max_separation_arcsec = 1.0

# Column names
exp_ra_col = 'ra'
exp_dec_col = 'dec'
exp_file_col = 'file' 

phot_id_col = 'NUMBER'
phot_ra_col = 'ALPHA_J2000'
phot_dec_col = 'DELTA_J2000'

photometry_hdu_index = 1

# ---------------------- FUNCTIONS ----------------------

def load_valid_filenames(file_list):
    """
    Reads a list of CSV files and returns a set of unique filenames
    from the 'file' column of each file.
    """
    valid_files = set()
    print("Loading filter filenames from:")
    for f_path in file_list:
        try:
            df_filter = pd.read_csv(f_path)
            if 'file' in df_filter.columns:
                
                # --- NEW DIAGNOSTIC BLOCK ---
                all_filenames_list = df_filter['file'].astype(str).tolist()
                unique_filenames_set = set(all_filenames_list)
                
                num_total = len(all_filenames_list)
                num_unique = len(unique_filenames_set)

                print(f"  - File: {os.path.basename(f_path)}")
                print(f"    - Total rows/filenames found: {num_total}")
                print(f"    - Unique filenames found: {num_unique}")

                if num_total > num_unique:
                    # Find and print the duplicates
                    all_filenames_series = pd.Series(all_filenames_list)
                    duplicates = all_filenames_series[all_filenames_series.duplicated()].unique()
                    print(f"    - DUPLICATES FOUND ({num_total - num_unique} extra rows): {duplicates.tolist()}")
                # --- END DIAGNOSTIC BLOCK ---

                valid_files.update(unique_filenames_set)
                
            else:
                print(f"  - Warning: 'file' column not found in {os.path.basename(f_path)}")
        except FileNotFoundError:
            print(f"  - Warning: Filter file not found: {f_path}")
        except Exception as e:
            print(f"  - Warning: Error reading {f_path}: {e}")
            
    print(f"Total unique filenames loaded for filtering: {len(valid_files)}")
    return valid_files

# ---------------------- SCRIPT START ----------------------

# --- Load the set of filenames to keep ---
valid_filenames = load_valid_filenames(filter_csv_files)
if not valid_filenames:
    print("ERROR: No valid filenames loaded from filter CSVs. Cannot proceed.")
    exit()

# --- Load Exposures CSV ---
print(f"\nLoading exposures from: {exposures_csv_path}")
try:
    df_exposures = pd.read_csv(exposures_csv_path)
    df_exposures[exp_ra_col] = pd.to_numeric(df_exposures[exp_ra_col], errors='coerce')
    df_exposures[exp_dec_col] = pd.to_numeric(df_exposures[exp_dec_col], errors='coerce')
    df_exposures = df_exposures.dropna(subset=[exp_ra_col, exp_dec_col])
    print(f"Loaded {len(df_exposures)} exposures with valid coordinates.")
except FileNotFoundError:
    print(f"ERROR: Exposures CSV file not found at {exposures_csv_path}")
    exit()
except KeyError as e:
    print(f"ERROR: Missing expected column in exposures CSV: {e}")
    exit()

# --- Load and Combine Photometry FITS ---
print("Loading photometry catalogs...")
all_phot_dfs = []
required_phot_cols = [phot_id_col, phot_ra_col, phot_dec_col]

for field_name, phot_path in photometry_catalogs.items():
    try:
        print(f"  - Loading {field_name} from {os.path.basename(phot_path)}")
        with fits.open(phot_path) as hdul:
            if photometry_hdu_index >= len(hdul):
                 raise IndexError(f"HDU index {photometry_hdu_index} out of bounds.")
            if not isinstance(hdul[photometry_hdu_index], (fits.BinTableHDU, fits.TableHDU)):
                raise TypeError(f"HDU {photometry_hdu_index} is not a Table.")
            
            photometry_data = hdul[photometry_hdu_index].data
            df_phot_current = pd.DataFrame(photometry_data)
        
        if not all(col in df_phot_current.columns for col in required_phot_cols):
            raise KeyError(f"Photometry FITS table missing one of required columns.")
            
        df_phot_current[phot_ra_col] = pd.to_numeric(df_phot_current[phot_ra_col], errors='coerce')
        df_phot_current[phot_dec_col] = pd.to_numeric(df_phot_current[phot_dec_col], errors='coerce')
        df_phot_current = df_phot_current.dropna(subset=[phot_ra_col, phot_dec_col])
        
        # ADD THE NEW FIELD COLUMN
        df_phot_current['field'] = field_name
        
        all_phot_dfs.append(df_phot_current)
        print(f"    Loaded {len(df_phot_current)} {field_name} sources with valid coordinates.")

    except FileNotFoundError:
        print(f"    ERROR: Photometry FITS file not found at {phot_path}")
    except (IndexError, TypeError, KeyError) as e:
         print(f"    ERROR reading photometry FITS table from {phot_path}: {e}")

if not all_phot_dfs:
    print("ERROR: No photometry data was successfully loaded. Exiting.")
    exit()

# Combine the list of DataFrames into one
df_photometry = pd.concat(all_phot_dfs, ignore_index=True)
print(f"Total combined photometry sources loaded: {len(df_photometry)}")


# --- Create SkyCoord objects ---
print("\nCreating SkyCoord objects...")
coords_exposures = SkyCoord(ra=df_exposures[exp_ra_col].values*u.deg, 
                            dec=df_exposures[exp_dec_col].values*u.deg)
coords_photometry = SkyCoord(ra=df_photometry[phot_ra_col].values*u.deg, 
                             dec=df_photometry[phot_dec_col].values*u.deg)

# --- Perform the cross-match ---
print(f"Performing cross-match (max separation = {max_separation_arcsec} arcsec)...")
idx_phot, sep2d, _ = coords_exposures.match_to_catalog_sky(coords_photometry)

# --- Filter matches based on separation ---
max_sep = max_separation_arcsec * u.arcsec
match_mask = sep2d <= max_sep
n_matches = np.sum(match_mask)
print(f"Found {n_matches} coordinate matches within {max_separation_arcsec} arcsec.")

# --- Create the initial output DataFrame (including all coordinate matches) ---
output_data = []
matched_exp_indices = np.where(match_mask)[0]

for exp_idx in matched_exp_indices:
    # Get the index of the corresponding match in the photometry catalog
    phot_idx = idx_phot[exp_idx]
    
    output_data.append({
        'file': df_exposures.iloc[exp_idx][exp_file_col], 
        'ra': df_exposures.iloc[exp_idx][exp_ra_col],
        'dec': df_exposures.iloc[exp_idx][exp_dec_col],
        'photometry_NUMBER': df_photometry.iloc[phot_idx][phot_id_col],
        'field': df_photometry.iloc[phot_idx]['field'], # ADD THE NEW FIELD
        'separation_arcsec': sep2d[exp_idx].to(u.arcsec).value
    })

df_output_unfiltered = pd.DataFrame(output_data)

# --- Apply the filename filter ---
print(f"\nFiltering {len(df_output_unfiltered)} coordinate matches based on filename lists...")
# Keep rows where the 'file' column value exists in the set loaded from filter CSVs
df_output_filtered = df_output_unfiltered[df_output_unfiltered['file'].isin(valid_filenames)].copy()

n_filtered_matches = len(df_output_filtered)
print(f"Kept {n_filtered_matches} matches after applying filename filter.")

# --- MODIFIED: Save the results into separate files ---
if n_filtered_matches > 0:
    
    # Split the final filtered DataFrame by the 'field' column
    df_south = df_output_filtered[df_output_filtered['field'] == 'South']
    df_east = df_output_filtered[df_output_filtered['field'] == 'East']

    # Define output paths
    output_path_south = output_csv_base + "_South.csv"
    output_path_east = output_csv_base + "_East.csv"

    # Save South file
    if len(df_south) > 0:
        print(f"\nSaving {len(df_south)} filtered matches for South to: {output_path_south}")
        df_south.to_csv(output_path_south, index=False)
    else:
        print("\nNo 'South' matches found after filtering.")

    # Save East file
    if len(df_east) > 0:
        print(f"\nSaving {len(df_east)} filtered matches for East to: {output_path_east}")
        df_east.to_csv(output_path_east, index=False)
    else:
        print("\nNo 'East' matches found after filtering.")

else:
    print("\nNo matches found after applying filename filter. No output files saved.")

print("\nCross-matching complete.")

# Load filter csv
df_filter = pd.read_csv(filter_csv_files[0])

# IDs in the 'Index' column that you want to investigate
spec_ids = [180835, 202208, 289178]

# Step A: Extract file names for those Index values
missing_rows = df_filter[df_filter['Index'].isin(spec_ids)]

print("\nRows matching Index values:")
print(missing_rows)

missing_files = missing_rows['file'].astype(str).tolist()
print("\nFilenames for missing Index values:")
print(missing_files)

# Step B: Did they match photometry by coordinates?
print("\nCoordinate matches (before filename filtering):")
print(df_output_unfiltered[df_output_unfiltered['file'].isin(missing_files)])

# Step C: Were they removed by filename filtering?
print("\nRemoved by filename filter:")
print(
    df_output_unfiltered[
        df_output_unfiltered['file'].isin(missing_files)
        & ~df_output_unfiltered['file'].isin(valid_filenames)
    ]
)

# Step D: Field information for matched ones
print("\nField assignments (if matched):")
print(
    df_output_unfiltered[
        df_output_unfiltered['file'].isin(missing_files)
    ][['file','photometry_NUMBER','field','separation_arcsec']]
)

print("\n==============================")
print("Nearest photometry sources for missing spectroscopy IDs")
print("==============================\n")

# These are the spectroscopy Index values you care about
spec_ids = [180835, 202208, 289178]

# Get their rows from the filter CSV
missing_rows = df_filter[df_filter['Index'].isin(spec_ids)]

if missing_rows.empty:
    print("ERROR: Could not find any rows for the requested Index values!")
else:
    for _, row in missing_rows.iterrows():
        spec_index = row['Index']
        spec_file  = row['file']
        spec_ra    = float(row['ra'])
        spec_dec   = float(row['dec'])

        # Build SkyCoord for the spectroscopy position
        spec_coord = SkyCoord(ra=spec_ra * u.deg, dec=spec_dec * u.deg)

        # Find nearest photometry source
        sep = spec_coord.separation(coords_photometry)
        min_idx = np.argmin(sep)
        min_sep_arcsec = sep[min_idx].arcsec

        nearest_phot = df_photometry.iloc[min_idx]
        phot_ra = nearest_phot[phot_ra_col]
        phot_dec = nearest_phot[phot_dec_col]
        phot_num = nearest_phot[phot_id_col]
        phot_field = nearest_phot['field']

        print(f"--- Spectroscopy ID {spec_index} ---")
        print(f"Spectrum file: {spec_file}")
        print(f"Spectrum RA,Dec:   {spec_ra:.6f}, {spec_dec:.6f}")
        print(f"Nearest phot RA,Dec: {phot_ra:.6f}, {phot_dec:.6f}")
        print(f"Photometry NUMBER:   {phot_num}")
        print(f"Photometry field:    {phot_field}")
        print(f"Minimum separation:  {min_sep_arcsec:.3f} arcsec")

        if min_sep_arcsec <= max_separation_arcsec:
            print("→ This SHOULD have matched photometry (within 1 arcsec).")
        else:
            print("→ This is OUTSIDE matching radius (no photometry match).")

        print("")
