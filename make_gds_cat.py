#!/usr/bin/env python3
# This script takes the EPOCHS catalogue and outputs only the JADES GOODS-South entries
# by keeping rows where the 'file' column contains 'gds'.

import os
import numpy as np
from astropy.table import Table

EPOCHS_FILE = "/nvme/scratch/work/austind/EPOCHS-v2/tabs/spectra/EPOCHS-v2.fits"
OUTPUT_FILE = "/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/gdsgdn_catalogue.fits"

FILE_COL = "file"  # column to search for "gds"
PATTERN1 = "gds"    # substring to match (case-insensitive)
PATTERN2 = 'gdn'
PATTERN3 = 'goodsn'

def to_str_array(col):
    """
    Convert an Astropy column to an array of Python strings robustly,
    handling bytes and masked values.
    """
    arr = np.array(col)
    out = []
    for v in arr:
        if v is None:
            out.append("")
            continue
        if isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(str(v))
    return np.array(out, dtype=str)

def main():
    # Load table
    tab = Table.read(EPOCHS_FILE)
    print(f"Loaded {len(tab)} rows from {EPOCHS_FILE}")
    # print(f"Columns available: {tab.colnames}")

    if FILE_COL not in tab.colnames:
        raise KeyError(f"Column '{FILE_COL}' not found. Available columns: {tab.colnames}")

    file_strings = to_str_array(tab[FILE_COL])
    file_strings_lower = np.char.lower(file_strings)
    # Create individual masks for each pattern
    mask1 = np.char.find(file_strings_lower, PATTERN1.lower()) >= 0
    mask2 = np.char.find(file_strings_lower, PATTERN2.lower()) >= 0
    mask3 = np.char.find(file_strings_lower, PATTERN3.lower()) >= 0

    # Combine them: Keep row if mask1 OR mask2 OR mask3 is True
    mask = mask1 | mask2 | mask3
    sub = tab[mask]
    print(f"Keeping {len(sub)} / {len(tab)} rows where '{FILE_COL}' contains '{PATTERN1}' or '{PATTERN2}' or '{PATTERN3}'")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Write subset to FITS
    sub.write(OUTPUT_FILE, overwrite=True)
    print(f"Wrote: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
