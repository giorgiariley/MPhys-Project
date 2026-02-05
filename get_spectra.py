import csv, numpy as np
import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'
os.environ["GRIZLI_LOGFILE"] = "/nvme/scratch/work/Griley/Masters/grizli.log"  # any path you own
from galfind import Spectrum, Spectral_Catalogue, config
from grizli import utils
utils.LOGFILE = None
from astropy.table import Table



# CSV = "mphys_GOODS_S_exposures.csv"
FITS_PATH = "/nvme/scratch/work/austind/EPOCHS-v2/tabs/spectra/EPOCHS-v2.fits"
VERSION = "v4_2"   # try "v4_2" for current DJA; use "v3" if your setup expects that

# --- Load FITS table ---
# If this errors, the table might be in a different HDU. See note below.
tab = Table.read(FITS_PATH)

print(f"Loaded {len(tab)} rows from {FITS_PATH}")
print("Columns:", tab.colnames)

# --- Helper to safely pull string-ish values ---
def get_str(row, key):
    if key not in row.colnames:
        return None
    val = row[key]
    if val is None:
        return None
    # astropy sometimes gives bytes
    if isinstance(val, (bytes, np.bytes_)):
        val = val.decode("utf-8", errors="ignore")
    return str(val).strip()

def get_float(row, key):
    if key not in row.colnames:
        return None
    try:
        v = row[key]
        if v is None:
            return None
        return float(v)
    except Exception:
        return None

specs = []
# with open(CSV, newline="") as f:
    # for row in csv.DictReader(f):
for row in tab:
        root = get_str(row, "root")
        fn   = get_str(row, "file")
        url  = f"{config['Spectra']['DJA_WEB_DIR']}/{root}/{fn}"
        grating = get_str(row, "grating")


        z_val = get_float(row, "z")


        sp = Spectrum.from_DJA(url, save=True, version=VERSION, z=z_val,
                               root=root, file=fn)
        specs.append(sp)
print(f"Keeping {len(specs)} spectra after filtering.")


cat = Spectral_Catalogue(np.array(specs))
cat.plot(src="msaexp")  # saves plots under your configured DJA_spec_plots folder
