import csv, numpy as np
import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'
os.environ["GRIZLI_LOGFILE"] = "/nvme/scratch/work/Griley/Masters/grizli.log"  # any path you own
from galfind import Spectrum, Spectral_Catalogue, config
from grizli import utils
utils.LOGFILE = None

CSV = "mphys_GOODS_S_exposures.csv"
VERSION = "v4_2"   # try "v4_2" for current DJA; use "v3" if your setup expects that

specs = []
with open(CSV, newline="") as f:
    for row in csv.DictReader(f):
        root = row["root"].strip()
        fn   = row["file"].strip()
        url  = f"{config['Spectra']['DJA_WEB_DIR']}/{root}/{fn}"

        # optional: filter to grade 3 or to a specific grating
        if row.get("grade") and str(row["grade"]).strip() != "3":
            continue
        # if you want only PRISM rows:
        # if "PRISM" not in row.get("grating","").upper(): continue

        # optional: pass catalogue z if present
        z_val = None
        if row.get("z"):
            try:
                z_val = float(row["z"])
            except ValueError:
                z_val = None

        sp = Spectrum.from_DJA(url, save=True, version=VERSION, z=z_val,
                               root=root, file=fn)
        specs.append(sp)

cat = Spectral_Catalogue(np.array(specs))
cat.plot(src="msaexp")  # saves plots under your configured DJA_spec_plots folder
