import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

# ----------------------------------------------------
# INPUT FILES
# ----------------------------------------------------

F560W_PATH = "/raid/scratch/data/jwst-miri/SMILES/60mas/hlsp_smiles_jwst_miri_goodss_F560W_v1.0_drz_aligned.fits"
TARGET_CSV = "/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/specFitMSA/data/project_mphys_ultrablue/HeII_Ha_high_SNR_allgratings.csv"
EXPOSURES_CSV = "/nvme/scratch/work/Griley/Masters/mphys_GOODS_S_exposures.csv"

# ----------------------------------------------------
# 1. LOAD MIRI IMAGE + WCS
# ----------------------------------------------------

hdul = fits.open(F560W_PATH)
img = hdul[1].data
w = WCS(hdul[1].header)
hdul.close()

# ----------------------------------------------------
# 2. LOAD TARGET OBJECT IDs (these are the 39)
# ----------------------------------------------------

df_targets = pd.read_csv(TARGET_CSV)
target_ids = set(df_targets["object_id"])
print(f"Loaded {len(target_ids)} object IDs (should be 39).")

# ----------------------------------------------------
# 3. LOAD EXPOSURES CSV AND MATCH RA/DEC
# ----------------------------------------------------

def extract_object_id(filename):
    try:
        return int(filename.split("_")[-1].split(".")[0])
    except:
        return None

df_exp = pd.read_csv(EXPOSURES_CSV)
df_exp["object_id"] = df_exp["file"].apply(extract_object_id)
df_exp = df_exp.dropna(subset=["object_id"])

# Keep only the 39 targets
df_matched = df_exp[df_exp["object_id"].isin(target_ids)][["object_id","ra","dec"]]

print(f"Matched {len(df_matched)} RA/Dec positions.")

# ----------------------------------------------------
# 4. PLOT MIRI IMAGE
# ----------------------------------------------------

fig = plt.figure(figsize=(10,10))
ax = plt.subplot(111, projection=w)

vmin = np.nanpercentile(img, 5)
vmax = np.nanpercentile(img, 99)
ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)



# ----------------------------------------------------
# 5. FORMAT AXES IN DECIMAL DEGREES
# ----------------------------------------------------

ra = ax.coords[0]
dec = ax.coords[1]
ra.set_format_unit('deg')
dec.set_format_unit('deg')
ra.set_major_formatter('d.ddd')
dec.set_major_formatter('d.ddd')
ra.set_axislabel("Right Ascension (deg)")
dec.set_axislabel("Declination (deg)")

# ----------------------------------------------------
# 6. OVERLAY THE 39 OBJECTS
# ----------------------------------------------------

xs = []
ys = []

for _, row in df_matched.iterrows():
    sky = SkyCoord(row["ra"] * u.deg, row["dec"] * u.deg)
    x, y = sky.to_pixel(w)

    # Only plot those inside the mosaic
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        xs.append(x)
        ys.append(y)

# Determine which objects fall outside the mosaic
outside_rows = []

ny, nx = img.shape  # img is F560W mosaic

outside = []  # store unique objects

# ----------------------------------------------------
# 6b. DETERMINE WHICH TARGETS ARE OUTSIDE THE MOSAIC
# ----------------------------------------------------

df_coords = df_matched   # <-- the correct dataframe of your 39 RA/Dec values

outside = []

for _, row in df_coords.iterrows():
    obj_id = int(row["object_id"])
    ra = float(row["ra"])
    dec = float(row["dec"])

    x, y = SkyCoord(ra*u.deg, dec*u.deg).to_pixel(w)

    # check if outside image bounds
    if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
        outside.append((obj_id, ra, dec, x, y))

# Remove duplicates just in case
outside_unique = { obj[0]: obj for obj in outside }.values()

print("\n=== UNIQUE Objects OUTSIDE the MIRI F560W mosaic ===")
for obj_id, ra, dec, x, y in outside_unique:
    print(f"object_id={obj_id:6d}   RA={ra:.6f}   Dec={dec:.6f}   pixel=({x:.1f}, {y:.1f})")

# ----------------------------------------------------
# 6. OVERLAY THE 39 OBJECTS
# ----------------------------------------------------

# Lists for on-image points
xs = []
ys = []

# Also keep pixel coords for *all* 39 (even if off-image)
xs_all = []
ys_all = []

for _, row in df_matched.iterrows():
    sky = SkyCoord(row["ra"] * u.deg, row["dec"] * u.deg)
    x, y = sky.to_pixel(w)

    xs_all.append(x)
    ys_all.append(y)

    # Only plot those inside the mosaic
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        xs.append(x)
        ys.append(y)


# ----------------------------------------------------
# 7. FIND OBJECTS OUTSIDE THE MOSAIC AND PRINT THEM
# ----------------------------------------------------

ny, nx = img.shape
outside = []

for obj_id, ra, dec in df_matched[["object_id", "ra", "dec"]].values:
    sky = SkyCoord(ra * u.deg, dec * u.deg)
    x, y = sky.to_pixel(w)

    if not (0 <= x < nx and 0 <= y < ny):
        outside.append((int(obj_id), ra, dec, x, y))

# Remove duplicates by object_id
outside_unique = {obj[0]: obj for obj in outside}.values()

print("\n=== UNIQUE Objects OUTSIDE the MIRI F560W mosaic ===")
for obj_id, ra, dec, x, y in outside_unique:
    print(f"object_id={obj_id:6d}   RA={ra:.6f}   Dec={dec:.6f}   pixel=({x:.1f}, {y:.1f})")

# ----------------------------------------------------
# 8. EXPAND AXES IN *PIXEL* SPACE TO SHOW EVERYTHING
#    (include image corners so the whole mosaic stays visible)
# ----------------------------------------------------

pad = 50  # pixels of padding

# Image corners in pixel coords
corner_x = [0, nx]
corner_y = [0, ny]

# Include:
#  - all 39 galaxies (xs_all, ys_all)
#  - image corners
all_x = corner_x + xs_all
all_y = corner_y + ys_all

xmin = min(all_x) - pad
xmax = max(all_x) + pad
ymin = min(all_y) - pad
ymax = max(all_y) + pad

# How much to expand the view (e.g., 1.05 = expand by 5 percent)
expand_factor = 1.05  

# Compute plot centre
xc = (xmin + xmax) / 2
yc = (ymin + ymax) / 2

# Apply multiplicative expansion
xhalf = (xmax - xmin) * expand_factor / 2
yhalf = (ymax - ymin) * expand_factor / 2

xmin_expanded = xc - xhalf
xmax_expanded = xc + xhalf
ymin_expanded = yc - yhalf
ymax_expanded = yc + yhalf

# Apply limits
ax.set_xlim(xmin_expanded, xmax_expanded)
ax.set_ylim(ymin_expanded, ymax_expanded)







xs_in, ys_in = [], []
xs_out, ys_out = [], []

for _, row in df_matched.iterrows():
    ra = float(row["ra"])
    dec = float(row["dec"])
    sky = SkyCoord(ra * u.deg, dec * u.deg)

    x, y = sky.to_pixel(w)

    # inside the mosaic?
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        xs_in.append(x)
        ys_in.append(y)
    else:
        xs_out.append(x)
        ys_out.append(y)

# -----------------------
# PLOT THEM
# -----------------------

# red = inside MIRI
ax.scatter(xs_in, ys_in,
           s=40, facecolor="none", edgecolor="red", linewidth=1.8,
           label="Inside MIRI Mosaic")

# blue = outside MIRI
ax.scatter(xs_out, ys_out,
           s=40, facecolor="none", edgecolor="blue", linewidth=1.8,
           label="Outside Mosaic")


plt.tight_layout()
plt.savefig("miri_f560w_overlay_39.png", dpi=300)
plt.show()
