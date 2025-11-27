import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib as mpl


# --- Global Matplotlib style for publication-quality plots ---
mpl.rcParams.update({
    "font.size": 20,             # Base font size
    "axes.titlesize": 20,        # Title size
    "axes.labelsize": 20,        # Axis label size
    "xtick.labelsize": 16,       # X tick label size
    "ytick.labelsize": 16,       # Y tick label size
    "legend.fontsize": 20,       # Legend text
    "figure.titlesize": 20,      # Overall figure title
    "axes.linewidth": 1.4,       # Thicker axes
    "xtick.major.width": 1.2,    # Tick line width
    "ytick.major.width": 1.2,
    "xtick.major.size": 6,       # Tick size
    "ytick.major.size": 6,
    "lines.linewidth": 1.6,      # Slightly thicker default lines
    "savefig.dpi": 300,          # High-resolution output for publication
})

# ----------------------------------------------------
# INPUT FILES
# ----------------------------------------------------

F560W_PATH = "/raid/scratch/data/jwst-miri/SMILES/60mas/hlsp_smiles_jwst_miri_goodss_F560W_v1.0_drz_aligned.fits"
TARGET_CSV = "/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/specFitMSA/data/project_mphys_ultrablue/HeII_Ha_high_SNR_allgratings.csv"
EXPOSURES_CSV = "/nvme/scratch/work/Griley/Masters/mphys_GOODS_S_exposures.csv"


# ============================================================
# MODULE 1 — LOAD MIRI MOSAIC + WCS + WEIGHT MAP
# ============================================================

def load_miri_mosaic(path):
    """
    Loads a JWST/MIRI mosaic and returns:
        - img (2D array)
        - wht (weight map)
        - wcs object
    """
    hdul = fits.open(path)
    img = hdul[1].data
    wht = hdul[4].data     # real coverage footprint
    wcs = WCS(hdul[1].header)
    hdul.close()
    return img, wht, wcs


# ============================================================
# MODULE 2 — LOAD TARGET LIST (39 IDs)
# ============================================================

def load_target_ids(csv_path):
    df = pd.read_csv(csv_path)
    return set(df["object_id"])


# ============================================================
# MODULE 3 — LOAD EXPOSURES & MATCH RA/DEC
# ============================================================

def extract_object_id(filename):
    try:
        return int(filename.split("_")[-1].split(".")[0])
    except:
        return None

def load_radec_for_targets(exposures_csv, target_ids):
    """
    Reads the exposures CSV, extracts object_id from filenames,
    filters for the 39 IDs, returns dataframe with (object_id, ra, dec).
    """
    df = pd.read_csv(exposures_csv)
    df["object_id"] = df["file"].apply(extract_object_id)
    df = df.dropna(subset=["object_id"])
    df_match = df[df["object_id"].isin(target_ids)][["object_id","ra","dec"]]
    return df_match


# ============================================================
# MODULE 4 — CLASSIFY POSITIONS USING WHT MAP
# ============================================================

def classify_positions(df_coords, wcs, wht):
    """
    Returns:
        - inside_x, inside_y (pixel positions of objects inside mosaic)
        - outside_x, outside_y (pixel positions outside mosaic)
        - outside_list   (list of full info tuples)

    Uses *weight map* to determine true MIRI coverage.
    """
    inside_x, inside_y = [], []
    outside_x, outside_y = [], []
    outside_list = []

    ny, nx = wht.shape

    for _, row in df_coords.iterrows():
        obj_id = int(row["object_id"])
        ra = float(row["ra"])
        dec = float(row["dec"])

        sky = SkyCoord(ra*u.deg, dec*u.deg)
        x, y = sky.to_pixel(wcs)

        inside = False

        # Inside array?
        if 0 <= x < nx and 0 <= y < ny:
            # Inside coverage? (wht > 0)
            if wht[int(y), int(x)] > 0:
                inside = True

        if inside:
            inside_x.append(x)
            inside_y.append(y)
        else:
            outside_x.append(x)
            outside_y.append(y)
            outside_list.append((obj_id, ra, dec, x, y))

    return inside_x, inside_y, outside_x, outside_y, outside_list


# ============================================================
# MODULE 5 — PLOT MIRI MOSAIC + SOURCES
# ============================================================

def plot_miri_with_sources(img, wcs, inside_x, inside_y, outside_x, outside_y, output_path):
    """
    Creates MIRI mosaic overlay plot with:
        - red circles = inside footprint
        - blue circles = outside footprint
    Automatically expands axes to include all points.
    """

    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(111, projection=wcs)

    # image display
    vmin = np.nanpercentile(img, 5)
    vmax = np.nanpercentile(img, 99)
    ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    # axis formatting
    ra = ax.coords[0]
    dec = ax.coords[1]
    ra.set_format_unit('deg')
    dec.set_format_unit('deg')
    ra.set_major_formatter('d.ddd')
    dec.set_major_formatter('d.ddd')
    ra.set_axislabel("Right Ascension (deg)")
    dec.set_axislabel("Declination (deg)")

    # plot points
    ax.scatter(inside_x, inside_y, s=40, facecolor="none", edgecolor="red",
               linewidth=1.8, label="Inside MIRI Mosaic")

    ax.scatter(outside_x, outside_y, s=40, facecolor="none", edgecolor="blue",
               linewidth=1.8, label="Outside Mosaic")

    # expand view
    pad = 200
    all_x = inside_x + outside_x + [0, img.shape[1]]
    all_y = inside_y + outside_y + [0, img.shape[0]]

    xmin = min(all_x) - pad
    xmax = max(all_x) + pad
    ymin = min(all_y) - pad
    ymax = max(all_y) + pad

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


# ============================================================
# MODULE 6 — PRINT OUTSIDE OBJECTS
# ============================================================

def print_outside_objects(outside_list):
    print("\n=== Objects OUTSIDE the true MIRI footprint ===")
    for obj_id, ra, dec, x, y in outside_list:
        print(f"object_id={obj_id:6d}  RA={ra:.6f}  Dec={dec:.6f}  pixel=({x:.1f}, {y:.1f})")
    print(f"Total outside = {len(outside_list)}")



# ============================================================
# MAIN DRIVER
# ============================================================

def main():

    # 1. Load image
    img, wht, wcs = load_miri_mosaic(F560W_PATH)

    # 2. Load the 39 IDs
    target_ids = load_target_ids(TARGET_CSV)
    print(f"Loaded {len(target_ids)} target IDs")

    # 3. Match RA/Dec
    df_coords = load_radec_for_targets(EXPOSURES_CSV, target_ids)
    print(f"Matched {len(df_coords)} sources")

    # 4. Classify inside/outside
    inside_x, inside_y, outside_x, outside_y, outside_list = classify_positions(
        df_coords, wcs, wht
    )

    # 5. Print outside sources
    print_outside_objects(outside_list)

    # 6. Plot everything
    plot_miri_with_sources(
        img, wcs,
        inside_x, inside_y,
        outside_x, outside_y,
        output_path="miri_f560w_overlay_modular.png"
    )


# ----------------------------------------------------
# RUN IF EXECUTED DIRECTLY
# ----------------------------------------------------

if __name__ == "__main__":
    main()
