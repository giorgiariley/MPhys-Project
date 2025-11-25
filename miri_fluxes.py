import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------

F560W_PATH = "/raid/scratch/data/jwst-miri/SMILES/60mas/hlsp_smiles_jwst_miri_goodss_F560W_v1.0_drz_aligned.fits"
F770W_PATH = "/raid/scratch/data/jwst-miri/SMILES/60mas/hlsp_smiles_jwst_miri_goodss_F770W_v1.0_drz_aligned.fits"
TARGET_CSV = "/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/specFitMSA/data/project_mphys_ultrablue/HeII_Ha_high_SNR_allgratings.csv"
EXPOSURES_CSV = "/nvme/scratch/work/Griley/Masters/mphys_GOODS_S_exposures.csv"
OUTPUT_CSV = "/nvme/scratch/work/Griley/Masters/F770W_fluxes.csv"

# aperture radius in arcsec
AP_RADIUS_ARCSEC = 0.3   # good default for MIRI 60 mas mosaics

# background annulus for sky subtraction
ANN_INNER_ARCSEC = 0.4
ANN_OUTER_ARCSEC = 0.6


# ----------------------------------------------------
# HELPERS AND MODULAR FUNCTIONS
# ----------------------------------------------------

def load_image_and_wcs(path):
    """
    Load image and WCS from a MIRI mosaic.
    Returns: img (2D array), wcs, pixel scale in arcsec per pixel.
    """
    print(f"Loading image from {path} ...")
    hdul = fits.open(path)
    img = hdul[1].data
    w = WCS(hdul[1].header)  # SCI extension WCS
    hdul.close()

    pixscale = np.abs(w.pixel_scale_matrix[0, 0]) * 3600.0  # arcsec per pixel
    print(f"Pixel scale = {pixscale:.4f} arcsec/pixel")
    return img, w, pixscale


def load_target_ids(target_csv):
    """
    Load the set of target object_ids from the target CSV.
    """
    df_targets = pd.read_csv(target_csv)
    target_ids = set(df_targets["object_id"])
    print(f"Loaded {len(target_ids)} target object IDs from {target_csv}")
    return target_ids


def extract_id_from_filename(filename):
    """
    Extract integer object ID from filenames like:
    'gds-udeep-v4_g140m-f070lp_3215_101062.spec.fits'
    """
    try:
        return int(filename.split("_")[-1].split(".")[0])
    except Exception:
        return None


def load_exposures_with_radec(exposures_csv, target_ids):
    """
    Load exposures CSV, extract object_id from file column,
    and return a DataFrame with object_id, ra, dec for targets.
    """
    df_exp = pd.read_csv(exposures_csv)

    # Extract ID from filenames
    df_exp["object_id"] = df_exp["file"].apply(extract_id_from_filename)
    df_exp = df_exp.dropna(subset=["object_id"])

    # ---- NEW: group so each object appears once ----
    df_exp_grouped = (
        df_exp
        .sort_values("file")
        .groupby("object_id")
        .first()
        .reset_index()
    )

    # Now merge precisely
    df_merged = df_exp_grouped[df_exp_grouped["object_id"].isin(target_ids)][["object_id", "ra", "dec"]]

    print(f"Matched {len(df_merged)} objects (expected ~39).")

    return df_merged


def measure_flux_for_object(obj_id, ra, dec, img, wcs, ap_radius_pix, ann_inner_pix, ann_outer_pix):
    """
    Measure background-subtracted flux at RA/Dec.
    Returns flux (float) or None if object is invalid / outside image.
    """

    # Convert RA/Dec → pixel coordinates
    skycoord = SkyCoord(ra * u.deg, dec * u.deg)
    x, y = skycoord.to_pixel(wcs)

    # ------------------------------------------------
    # DIAGNOSTICS FOR MISSING FLUX
    # ------------------------------------------------

    # 1. RA/Dec → pixel conversion failed
    if not np.isfinite(x) or not np.isfinite(y):
        print(f"[SKIP] {obj_id}: RA/Dec mapped to NaN pixel coords")
        return None

    ny, nx = img.shape

    # 2. Outside image footprint
    if not (0 <= x < nx and 0 <= y < ny):
        print(f"[OUT OF BOUNDS] {obj_id}: pixel=({x:.1f}, {y:.1f}) lies outside {nx}×{ny} image")
        return None

    # 3. Too close to edge
    if not (ap_radius_pix < x < nx - ap_radius_pix and ap_radius_pix < y < ny - ap_radius_pix):
        print(f"[EDGE] {obj_id}: aperture would fall off image boundary")
        return None

    # 4. NaN central pixel
    if np.isnan(img[int(y), int(x)]):
        print(f"[NAN PIXEL] {obj_id}: central pixel value is NaN")
        return None

    # ------------------------------------------------
    # APERTURE PHOTOMETRY
    # ------------------------------------------------
    aperture = CircularAperture((x, y), r=ap_radius_pix)
    annulus = CircularAnnulus((x, y), r_in=ann_inner_pix, r_out=ann_outer_pix)

    aper_phot = aperture_photometry(img, aperture)
    ann_phot = aperture_photometry(img, annulus)

    aper_flux = aper_phot["aperture_sum"][0]
    ann_flux = ann_phot["aperture_sum"][0]

    # background subtraction
    ann_area = annulus.area
    aper_area = aperture.area
    bkg_per_pix = ann_flux / ann_area
    bkg_total = bkg_per_pix * aper_area

    flux = aper_flux - bkg_total
    return flux



def run_aperture_photometry(
    image_path,
    target_csv,
    exposures_csv,
    output_csv,
    ap_radius_arcsec=0.3,
    ann_inner_arcsec=0.4,
    ann_outer_arcsec=0.6,
    flux_column_name="flux_f770w",
):
    """
    High level driver that:
      - loads image and WCS
      - loads targets and exposures, matches RA/Dec
      - runs aperture photometry for each object
      - writes out a CSV with fluxes
    """
    # Load image and WCS
    img, wcs, pixscale = load_image_and_wcs(image_path)

    # Convert radii to pixels
    ap_radius_pix = ap_radius_arcsec / pixscale
    ann_inner_pix = ann_inner_arcsec / pixscale
    ann_outer_pix = ann_outer_arcsec / pixscale

    # Load target IDs and RA/Dec
    target_ids = load_target_ids(target_csv)
    df_radec = load_exposures_with_radec(exposures_csv, target_ids)

    # Loop over objects
    print("\nBeginning aperture photometry...\n")
    rows = []

    for _, row in df_radec.iterrows():
        obj_id = row["object_id"]
        ra = row["ra"]
        dec = row["dec"]

        flux = measure_flux_for_object(
            obj_id=obj_id,
            ra=ra,
            dec=dec,
            img=img,
            wcs=wcs,
            ap_radius_pix=ap_radius_pix,
            ann_inner_pix=ann_inner_pix,
            ann_outer_pix=ann_outer_pix,
        )


        rows.append({
            "object_id": obj_id,
            "ra": ra,
            "dec": dec,
            flux_column_name: flux,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)

    print("\nFinished aperture photometry.")
    print(f"Saved results to: {output_csv}")
    return df_out


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

if __name__ == "__main__":
    run_aperture_photometry(
        image_path=F770W_PATH,
        target_csv=TARGET_CSV,
        exposures_csv=EXPOSURES_CSV,
        output_csv=OUTPUT_CSV,
        ap_radius_arcsec=AP_RADIUS_ARCSEC,
        ann_inner_arcsec=ANN_INNER_ARCSEC,
        ann_outer_arcsec=ANN_OUTER_ARCSEC,
        flux_column_name="flux_f770w",
    )
