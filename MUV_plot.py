import csv
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional, List, Dict, Any, Set
import pandas as pd
import astropy.units as au
from astropy.cosmology import Planck18
from scipy.integrate import simpson
from scipy.optimize import curve_fit

# --- CONSTANTS ---
import matplotlib as mpl
mpl.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18,
    "figure.titlesize": 20,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "lines.linewidth": 1.6,
    "savefig.dpi": 300,
    # Ensure mathtext is used for labels
    "text.usetex": False, 
    "mathtext.fontset": "cm" 
})

C_LIGHT_AA_PER_S = 2.99792458e18
AB_MAG_ZP_JY = 8.90
W_UV_MIN = 1350.0
W_UV_MAX = 1800.0
cosmo = Planck18

LOWER_C94_FILT = np.array([1268., 1309., 1342., 1407., 1562., 1677., 1760., 1866., 1930., 2400.])
UPPER_C94_FILT = np.array([1284., 1316., 1371., 1515., 1583., 1740., 1833., 1890., 1950., 2580.])

SPECTRA_BASE_DIR = "/raid/scratch/work/Griley/GALFIND_WORK/Spectra/2D"
CSV_PATH_GLOBAL = Path("./mphys_GOODS_S_exposures.csv")
OUTPUT_DIR = Path.cwd() / "MUV_plots_40"
EXTERNAL_SNR_CSV_PATH = Path("/nvme/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/uv_snr5plus_with_prism_and_medium.csv")
TARGET_CSV = Path("/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/specFitMSA/data/project_mphys_ultrablue/HeII_Ha_high_SNR_allgratings.csv")

# Defined Photometry Catalogues (South and East)
PHOTOMETRY_CAT_SOUTH = Path("/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-South/(0.32)as/JADES-DR3-GS-South_MASTER_Sel-F277W+F356W+F444W_v13.fits")
PHOTOMETRY_CAT_EAST = Path("/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits")

# -----------------------------------------------------------
# Load the 39 target IDs
# -----------------------------------------------------------
def load_target_object_ids(csv_path: Path) -> Set[int]:
    target_ids = set()
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("object_id"):
                continue
            first = line.split(",", 1)[0].strip()
            try:
                target_ids.add(int(first))
            except:
                continue
    print(f"Loaded {len(target_ids)} target IDs.")
    return target_ids

def get_redshift_from_csv(csv_file: Path, fits_file: Path) -> Optional[float]:
    """Look up the redshift 'z' for a given FITS filename."""
    target = fits_file.name.strip()
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("file", "").strip() == target:
                try:
                    return float(row.get("z", "").strip())
                except (TypeError, ValueError):
                    return None
    return None


# -----------------------------------------------------------
# Load Photometric Background (FIXED ERROR LOGIC)
# -----------------------------------------------------------
def load_photometric_background(cat_paths: List[Path]) -> pd.DataFrame:
    """
    Loads M_UV and Beta. 
    Calculates M_UV errors from percentiles (distance from median).
    Uses Beta l1/u1 directly as errors.
    """
    data_frames = []

    for path in cat_paths:
        if not path.exists():
            print(f"Warning: Photometry file not found: {path}")
            continue
        
        print(f"Loading photometry from: {path.name}")
        with fits.open(path) as hdul:
            # --- Extract M_UV (HDU 4) ---
            try:
                data_muv = hdul[4].data
                muv_50 = data_muv['M_UV_50']
                muv_16 = data_muv['M_UV_16']
                muv_84 = data_muv['M_UV_84']
            except KeyError as e:
                print(f"KeyError in HDU 4 for {path.name}: {e}")
                continue

            # --- Extract Beta (HDU 6) ---
            try:
                data_beta = hdul[6].data
                beta_col_name = "beta_[1250,3000]AA_0.32as"
                
                beta_50 = data_beta[beta_col_name]
                # Assuming these are already ERROR DISTANCES based on user input
                beta_l1 = data_beta[f"{beta_col_name}_l1"]
                beta_u1 = data_beta[f"{beta_col_name}_u1"]
            except KeyError as e:
                 print(f"KeyError in HDU 6 for {path.name}: {e}")
                 continue

            df_temp = pd.DataFrame({
                'muv': muv_50,
                'muv_16': muv_16,
                'muv_84': muv_84,
                'beta': beta_50,
                'beta_l1': beta_l1,
                'beta_u1': beta_u1
            })
            
            df_temp = df_temp.replace([np.inf, -np.inf], np.nan).dropna()
            data_frames.append(df_temp)

    if not data_frames:
        return pd.DataFrame()

    combined_df = pd.concat(data_frames, ignore_index=True)
    
    # --- Calculate Errors ---
    
    # M_UV: Still assuming these are percentiles (absolute values)
    combined_df['muv_err_low'] = np.abs(combined_df['muv'] - combined_df['muv_16'])
    combined_df['muv_err_high'] = np.abs(combined_df['muv_84'] - combined_df['muv'])
    
    # Beta: User confirmed these are uncertainties (approx 0.5), use DIRECTLY.
    combined_df['beta_err_low'] = combined_df['beta_l1']
    combined_df['beta_err_high'] = combined_df['beta_u1']

    # --- FILTERING ---
    print(f"Photometry size before cleaning: {len(combined_df)}")
    
    # 1. Physical limits
    combined_df = combined_df[(combined_df['beta'] > -5.0) & (combined_df['beta'] < 5.0)]
    
    # 2. Error size cuts (Remove unconstrained fits)
    # We only plot points with reasonable error bars to match your clean plot
    error_threshold = 1.0
    mask_good_errors = (
        (combined_df['beta_err_low'] < error_threshold) & 
        (combined_df['beta_err_high'] < error_threshold) &
        (combined_df['muv_err_low'] < 1.0) & 
        (combined_df['muv_err_high'] < 1.0)
    )
    combined_df = combined_df[mask_good_errors]
    
    print(f"Photometry size after cleaning: {len(combined_df)}")

    return combined_df

# -----------------------------------------------------------
# Load SNR map
# -----------------------------------------------------------
def load_snr_values(path: Path):
    df = pd.read_csv(path, usecols=["prism_file", "avg_snr_uv"])
    return df.set_index("prism_file")["avg_snr_uv"].to_dict()


# -----------------------------------------------------------
# Find prism FITS files
# -----------------------------------------------------------
def find_prism_fits(base_dir):
    all_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".fits") and "prism" in f:
                all_files.append(Path(root) / f)
    return all_files


# -----------------------------------------------------------
# Read spectrum
# -----------------------------------------------------------
def read_observed_spectrum(fpath):
    with fits.open(fpath) as hdul:
        data = hdul[1].data
        wave = np.array(data["wave"])
        flux = np.array(data["flux"])
        err = np.array(data["err"] if "err" in data.columns.names else data["full_err"])
    return wave, flux, err


# -----------------------------------------------------------
# Convert to rest-frame
# -----------------------------------------------------------
def get_rest_frame_spectrum(wave_obs_um, flux_obs_uJy, err_obs_uJy, z):
    lam_obs_AA = wave_obs_um * 1e4
    lam_rest = lam_obs_AA / (1+z)

    factor = 1e-29 * C_LIGHT_AA_PER_S
    flam = flux_obs_uJy * factor / (lam_obs_AA**2)
    elam = err_obs_uJy * factor / (lam_obs_AA**2)

    idx = np.argsort(lam_rest)
    lam_rest, flam, elam = lam_rest[idx], flam[idx], elam[idx]

    keep = np.isfinite(lam_rest) & np.isfinite(flam) & np.isfinite(elam) & (elam >= 0)
    return lam_rest[keep], flam[keep], elam[keep]


# -----------------------------------------------------------
# Compute MUV
# -----------------------------------------------------------
def calculate_integral_error(w, e):
    if len(w) < 2:
        return np.nan
    dw = np.diff(w, prepend=w[0], append=w[-1])
    dw_i = (dw[:-1] + dw[1:]) / 2.0
    return np.sqrt(np.sum((e * dw_i)**2))


def calculate_muv_and_error(w_rest, f_rest, e_rest, z):
    mask = (w_rest >= W_UV_MIN) & (w_rest <= W_UV_MAX)
    if not mask.any():
        return None, None
    w, f, e = w_rest[mask], f_rest[mask], e_rest[mask]

    ok = np.isfinite(f) & np.isfinite(e) & (e >= 0)
    w, f, e = w[ok], f[ok], e[ok]

    if len(w) < 2:
        return None, None

    integral = simpson(f, x=w)
    Flam_mean = integral / (W_UV_MAX - W_UV_MIN)
    if Flam_mean <= 0:
        return None, None

    lam_eff = 0.5 * (W_UV_MIN + W_UV_MAX)
    F_nu = Flam_mean * (lam_eff**2 / C_LIGHT_AA_PER_S)
    F_Jy = F_nu / 1e-23
    mUV = -2.5*np.log10(F_Jy) + AB_MAG_ZP_JY

    dL = cosmo.luminosity_distance(z).to(au.pc).value
    MUV = mUV - 5*(np.log10(dL) - 1) - 2.5*np.log10(1+z)

    # error
    integral_err = calculate_integral_error(w, e)
    Flam_err = integral_err / (W_UV_MAX - W_UV_MIN)
    delta_M = (2.5/np.log(10)) * (Flam_err/Flam_mean)

    return MUV, delta_M


# -----------------------------------------------------------
# Compute Beta
# -----------------------------------------------------------
def sample_spectrum_C94(w, f, e):
    waves, fluxes, errs = [], [], []
    for wmin, wmax in zip(LOWER_C94_FILT, UPPER_C94_FILT):
        mask = (w>=wmin)&(w<=wmax)
        if not mask.any():
            continue
        fw, ew = f[mask], e[mask]
        ok = (fw>0)&np.isfinite(fw)&np.isfinite(ew)
        if ok.sum() < 2:
            continue
        median_flux = np.median(fw[ok])
        median_err = np.median(ew[ok]) / np.sqrt(ok.sum())
        waves.append((wmin+wmax)/2)
        fluxes.append(median_flux)
        errs.append(median_err)
    return np.array(waves), np.array(fluxes), np.array(errs)


def calculate_beta_and_error(w, f, e):
    waves, fluxes, errs = sample_spectrum_C94(w, f, e)
    if len(waves) < 2:
        return None, None
    ok = (fluxes>0)&(errs>0)
    if ok.sum() < 2:
        return None, None
    waves, fluxes, errs = waves[ok], fluxes[ok], errs[ok]
    log_err = (2.5 / np.log(10)) * (errs/fluxes)
    try:
        popt, pcov = curve_fit(
            lambda lw, a, b: a + b*lw,
            np.log10(waves),
            np.log10(fluxes),
            sigma=log_err,
            absolute_sigma=True,
            p0=[0,-2]
        )
        beta = popt[1]
        beta_err = np.sqrt(pcov[1,1])
        return beta, beta_err
    except:
        return None, None


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def process_and_plot():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_ids = load_target_object_ids(TARGET_CSV)
    snr_map = load_snr_values(EXTERNAL_SNR_CSV_PATH)
    fits_files = find_prism_fits(SPECTRA_BASE_DIR)

    # --- Load Photometric Background Data ---
    print("Loading photometric background data...")
    df_photo = load_photometric_background([PHOTOMETRY_CAT_SOUTH, PHOTOMETRY_CAT_EAST])

    # --- Process Spectroscopic Data ---
    results = []
    for fpath in fits_files:
        fname = fpath.name
        try:
            object_id = int(fname.split("_")[-1].split(".")[0])
        except:
            continue

        if object_id not in target_ids:
            continue
        if fname not in snr_map:
            continue
        

        z = get_redshift_from_csv(CSV_PATH_GLOBAL, fpath)
        if z is None or not np.isfinite(z):
            continue
            
        wave, flux, err = read_observed_spectrum(fpath)
        w_rest, f_rest, e_rest = get_rest_frame_spectrum(wave, flux, err, z)

        MUV, MUV_err = calculate_muv_and_error(w_rest, f_rest, e_rest, z)
        beta, beta_err = calculate_beta_and_error(w_rest, f_rest, e_rest)

        if MUV is None or not np.isfinite(MUV_err) or beta is None or not np.isfinite(beta_err):
            continue


        results.append({
            "prism_file": fname,
            "object_id": object_id,
            "z": z,
            "muv": MUV,
            "muv_err": MUV_err,
            "beta": beta,
            "beta_err": beta_err,
            "snr": snr_map[fname]
        })

    if not results:
        print("No valid data to plot.")
        return

    df = pd.DataFrame(results)
    print(df)

    def find_full_fits_path(base_dir, filename):
        """
        Search recursively within base_dir for the prism file.
        Returns full path if found, else None.
        """
        import os
        for root, _, files in os.walk(base_dir):
            if filename in files:
                return Path(root) / filename
        return None

    # --- Inspect unusually high β values ---
    suspicious = df[df["beta"] > -1]

    print("\n=== Galaxies with beta > -1 (suspiciously red UV slopes) ===")
    print(suspicious[["object_id", "prism_file", "z", "beta", "beta_err", "muv", "muv_err"]])

    print("\nCount:", len(suspicious))

    for row in suspicious.itertuples():
        print(f"\nInspecting beta windows for object {row.object_id}:")
        
        # Locate correct FITS file
        fpath = find_full_fits_path(SPECTRA_BASE_DIR, row.prism_file)
        
        if fpath is None:
            print("  ERROR — FITS file not found anywhere under Spectra/2D!")
            continue

        # Load the spectrum
        wave, flux, err = read_observed_spectrum(fpath)
        w_rest, f_rest, e_rest = get_rest_frame_spectrum(wave, flux, err, row.z)

        # Count C94 windows
        waves, fluxes, errs = sample_spectrum_C94(w_rest, f_rest, e_rest)

        print("  UV windows used:", len(waves))
        print("  Windows:", list(zip(waves, fluxes, errs)))




    # ================== PLOTS ======================
    
    # PLOT 1: Beta vs M_UV (with Photometric Background)
    plt.figure(figsize=(10,7))
    
    # -- Photometry Overlay --
    if not df_photo.empty:
        plt.errorbar(
            df_photo['muv'], 
            df_photo['beta'],
            xerr=[df_photo['muv_err_low'], df_photo['muv_err_high']],
            yerr=[df_photo['beta_err_low'], df_photo['beta_err_high']],
            fmt='o',
            color='#dc3545',       # Pale red
            markersize=2,
            alpha=0.3,             # Transparency
            elinewidth=0.5,        # Very thin lines
            zorder=0,              # Background layer
            label='JADES Photometry'
        )
    
    # -- Spectroscopic Data --
    plt.errorbar(df["muv"], df["beta"],
                 xerr=df["muv_err"], yerr=df["beta_err"],
                 fmt="o", color="green", zorder=5, label='This Work (Spec)')
                 
    plt.xlabel(r"Absolute UV Magnitude ($M_{UV}$)")
    plt.ylabel(r"UV Spectral Slope ($\beta$)")
    plt.title(r"UV Spectral Slope ($\beta$) vs $M_{UV}$")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR/"beta_vs_muv_40.png")

    # PLOT 2: M_UV vs Redshift
    plt.figure(figsize=(10,7))
    plt.errorbar(df["z"], df["muv"],
                 yerr=df["muv_err"],
                 fmt="o", color="blue")
    plt.xlabel(r"Redshift ($z$)")
    plt.ylabel(r"Absolute UV Magnitude ($M_{UV}$)")
    plt.gca().invert_yaxis()
    plt.title(r"$M_{UV}$ vs $z$")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR/"muv_vs_z_40.png")

    # PLOT 3: Beta vs Redshift
    plt.figure(figsize=(10,7))
    plt.errorbar(df["z"], df["beta"],
                 yerr=df["beta_err"],
                 fmt="o", color="red")
    plt.xlabel(r"Redshift ($z$)")
    plt.ylabel(r"UV Spectral Slope ($\beta$)")
    plt.title(r"$\beta$ vs $z$")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR/"beta_vs_z_40.png")

    df.to_csv(OUTPUT_DIR/"results_40.csv", index=False)
    print("Saved results!")


if __name__ == "__main__":
    process_and_plot()