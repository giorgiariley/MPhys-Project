#making an muv beta plot 
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

#first, lets match up the uv snr file names to fits files in my master fits catalogue
fits_cat = "/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/gdsgdn_catalogue.fits"
uv_snr_csv = "/raid/scratch/work/rroberts/mphys_pop_III/ultrablue-galaxies-mphys/uv_snr5plus_with_prism_and_medium.csv"
spectra_folder = "/raid/scratch/work/austind/GALFIND_WORK/Spectra/2D"
C_LIGHT_AA_PER_S = 2.99792458e18
AB_MAG_ZP_JY = 8.90
W_UV_MIN = 1350.0
W_UV_MAX = 1800.0
cosmo = Planck18
LOWER_C94_FILT = np.array([1268., 1309., 1342., 1407., 1562., 1677., 1760., 1866., 1930., 2400.])
UPPER_C94_FILT = np.array([1284., 1316., 1371., 1515., 1583., 1740., 1833., 1890., 1950., 2580.])

#open up the fits catalogue and find the file name which is in the column file, and the redshift in column z
#open fits catalogue 
with fits.open(fits_cat) as hdul: 
    data = hdul[1].data 
    file_names = data["file"] 
    redshifts = data["zrf"]


#open the uv snr csv and find the file name and the snr value
uv_snr_df = pd.read_csv(uv_snr_csv, usecols=["prism_file", "avg_snr_uv"]) 
uv_snr_dict = uv_snr_df.set_index("prism_file")["avg_snr_uv"].to_dict()


#now we have the file names, redshifts, and snr values, we can match them up and create a new list of dictionaries with the file name, redshift, and snr value
matched_data = []
for file_name, redshift in zip(file_names, redshifts):
    if file_name in uv_snr_dict:
        snr_value = uv_snr_dict[file_name]
        matched_data.append({"file": file_name, "z": redshift, "snr": snr_value})

#now we have a list of dictionaries with the file name, redshift, and snr value, we can create a new csv file with this data
output_csv = "matched_uv_snr_redshift.csv"
with open(output_csv, mode="w", newline="") as csv_file:
    fieldnames = ["file", "z", "snr"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for data in matched_data:
        writer.writerow(data)
    print(f"Matched data has been written to {output_csv}")

#need to open the spectra, and extract wave flux error etc for each to later calc beta and muv 
def read_observed_spectrum(fpath):
    with fits.open(fpath) as hdul:
        data = hdul[1].data
        wave = np.array(data["wave"])
        flux = np.array(data["flux"])
        err = np.array(data["err"] if "err" in data.columns.names else data["full_err"])
    return wave, flux, err

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


#---------------------------------------------------------
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

#main code
def main():
    """
    Uses the precomputed matched_data (file, z, snr) to:
      1) locate each spectrum FITS under spectra_folder
      2) compute MUV and beta (with errors)
      3) make a beta vs MUV plot
    """
    # Build a fast lookup from filename -> full path (only once)
    spectra_base = Path(spectra_folder)
    all_fits_paths = {p.name: p for p in spectra_base.rglob("*.fits")}

    results = []
    n_missing = 0
    n_failed = 0

    for row in matched_data:
        fname = str(row["file"]).strip()
        z = row["z"]

        # Locate the spectrum file
        fpath = all_fits_paths.get(fname)
        if fpath is None:
            n_missing += 1
            continue

        try:
            wave, flux, err = read_observed_spectrum(fpath)
            w_rest, f_rest, e_rest = get_rest_frame_spectrum(wave, flux, err, z)

            MUV, MUV_err = calculate_muv_and_error(w_rest, f_rest, e_rest, z)
            beta, beta_err = calculate_beta_and_error(w_rest, f_rest, e_rest)

            if MUV is None or beta is None:
                n_failed += 1
                continue

            results.append({
                "file": fname,
                "z": z,
                "snr": row["snr"],
                "muv": MUV,
                "muv_err": MUV_err,
                "beta": beta,
                "beta_err": beta_err
            })
        except Exception:
            n_failed += 1
            continue

    df = pd.DataFrame(results)
    print(f"Total matched rows: {len(matched_data)}")
    print(f"Spectra missing on disk: {n_missing}")
    print(f"Failed / insufficient data: {n_failed}")
    print(f"Final plotted points: {len(df)}")

    #print the number of points below beta value of -3 as wel as their filenames and muv
    below_beta_3 = df[df["beta"] < -3]
    print(f"Number of points with beta < -3: {len(below_beta_3)}")
    if not below_beta_3.empty:
        print("Files with beta < -3:")
        for _, row in below_beta_3.iterrows():
            print(f" {row['file']}: MUV={row['muv']:.2f}, beta={row['beta']:.2f}")

    if df.empty:
        print("No results to plot.")
        return

    # Save results table (optional but useful)
    df.to_csv("muv_beta_results.csv", index=False)

    # --- Plot: beta vs MUV ---
    plt.figure(figsize=(10, 7))
    plt.errorbar(
        df["muv"], df["beta"],
        xerr=df["muv_err"], yerr=df["beta_err"],
        fmt="o", alpha=0.8
    )
    plt.xlabel(r"Absolute UV Magnitude ($M_{UV}$)")
    plt.ylabel(r"UV Spectral Slope ($\beta$)")
    plt.title(r"$\beta$ vs $M_{UV}$ (Spectroscopy)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("beta_vs_muv.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
