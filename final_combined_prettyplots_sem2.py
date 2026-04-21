import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from msaexp import msa

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

# --- Paths ---
CUTOUT_BASE = '/raid/scratch/work/Griley/GALFIND_WORK/Cutouts/v13'
SPECTRA_BASE = '/raid/scratch/work/austind/GALFIND_WORK/Spectra/2D'
MSA_METAFILE_BASE = '/raid/scratch/work/austind/GALFIND_WORK/Spectra/MSA_metafiles'
OUTPUT_DIR = '/nvme/scratch/work/Griley/Masters/prettyplots_sem2'
FILTER = 'F444W'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Spectrum functions ---
def read_spectrum(fits_path):
    with fits.open(fits_path) as hdul:
        try:
            spec1d = hdul['SPEC1D'].data
        except KeyError:
            spec1d = hdul[1].data
        wave, flux, err = spec1d['wave'], spec1d['flux'], spec1d['err']
        if np.nanmax(wave) > 100.0:
            wave = wave / 1e4
    return wave, flux, err

def read_2d_spectrum(fits_path):
    with fits.open(fits_path) as hdul:
        if 'SCI' in hdul:
            return hdul['SCI'].data
        raise KeyError("Could not find 2D spectral data.")

def clean_data(wave, flux, err, data_2d):
    mask = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)
    if data_2d.shape[1] != len(mask):
        min_len = min(data_2d.shape[1], len(mask))
        mask = mask[:min_len]
        return wave[:min_len][mask], flux[:min_len][mask], err[:min_len][mask], data_2d[:, mask]
    return wave[mask], flux[mask], err[mask], data_2d[:, mask]

def plot_2d_spectrum(ax, data_2d, wave):
    vmin, vmax = np.percentile(data_2d[np.isfinite(data_2d)], [10, 99])
    n_spatial = data_2d.shape[0]
    y_corners = np.arange(n_spatial + 1) - n_spatial / 2.0
    wave_mid = (wave[:-1] + wave[1:]) / 2.0
    x_corners = np.concatenate([[wave[0] - (wave[1]-wave[0])/2], wave_mid, [wave[-1] + (wave[-1]-wave[-2])/2]])
    if not np.all(np.diff(x_corners) > 0):
        x_corners = np.linspace(wave[0], wave[-1], len(wave) + 1)
    ax.pcolormesh(x_corners, y_corners, data_2d, cmap='magma', vmin=vmin, vmax=vmax, shading='auto')
    ax.set_ylabel("Spatial Pixel")

def plot_1d_spectrum(ax, wave, flux, err):
    ax.plot(wave, flux, color='black', lw=0.7)
    ax.fill_between(wave, flux - err, flux + err, color='grey', alpha=0.4)
    finite = np.concatenate([flux - err, flux + err])
    finite = finite[np.isfinite(finite)]
    if len(finite) > 10:
        y_min, y_max = np.percentile(finite, [0.9, 99.8])
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.15*y_range, y_max + 0.15*y_range)
    ax.set_xlabel(r"Observed Wavelength [$\mu$m]")
    ax.set_ylabel(r"$f_{\nu}$ [$\mu$Jy]")
    ax.grid(alpha=0.3)

def plot_slit_cutout(ax, cutout_path, spec_path, msa_metafile_base):
    # load cutout
    with fits.open(cutout_path) as hdul:
        image = hdul['SCI'].data
        wcs = WCS(hdul['SCI'].header)
    # load MSA info from spec header
    with fits.open(spec_path) as hdul:
        h = hdul['SCI'].header
        msametfl = h['MSAMETFL']
        msametid = int(h['MSAMETID'])
        patt_num = int(h['PATT_NUM'])
    # load slit regions
    metafile_path = f"{msa_metafile_base}/{msametfl}"
    MSA_metafile = msa.MSAMetafile(metafile_path)
    slits = MSA_metafile.regions_from_metafile(
        dither_point_index=patt_num,
        as_string=False,
        with_bars=True,
        msa_metadata_id=msametid,
    )
    # display image
    vmin = np.percentile(image, 10)
    vmax = np.percentile(image, 99.5)
    ax.imshow(image, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    # overlay slits
    for s in slits:
        xy = np.array(s.xy[0])
        pixels = wcs.world_to_pixel_values(xy[:, 0], xy[:, 1])
        x_pix = np.append(pixels[0], pixels[0][0])
        y_pix = np.append(pixels[1], pixels[1][0])
        color = 'magenta' if s.meta['is_source'] else 'lightpink'
        lw = 2.0 if s.meta['is_source'] else 2.0
        ax.plot(x_pix, y_pix, color=color, lw=lw, alpha=0.8)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.axis('off')

# --- Main ---
df = pd.read_csv('/nvme/scratch/work/Griley/Masters/AGN/subsample_photometric_ids.csv')

# select single object
target_file = 'gds-deep-v4_prism-clear_1210_9880.spec.fits'
row = df[df['file'] == target_file].iloc[0]

survey_id = row['SURVEY_ID']
survey = row['SURVEY']
z = row['zrf']

subdir = target_file.split('_prism-clear')[0]
spec_path = f"{SPECTRA_BASE}/{subdir}/{target_file}"
cutout_path = f"{CUTOUT_BASE}/{survey}/ACS_WFC+NIRCam/3.00as/F444W/data/{survey_id}.fits"

# read spectra
wave, flux, err = read_spectrum(spec_path)
data_2d = read_2d_spectrum(spec_path)
wave, flux, err, data_2d = clean_data(wave, flux, err, data_2d)

# build figure
fig = plt.figure(figsize=(16, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[3, 1])
ax_2d = fig.add_subplot(gs[0, 0])
ax_1d = fig.add_subplot(gs[1, 0], sharex=ax_2d)
ax_cut = fig.add_subplot(gs[:, 1])

plot_2d_spectrum(ax_2d, data_2d, wave)
ax_2d.text(0.98, 0.92, f"z = {z:.3f}", transform=ax_2d.transAxes,
           ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
ax_2d.set_title(f"{target_file.replace('.spec.fits', '')}", pad=10)
plt.setp(ax_2d.get_xticklabels(), visible=False)

plot_1d_spectrum(ax_1d, wave, flux, err)
ax_2d.set_xlim(wave[0], wave[-1])

plot_slit_cutout(ax_cut, cutout_path, spec_path, MSA_METAFILE_BASE)
ax_cut.set_title(f"F444W | ID {survey_id}", pad=10)

out_path = f"{OUTPUT_DIR}/1210_9880_{survey_id}.png"
plt.savefig(out_path, dpi=200)
plt.close()
print(f"Saved: {out_path}")