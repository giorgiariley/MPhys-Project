import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Union
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

# --------------------------
# Gutkin+16 file I/O
# --------------------------
GUTKIN_COLS = [
    "logU", "xid", "nh", "co_rel_sol", "mup",
    "OII_3727", "Hbeta", "OIII_4959", "OIII_5007", "NII_6548", "Halpha", "NII_6584",
    "SII_6717", "SII_6731",
    "NV_1240", "CIV_1548", "CIV_1551", "HeII_1640", "OIII_1661", "OIII_1666",
    "SiIII_1883", "SiIII_1888", "CIII_1908",
]

def read_gutkin_directory(gutkin_dir: Union[str, Path]) -> pd.DataFrame:
    gutkin_dir = Path(gutkin_dir)
    files = sorted(gutkin_dir.glob("nebular_emission_Z*.txt"))
    if not files:
        raise FileNotFoundError(f"No nebular_emission_Z*.txt files found in {gutkin_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(
            f,
            delim_whitespace=True,
            comment="#",
            header=None,
            names=GUTKIN_COLS,
            engine="python",
        )
        df["Zism_file"] = f.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

def safe_log10_ratio(num, den):
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    good = np.isfinite(num) & np.isfinite(den) & (num > 0) & (den > 0)
    out = np.full_like(num, np.nan, dtype=float)
    out[good] = np.log10(num[good] / den[good])
    return out
# --------------------------
# Feltre+16 AGN NLR file I/O
# --------------------------
AGN_COLS = [
    "logUs", "xid", "nh", "alpha",
    "OII_3727", "Hbeta", "OIII_4959", "OIII_5007", "OI_6300", "NII_6548", "Halpha", "NII_6584",
    "SII_6717", "SII_6731",
    "NV_1240", "CIV_1548", "CIV_1551", "HeII_1640", "OIII_1661", "OIII_1666",
    "SiIII_1883", "SiIII_1888", "CIII_1907", "CIII_1910",
]

def read_feltre_agn_directory(agn_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Reads Feltre+16 AGN NLR grids: nlr_nebular_Z*.txt (24 columns).
    """
    agn_dir = Path(agn_dir)
    files = sorted(agn_dir.glob("nlr_nebular_Z*.txt"))
    if not files:
        raise FileNotFoundError(f"No nlr_nebular_Z*.txt files found in {agn_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(
            f,
            delim_whitespace=True,
            comment="#",
            header=None,
            names=AGN_COLS,
            engine="python",
        )
        df["Zism_file"] = f.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def gutkin_plane(df: pd.DataFrame, plane: str) -> tuple[np.ndarray, np.ndarray]:
    # common combined lines
    civ1550  = df["CIV_1548"] + df["CIV_1551"]
    oiii1663 = df["OIII_1661"] + df["OIII_1666"]

    if plane == "C4He2_vs_C3He2":
        x = safe_log10_ratio(df["CIII_1908"], df["HeII_1640"])
        y = safe_log10_ratio(civ1550, df["HeII_1640"])
    elif plane == "O3He2_vs_C3He2":
        x = safe_log10_ratio(df["CIII_1908"], df["HeII_1640"])
        y = safe_log10_ratio(oiii1663, df["HeII_1640"])
    elif plane == "C4C3_vs_C3He2":
        x = safe_log10_ratio(df["CIII_1908"], df["HeII_1640"])
        y = safe_log10_ratio(civ1550, df["CIII_1908"])
    elif plane == "Si3He2_vs_C3He2":
        x = safe_log10_ratio(df["CIII_1908"], df["HeII_1640"])
        y = safe_log10_ratio(df["SiIII_1883"], df["HeII_1640"])
    else:
        raise ValueError(f"Unknown plane: {plane}")

    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]

def agn_plane(df: pd.DataFrame, plane: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Same planes as Gutkin, but with Feltre+16 AGN column names.
    Note: CIII]1908 is the sum of CIII]1907 + CIII]1910 in Feltre files.
    """
    civ1550  = df["CIV_1548"] + df["CIV_1551"]
    oiii1663 = df["OIII_1661"] + df["OIII_1666"]
    ciii1908 = df["CIII_1907"] + df["CIII_1910"]

    if plane == "C4He2_vs_C3He2":
        x = safe_log10_ratio(ciii1908, df["HeII_1640"])
        y = safe_log10_ratio(civ1550, df["HeII_1640"])
    elif plane == "O3He2_vs_C3He2":
        x = safe_log10_ratio(ciii1908, df["HeII_1640"])
        y = safe_log10_ratio(oiii1663, df["HeII_1640"])
    elif plane == "C4C3_vs_C3He2":
        x = safe_log10_ratio(ciii1908, df["HeII_1640"])
        y = safe_log10_ratio(civ1550, ciii1908)
    elif plane == "Si3He2_vs_C3He2":
        x = safe_log10_ratio(ciii1908, df["HeII_1640"])
        y = safe_log10_ratio(df["SiIII_1883"], df["HeII_1640"])
    else:
        raise ValueError(f"Unknown plane: {plane}")

    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]

# --------------------------
# KDE “empty contours”
# --------------------------
def kde_mass_contours(
    ax,
    x,
    y,
    levels_mass=(0.1, 0.30, 0.68, 0.90),
    gridsize=260,
    color="#b05ad6",
    linewidth=1.3,
    alphas= (0.9, 0.7, 0.5, 0.2),
    zorder=5,
):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 10:
        return

    kde = gaussian_kde(np.vstack([x, y]))

    xmin, xmax = np.nanpercentile(x, [0.5, 99.5])
    ymin, ymax = np.nanpercentile(y, [0.5, 99.5])

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, gridsize),
        np.linspace(ymin, ymax, gridsize),
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    zflat = zz.ravel()
    order = np.argsort(zflat)[::-1]
    zsorted = zflat[order]
    cumsum = np.cumsum(zsorted)
    cumsum /= cumsum[-1]

    thr = []
    for m in levels_mass:
        idx = np.searchsorted(cumsum, m)
        thr.append(zsorted[min(idx, len(zsorted) - 1)])
    thr = np.array(thr)
    thr_sorted = np.sort(thr)[::-1]

    for k,level in enumerate(thr_sorted):
        ax.contour(
            xx, yy, zz,
            levels=[level],
            colors=[color],
            linewidths=linewidth,
            alpha=alphas[k],
            zorder=zorder
        )



def plot_popIII_contours(
    fname: Union[str, Path],
    out_png: Union[str, Path, None] = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    add_legend: bool = True,
    figsize=(6, 6),
    ax=None,
):
    fname = Path(fname)
    contours = np.load(fname, allow_pickle=True)
    n_indices = len(contours)

    pattern = ['O', 'G', 'L', 'D']
    reps = (n_indices + len(pattern) - 1) // len(pattern)
    idx_colour_code = (pattern * reps)[:n_indices]

    letter_to_colour = {
        'O': "#f28e2b",
        'G': "#b6d63b",
        'L': "#76d4e3",
        'D': "#4a7bd8",
    }
    phase_legend_labels = {
        'O': "Self-polluted PopIII",
        'G': "PopIII-rich Hybrid",
        'L': "PopIII-mid Hybrid",
        'D': "PopIII-poor Hybrid",
    }

    n_logu = n_indices // (4 * 2)

    owns_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        owns_fig = True
    else:
        fig = ax.figure

    for i, level_paths in enumerate(contours):
        code = idx_colour_code[i]
        colour = letter_to_colour[code]

        logu_idx = i // (4 * 2)
        within = i % (4 * 2)
        first_quad = within < 4

        logu_rank = (n_logu - 1) - logu_idx

        lw_outer = 0.9 + 0.25 * logu_rank
        lw_inner = 0.7 * lw_outer

        fill_outer = 0.07 + 0.015 * logu_rank
        fill_inner = 0.5 * fill_outer

        line_outer = min(1.0, 0.7 + 0.06 * logu_rank)
        line_inner = 0.4 * line_outer

        if first_quad:
            fill_alpha, line_alpha, lw = fill_outer, line_outer, lw_outer
        else:
            fill_alpha, line_alpha, lw = fill_inner, line_inner, lw_inner

        for verts in level_paths:
            xs, ys = verts[:, 0], verts[:, 1]
            ax.fill(xs, ys, facecolor=colour, alpha=fill_alpha, edgecolor="none", zorder=1)
            ax.plot(xs, ys, color=colour, alpha=line_alpha, lw=lw, zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if add_legend:
        legend_handles = [
            mpatches.Patch(facecolor=letter_to_colour[k], edgecolor="none",
                        label=phase_legend_labels[k])
            for k in ['O', 'G', 'L', 'D']
        ]

        # Add contour proxies
        legend_handles += [
            Line2D([0], [0], color="#b05ad6", lw=1.5, label="SF galaxies (Gutkin+16)"),
            Line2D([0], [0], color="#8c7a6b", lw=1.5, label="AGN (Feltre+16)"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", frameon=False)
    # Match Rusta axis ranges 
    ax.set_xlim(-2.6, 1.8)
    ax.set_ylim(-3.1, 2.5)

    fig.tight_layout()

    if out_png is not None:
        fig.savefig(out_png, dpi=300)

    if owns_fig:
        plt.close(fig)

    return ax



if __name__ == "__main__":

    configs = [
        {
            "fname": "/raid/scratch/work/Griley/GALFIND_WORK/Contours/Rusta_data/contours_C4He2_C3He2.npy",
            "xlabel": r"log CIII]$\lambda1908$/He II$\lambda1640$",
            "ylabel": r"log CIV$\lambda1550$/He II$\lambda1640$",
            "title": r"PopIII / hybrid contours (C IV/He II vs C III]/He II)",
            "out": "panel_C4He2_C3He2.png",
        },
        {
            "fname": "/raid/scratch/work/Griley/GALFIND_WORK/Contours/Rusta_data/contours_O3He2_C3He2.npy",
            "xlabel": r"log CIII]$\lambda1908$/He II$\lambda1640$",
            "ylabel": r"log O III]$\lambda1663$/He II$\lambda1640$",
            "title": r"PopIII / hybrid contours (O III]/He II vs C III]/He II)",
            "out": "panel_O3He2_C3He2.png",
        },
        {
            "fname": "/raid/scratch/work/Griley/GALFIND_WORK/Contours/Rusta_data/contours_C4C3_C3He2.npy",
            "xlabel": r"log CIII]$\lambda1908$/He II$\lambda1640$",
            "ylabel": r"log CIV$\lambda1550$/CIII]$\lambda1908$",
            "title": r"PopIII / hybrid contours (C IV/C III] vs C III]/He II)",
            "out": "panel_C4C3_C3He2.png",
        },
        {
            "fname": "/raid/scratch/work/Griley/GALFIND_WORK/Contours/Rusta_data/contours_Si3He2_C3He2.npy",
            "xlabel": r"log CIII]$\lambda1908$/He II$\lambda1640$",
            "ylabel": r"log S III]$\lambda1883$/He II$\lambda1640$",
            "title": r"PopIII / hybrid contours (S III]/He II vs C III]/He II)",
            "out": "panel_S3He2_C3He2.png",
        },
    ]

    # Load Gutkin SF grids (purple)
    GUTKIN_DIR = "/raid/scratch/work/Griley/GALFIND_WORK/Contours/nebular_emission_gutkin16"
    df_gutkin = read_gutkin_directory(GUTKIN_DIR)

    # Load Feltre AGN grids (grey/brown)
    AGN_DIR = "/raid/scratch/work/Griley/GALFIND_WORK/Contours/AGN_NLR_nebular_feltre16"
    df_agn = read_feltre_agn_directory(AGN_DIR)

    plane_map = {
        "panel_C4He2_C3He2.png": "C4He2_vs_C3He2",
        "panel_O3He2_C3He2.png": "O3He2_vs_C3He2",
        "panel_C4C3_C3He2.png":  "C4C3_vs_C3He2",
        "panel_S3He2_C3He2.png": "Si3He2_vs_C3He2",
    }

    for cfg in configs:
        fig, ax = plt.subplots(figsize=(6, 6))

        # PopIII coloured contours
        plot_popIII_contours(
            fname=cfg["fname"],
            out_png=None,
            xlabel=cfg["xlabel"],
            ylabel=cfg["ylabel"],
            title=cfg["title"],
            add_legend=True,
            ax=ax
        )

        plane = plane_map[cfg["out"]]

        # Gutkin SF galaxies contours (purple)
        xg, yg = gutkin_plane(df_gutkin, plane)
        kde_mass_contours(
            ax, xg, yg,
            color="#b05ad6",
            linewidth=1.3,
            alphas= (0.9, 0.7, 0.5, 0.2),
            zorder=5
        )

        # Feltre AGN contours (grey)
        xa, ya = agn_plane(df_agn, plane)
        kde_mass_contours(
            ax, xa, ya,
            color="#7f7f7f",
            linewidth=1.1,
            alphas= (0.9, 0.7, 0.5, 0.2),
            zorder=6
        )

        fig.tight_layout()
        fig.savefig(cfg["out"], dpi=300)
        plt.close(fig)