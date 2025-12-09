import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Union


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Union


def plot_popIII_contours(
    fname: Union[str, Path],
    out_png: Union[str, Path],
    xlabel: str,
    ylabel: str,
    title: str,
    add_legend: bool = True,
    figsize=(6, 6),
):
    """
    Plot Rusta's PopIII / hybrid contours for a single diagnostic plane.

    Per logU stripe (logu_idx = 0..4) there are 8 indices:
        O, G, L, D, O, G, L, D
    First quartet (0–3) is darker, second quartet (4–7) lighter.
    Contour thickness (and slightly alpha) increase with logu_idx
    to mimic Rusta's "thickness increases with U" behaviour.
    """

    fname = Path(fname)
    contours = np.load(fname, allow_pickle=True)
    n_indices = len(contours)

    # --- build per-index colour code, repeating O G L D ---
    pattern = ['O', 'G', 'L', 'D']
    reps = (n_indices + len(pattern) - 1) // len(pattern)
    idx_colour_code = (pattern * reps)[:n_indices]

    # map letters to colours + legend labels
    letter_to_colour = {
        'O': "#f28e2b",  # orange
        'G': "#b6d63b",  # yellowish green
        'L': "#76d4e3",  # cyan / light blue
        'D': "#4a7bd8",  # blue
    }
    phase_legend_labels = {
        'O': "Self-polluted PopIII",
        'G': "PopIII-rich Hybrid",
        'L': "PopIII-mid Hybrid",
        'D': "PopIII-poor Hybrid",
    }
    # number of logU stripes (5 for these grids)
    n_logu = n_indices // (4 * 2)
    fig, ax = plt.subplots(figsize=figsize)

    for i, level_paths in enumerate(contours):
        code = idx_colour_code[i]
        colour = letter_to_colour[code]

        # decode position within (logU, phase, level) grid
        logu_idx = i // (4 * 2)      # 0..4  (5 logU values)
        within   = i %  (4 * 2)      # 0..7  (OGLD OGLD per logU)
        first_quad = within < 4      # 0–3 = first OGLD, 4–7 = second OGLD

        # --- NEW: invert logU order for thickness scaling ---
        # so that the top stripe (highest U) is thickest
        logu_rank = (n_logu - 1) - logu_idx   # 0 = lowest U, max = highest U
        # -----------------------------------------------

        # scale with logU rank: higher rank -> thicker / stronger
        lw_base_outer = 0.9
        lw_step       = 0.25
        lw_outer = lw_base_outer + lw_step * logu_rank
        lw_inner = 0.7 * lw_outer

        fill_base = 0.07
        fill_step = 0.015
        fill_outer = fill_base + fill_step * logu_rank
        fill_inner = 0.5 * fill_outer

        line_base = 0.7
        line_step = 0.06
        line_outer = min(1.0, line_base + line_step * logu_rank)
        line_inner = 0.4 * line_outer

        # first quartet darker, second quartet lighter
        if first_quad:
            fill_alpha = fill_outer
            line_alpha = line_outer
            lw = lw_outer
        else:
            fill_alpha = fill_inner
            line_alpha = line_inner
            lw = lw_inner

        for verts in level_paths:
            xs, ys = verts[:, 0], verts[:, 1]

            # filled polygon
            ax.fill(xs, ys, facecolor=colour, alpha=fill_alpha,
                    edgecolor="none", zorder=1)

            # outline
            ax.plot(xs, ys, color=colour, alpha=line_alpha,
                    lw=lw, zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if add_legend:
        legend_handles = [
            mpatches.Patch(facecolor=letter_to_colour['O'], edgecolor='none',
                           label=phase_legend_labels['O']),
            mpatches.Patch(facecolor=letter_to_colour['G'], edgecolor='none',
                           label=phase_legend_labels['G']),
            mpatches.Patch(facecolor=letter_to_colour['L'], edgecolor='none',
                           label=phase_legend_labels['L']),
            mpatches.Patch(facecolor=letter_to_colour['D'], edgecolor='none',
                           label=phase_legend_labels['D']),
        ]
        ax.legend(handles=legend_handles,
                  loc='upper left',
                  frameon=False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)



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

    for cfg in configs:
        print("Plotting {} -> {}".format(cfg["fname"], cfg["out"]))
        plot_popIII_contours(
            fname=cfg["fname"],
            out_png=cfg["out"],
            xlabel=cfg["xlabel"],
            ylabel=cfg["ylabel"],
            title=cfg["title"],
            add_legend=True,
        )
