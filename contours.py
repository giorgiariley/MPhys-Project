import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fname = "/raid/scratch/work/Griley/GALFIND_WORK/Contours/Rusta_data/contours_C4He2_C3He2.npy"
contours = np.load(fname, allow_pickle=True)

fig, ax = plt.subplots(figsize=(6, 6))

for i, level_paths in enumerate(contours):
    for verts in level_paths:
        xs, ys = verts[:, 0], verts[:, 1]
        ax.plot(xs, ys, color="lightgrey", lw=0.8, alpha=0.8)

        # label each contour roughly at its centre
        ax.text(xs.mean(), ys.mean(), str(i), fontsize=6, ha="center", va="center")
        
ax.set_xlabel(r"log CIII]$\lambda1908$/He II$\lambda1640$")
ax.set_ylabel(r"log CIV$\lambda1550$/He II$\lambda1640$")
ax.set_title("Index map for Rusta contours")

plt.tight_layout()
fig.savefig("contours_index_map.png", dpi=300)

import numpy as np
import matplotlib.pyplot as plt

fname = "/raid/scratch/work/Griley/GALFIND_WORK/Contours/Rusta_data/contours_C4He2_C3He2.npy"
contours = np.load(fname, allow_pickle=True)

# -------------------------------------------------
# Choose colours for each index 0..39 by hand
#   Use only these four letters:
#   'O' = orange, 'G' = green, 'L' = light blue, 'D' = dark blue
#   Start with a guess – you can change & rerun until it looks right.
# -------------------------------------------------
idx_colour_code = [
    'O','G','L','D','O','G','L','D',   # 0–7
    'O','G','L','D','O','G','L','D',   # 8–15
    'O','G','L','D','O','G','L','D',   # 8–15
    'O','G','L','D','O','G','L','D',   # 8–15
    'O','G','L','D','O','G','L','D',   # 8–15
]
# ^^^ edit these letters as you go:
# e.g. change some 'O' to 'G','L','D' etc.

# safety check
assert len(idx_colour_code) == len(contours)

# map letter -> actual colour
letter_to_colour = {
    'O': "#f28e2b",  # orange
    'G': "#59a14f",  # green
    'L': "#4dc9ff",  # light blue
    'D': "#1f77b4",  # dark blue
}

fig, ax = plt.subplots(figsize=(6, 6))

for i, level_paths in enumerate(contours):
    code = idx_colour_code[i]
    colour = letter_to_colour[code]

    # optionally vary alpha by level (outer vs inner) using our decode:
    within    = i % (4 * 2)
    level_idx = within % 2        # 0 or 1
    fill_alpha = 0.12 if level_idx == 0 else 0.06
    line_alpha = 0.8

    for verts in level_paths:
        xs, ys = verts[:, 0], verts[:, 1]

        # fill
        ax.fill(xs, ys, facecolor=colour, alpha=fill_alpha,
                edgecolor="none", zorder=1)

        # outline
        ax.plot(xs, ys, color=colour, alpha=line_alpha, lw=1.0, zorder=2)

ax.set_xlabel(r"log CIII]$\lambda1908$/He II$\lambda1640$")
ax.set_ylabel(r"log CIV$\lambda1550$/He II$\lambda1640$")
ax.set_title(" PopIII / hybrid contours (Rusta et al 25) ")

# --- legend ---

legend_handles = [
    mpatches.Patch(facecolor=letter_to_colour['O'], edgecolor='none',
                   label='Self-polluted PopIII'),
    mpatches.Patch(facecolor=letter_to_colour['G'], edgecolor='none',
                   label='PopIII-rich Hybrid'),
    mpatches.Patch(facecolor=letter_to_colour['L'], edgecolor='none',
                   label='PopIII-mid Hybrid'),
    mpatches.Patch(facecolor=letter_to_colour['D'], edgecolor='none',
                   label='PopIII-poor Hybrid'),
]

ax.legend(handles=legend_handles,
          loc='upper left',    # tweak position if it overlaps points
          frameon=False)


plt.tight_layout()
fig.savefig("contours_manual_index_colours.png", dpi=300)
