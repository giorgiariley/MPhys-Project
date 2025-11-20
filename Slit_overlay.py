import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy.io import fits
from pathlib import Path
from typing import Optional

# Configuration: You may need to adjust these based on the exact column names in HDU 3.
# These names are typical for JWST/PDR/DJA catalogues.
SLIT_COL_NAMES = {
    'RA_CENTRE': 'RA_TARG',  # Target RA
    'DEC_CENTRE': 'DEC_TARG', # Target Dec
    'SLIT_X_MIN': 'X_MIN',    # X coordinate of the slit starting point (often pixel)
    'SLIT_Y_MIN': 'Y_MIN',    # Y coordinate of the slit starting point (often pixel)
    'SLIT_LENGTH': 'SLIT_L',  # Length of the slit
    'SLIT_WIDTH': 'SLIT_W',   # Width of the slit
    'POS_ANGLE': 'POS_ANG',   # Position angle (rotation) of the slit (degrees)
    # The relevant columns might be pixel-based (like 'x_c', 'y_c', 'slit_len', 'slit_pa')
    # If the direct pixel positions are given, those are easier to plot.
}

def plot_slit_overlay(fits_path: str, slit_hdu_index: int = 3):
    """
    Reads the 2D finding chart image (HDU 1) and slit geometry (HDU 3/SLITS)
    and plots the slit overlay to replicate the 'slit_thumb' visual.

    NOTE: This uses generic column names. You might need to check the column
    names in your FITS file's HDU 3 table and update the SLIT_COL_NAMES dictionary.
    """
    fits_path = Path(fits_path)
    if not fits_path.exists():
        print(f"Error: FITS file not found at {fits_path}")
        return

    try:
        with fits.open(fits_path) as hdul:
            # --- 1. Load the 2D Image (HDU 1 is often the small cutout image) ---
            # NOTE: We assume the image is in HDU 1. This is standard for these quick-look FITS files.
            image_data = hdul[1].data 
            
            # --- 2. Load Slit Geometry Data (HDU 3 or specified index) ---
            # NOTE: HDU 3 is index 3 if HDU 0, 1, 2, 3...
            slit_data = hdul[slit_hdu_index].data
            
            if slit_data is None:
                print(f"Error: HDU {slit_hdu_index} is empty or not a recognized table type.")
                return

            # Check if necessary columns exist (using generic names for robustness)
            if 'X_START' not in slit_data.names or 'Y_START' not in slit_data.names or 'SLIT_PA' not in slit_data.names:
                 print("Warning: Standard pixel position/angle columns not found. Attempting to use RA/DEC columns.")
                 # Fallback to display the image only if geometric columns are not simple pixel coordinates
                 return _plot_image_only(image_data, fits_path.stem)
            
            # Assuming pixel-based coordinates are available for direct plotting:
            x_start = slit_data['X_START']
            y_start = slit_data['Y_START']
            slit_length = slit_data['SLIT_LEN']
            slit_width = slit_data['SLIT_WID'] # Assuming a constant width
            pos_angle = slit_data['SLIT_PA']

    except Exception as e:
        print(f"Error reading FITS file structure or data: {e}")
        return

    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Display the grayscale image
    ax.imshow(image_data, origin='lower', cmap='gray', vmin=np.percentile(image_data, 10), vmax=np.percentile(image_data, 99.5))
    
    # Iterate through all slits/segments found in the table
    for i in range(len(x_start)):
        # Calculate angle for matplotlib: 90 + PA (to match standard plotting conventions)
        # Note: The rotation logic can be highly dependent on the FITS standard used.
        # This implementation assumes the center of the rectangle is X_START, Y_START.
        
        # If the columns are pixel coordinates, we use them directly.
        # If the image origin is (0,0) and PA is relative to the Y-axis (North), use 90 - PA
        # We will use the common astrophysical convention of rotation angle.
        
        # Rectangle takes (xy, width, height, angle)
        rect = Rectangle(
            (x_start[i] - slit_width[i] / 2, y_start[i] - slit_length[i] / 2),
            width=slit_width[i],
            height=slit_length[i],
            angle=pos_angle[i],
            edgecolor='magenta',
            facecolor='none', # Pink/Magenta outline, no fill
            lw=1.5,
            alpha=0.8,
            rotation_point='center' # Rotate around the center of the target
        )
        
        # Add the patch to the axes
        ax.add_patch(rect)

    ax.set_title(f"Slit Overlay: {fits_path.stem}", fontsize=12)
    ax.axis('off') # Hide axes ticks and labels for the thumbnail look
    plt.tight_layout()
    plt.show()

def _plot_image_only(image_data, stem):
    """Fallback if slit data is complex/unavailable for simple plotting."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_data, origin='lower', cmap='gray', vmin=np.percentile(image_data, 10), vmax=np.percentile(image_data, 99.5))
    ax.set_title(f"Image Only (Slit Data Missing): {stem}", fontsize=12)
    ax.axis('off')
    plt.show()
    return

# --- Example of How to Use the Function ---
# Assuming you want to run this for one of your files:
FITS_FILE_PATH_EXAMPLE = "/raid/scratch/work/Griley/GALFIND_WORK/Spectra/2D/jades-gds-w04-v4/jades-gds-w04-v4_prism-clear_1212_5497.spec.fits"
plot_slit_overlay(FITS_FILE_PATH_EXAMPLE, slit_hdu_index=3)

