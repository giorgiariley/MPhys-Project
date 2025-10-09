import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
csv_path = "/nvme/scratch/work/Griley/Masters/UV_SNR_plots/uv_snr_results.csv"
out_path = "/nvme/scratch/work/Griley/Masters/UV_SNR_histogram.png"

# Read the CSV
df = pd.read_csv(csv_path)

# Ensure the 'uv_snr_mean' column is treated as numeric, coercing errors
df['uv_snr_mean'] = pd.to_numeric(df['uv_snr_mean'], errors='coerce')

# --- NEW: COUNT GALAXIES WITH SNR > 5 ---
snr_threshold_5 = 5
num_snr_gt_5 = (df['uv_snr_mean'] > snr_threshold_5).sum()
print(f"Number of galaxies with average SNR > {snr_threshold_5}: {num_snr_gt_5}")
print("-" * 40)
# ----------------------------------------

# Filter out high SNR values (> 30) for plotting
threshold = 30
df_filtered = df[df['uv_snr_mean'] <= threshold].dropna(subset=['uv_snr_mean']) # Also drop NaNs just in case

# Count how many spectra are above the threshold and will not be plotted
num_excluded = (df['uv_snr_mean'] > threshold).sum()
print(f"Number of spectra with avg SNR > {threshold} (excluded from plot): {num_excluded}")

# Count how many spectra are being plotted
num_plotted = len(df_filtered)
print(f"Number of spectra being plotted (≤ {threshold}): {num_plotted}")
print("-" * 40)

# Extract the filtered SNR values
snr_values = df_filtered['uv_snr_mean']

# --- Print highest and lowest values in the plotted range ---
if not df_filtered.empty:
    max_idx = df_filtered['uv_snr_mean'].idxmax()
    min_idx = df_filtered['uv_snr_mean'].idxmin()

    max_snr = df_filtered.loc[max_idx, 'uv_snr_mean']
    min_snr = df_filtered.loc[min_idx, 'uv_snr_mean']

    # Use .iloc[0] for safety when accessing potentially filtered index/column
    max_file = df_filtered.loc[max_idx, 'file']
    min_file = df_filtered.loc[min_idx, 'file']

    print(f"Highest average SNR (≤ {threshold}): {max_snr:.2f} ({max_file})")
    print(f"Lowest average SNR:                {min_snr:.2f} ({min_file})")
    print("-" * 40)

    # --- Plot histogram of filtered data ---
    plt.figure(figsize=(8,5))
    plt.hist(snr_values, bins=60, color='dodgerblue', edgecolor='black', alpha=0.7)

    plt.xlabel("Average SNR in UV continuum (1250–3000 Å)")
    plt.ylabel("Number of spectra")
    plt.title(f"Histogram of Average UV SNRs (SNR ≤ {threshold})")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(out_path, dpi=200)
    plt.show()
else:
    print("No spectra available for plotting after filtering and dropping NaNs.")