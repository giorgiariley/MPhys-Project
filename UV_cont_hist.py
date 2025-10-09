import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
csv_path = "/nvme/scratch/work/Griley/....csv"

# Read the CSV
df = pd.read_csv(csv_path)

# Extract the SNR values
snr_values = df['whatever i have called it ']

# Plot histogram
plt.figure(figsize=(8,5))
plt.hist(snr_values, bins=20, color='dodgerblue', edgecolor='black', alpha=0.7)
plt.xlabel("Average SNR in UV continuum (1250–3000 Å)")
plt.ylabel("Number of spectra")
plt.title("Histogram of Average UV SNRs")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
