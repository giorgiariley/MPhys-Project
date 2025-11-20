import pandas as pd
import os

# --- Configuration ---

# Path to your filtered CSV file
csv_file_path = "/nvme/scratch/work/Griley/Masters/exposure_photometry_matches_filtered.csv"

# The column name that contains the unique galaxy IDs
id_column_name = "photometry_NUMBER"

# ----------------------

print(f"Loading data from: {csv_file_path}")

# Check if file exists
if not os.path.exists(csv_file_path):
    print(f"ERROR: File not found at {csv_file_path}")
    exit()

try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Check if the required column exists in the DataFrame
    if id_column_name not in df.columns:
        print(f"ERROR: Column '{id_column_name}' not found in the CSV file.")
        print(f"Available columns are: {df.columns.tolist()}")
    else:
        # Use the .nunique() method to count the number of distinct values
        unique_object_count = df[id_column_name].nunique()
        
        print("\n--- Results ---")
        print(f"Total number of rows (all filters): {len(df)}")
        print(f"Number of distinct objects (unique '{id_column_name}' IDs): {unique_object_count}")

except Exception as e:
    print(f"An error occurred while processing the file: {e}")
