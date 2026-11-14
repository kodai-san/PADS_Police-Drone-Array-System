import os
import glob
import pandas as pd

# Base directory
base_dir = r"C:\Users\kodai\Downloads\78e15ac04eea17ea679722a7f8936599eef1f407"

# Collect all CSV files
all_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
print(f"Found {len(all_files)} CSV files")

# Load and combine
dfs = []
for f in all_files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
    except Exception as e:
        print(f"Skipping {f}: {e}")

combined = pd.concat(dfs, ignore_index=True)
print(f"Combined shape: {combined.shape}")

# Print column names to verify
print("Columns:", combined.columns.tolist())

# Filter by crime type = Burglary (column J = index 9)
crime_col = combined.columns[10]   # Crime type
lon_col   = combined.columns[5]   # Longitude (E)
lat_col   = combined.columns[6]   # Latitude (F)

print(f"Crime col: {crime_col}, Lon col: {lon_col}, Lat col: {lat_col}")

filtered = combined[combined[crime_col] == "Burglary"].copy()
print(f"After Burglary filter: {filtered.shape}")

# Save output
output_path = os.path.join(base_dir, "westminster_burglaries.csv")
filtered.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")