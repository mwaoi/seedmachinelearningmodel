import os
import pandas as pd

def extract_marker_intensity(df, target_mz, window=1.0):
    region = df[(df['m/z'] > target_mz - window) & (df['m/z'] < target_mz + window)]
    return region['intensity'].max() if not region.empty else 0

directory = "./spectra"
records = []

for file in os.listdir(directory):
    if file.endswith(".csv"):
        path = os.path.join(directory, file)
        
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        filename_parts = file.replace(".csv", "").split("_")
        
        well_id = filename_parts[-1] # e.g., '070001-A6'
        
        species_digest_part = filename_parts[1] 
        
        species_label = species_digest_part.split(" ")[0] 
        
        sample_id = f"{species_label}_{well_id}" 

        record = {
            "Well_ID": well_id,
            "Sample_ID": sample_id, 
            "Species_Label": species_label, 
            "marker_2593": extract_marker_intensity(df, 2593),  # Human
            "marker_2563": extract_marker_intensity(df, 2563),  # Horse
            "marker_2503": extract_marker_intensity(df, 2503),  # Dog
            "marker_2042": extract_marker_intensity(df, 2042),  # BSA
            "Notes": ""
        }
        records.append(record)

df_features = pd.DataFrame(records)
df_features.to_csv("Extracted_MALDI_Features.csv", index=False)

print("✅ Features extracted and saved to Extracted_MALDI_Features.csv")
