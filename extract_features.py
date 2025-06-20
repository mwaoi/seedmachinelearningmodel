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
        df = pd.read_csv(path)
        
        record = {
            "Well_ID": file.split("_")[0],
            "Sample_ID": "_".join(file.split("_")[1:3]),
            "Species_Label": file.split("_")[3].replace(".csv", ""),
            "marker_2593": extract_marker_intensity(df, 2593),
            "marker_2563": extract_marker_intensity(df, 2563),
            "marker_2503": extract_marker_intensity(df, 2503),
            "Notes": ""
        }
        records.append(record)

df_features = pd.DataFrame(records)
df_features.to_csv("Extracted_MALDI_Features.csv", index=False)

print("Features extracted and saved to Extracted_MALDI_Features.csv")
