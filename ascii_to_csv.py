import os
import pandas as pd

def convert_ascii_to_csv(input_path, output_path):
    mz_values = []
    intensity_values = []

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(',')
            if len(parts) < 2:
                continue

            try:
                mz = float(parts[0].strip())
                intensity = float(parts[1].strip())
                mz_values.append(mz)
                intensity_values.append(intensity)
            except ValueError:
                continue

    if not mz_values:
        print(f"⚠️ No data extracted from: {os.path.basename(input_path)}. Check file format.")
        return

    df = pd.DataFrame({'m/z': mz_values, 'intensity': intensity_values})
    df.to_csv(output_path, index=False)
    print(f"✅ Converted: {os.path.basename(input_path)} → {os.path.basename(output_path)}")

def batch_convert_txt_to_csv(folder_path):
    print(f"Starting batch conversion in folder: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found. Please create it and place your .txt files inside.")
        return

    txt_files_found = False
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            txt_files_found = True
            txt_path = os.path.join(folder_path, filename)
            csv_path = os.path.join(folder_path, filename.replace(".txt", ".csv"))
            convert_ascii_to_csv(txt_path, csv_path)
    
    if not txt_files_found:
        print(f"No .txt files found in '{folder_path}'. Please ensure your spectrum files are in .txt format and placed there.")
    print("Batch conversion complete.")

batch_convert_txt_to_csv("spectra")
