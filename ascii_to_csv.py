import os
import pandas as pd

def convert_ascii_to_csv(input_path, output_path):
    mz_values = []
    intensity_values = []

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue  

            parts = line.split(",")
            if len(parts) != 2:
                continue  

            try:
                mz = float(parts[0])
                intensity = float(parts[1])
                mz_values.append(mz)
                intensity_values.append(intensity)
            except ValueError:
                continue  

    df = pd.DataFrame({'m/z': mz_values, 'intensity': intensity_values})
    df.to_csv(output_path, index=False)
    print(f"✅ Converted: {os.path.basename(input_path)} → {os.path.basename(output_path)}")

def batch_convert_txt_to_csv(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            txt_path = os.path.join(folder_path, filename)
            csv_path = os.path.join(folder_path, filename.replace(".txt", ".csv"))
            convert_ascii_to_csv(txt_path, csv_path)

batch_convert_txt_to_csv("spectra")
