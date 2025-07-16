import streamlit as st
import pandas as pd
import joblib
import os
import io

def convert_ascii_to_dataframe(file_content):
    mz_values = []
    intensity_values = []

    content_str = file_content.decode('utf-8')
    
    for line in content_str.splitlines():
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
        return None 

    return pd.DataFrame({'m/z': mz_values, 'intensity': intensity_values})

def extract_marker_intensity(df, target_mz, window=1.0):
    region = df[(df['m/z'] > target_mz - window) & (df['m/z'] < target_mz + window)]
    return region['intensity'].max() if not region.empty else 0

st.set_page_config(page_title="Hair Species Predictor", layout="centered")

st.title("🔬 Hair Species Predictor")
st.markdown("Upload a MALDI-TOF spectrum file (.txt or .csv) to predict the hair species (Human, Horse, Dog, or BSA).")

MODEL_FILENAME = 'hair_species_classifier.pkl'
try:
    model = joblib.load(MODEL_FILENAME)
    st.success("Machine learning model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: '{MODEL_FILENAME}' not found. Please ensure your trained model file is in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure it's a valid joblib file.")
    st.stop()

uploaded_file = st.file_uploader("Choose a spectrum file", type=["txt", "csv"])

if uploaded_file is not None:
    st.write("File uploaded successfully!")
    
    spectrum_df = None
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == "txt":
        st.info("Processing .txt file...")
        spectrum_df = convert_ascii_to_dataframe(uploaded_file.getvalue())
        if spectrum_df is None or spectrum_df.empty:
            st.error("Could not extract valid m/z and intensity data from the .txt file. Please check its format.")
            st.stop()
    elif file_extension == "csv":
        st.info("Processing .csv file...")
        try:
            spectrum_df = pd.read_csv(uploaded_file)
            if 'm/z' not in spectrum_df.columns.str.lower() or 'intensity' not in spectrum_df.columns.str.lower():
                st.error("CSV file must contain 'm/z' and 'intensity' columns (case-insensitive).")
                st.stop()
            spectrum_df.columns = spectrum_df.columns.str.lower()
            spectrum_df = spectrum_df.rename(columns={'m/z': 'm/z', 'intensity': 'intensity'})

        except Exception as e:
            st.error(f"Error reading CSV file: {e}. Please ensure it's a valid CSV.")
            st.stop()
    else:
        st.error("Unsupported file type. Please upload a .txt or .csv file.")
        st.stop()

    if spectrum_df is not None and not spectrum_df.empty:
        st.subheader("Extracted Spectrum Data (First 5 rows):")
        st.dataframe(spectrum_df.head())

        marker_mzs = {
            "marker_2593": 2593, 
            "marker_2563": 2563, 
            "marker_2503": 2503, 
            "marker_2042": 2042  
        }

        st.subheader("Extracting Features for Prediction...")
        extracted_features = {}
        for marker_name, mz_value in marker_mzs.items():
            extracted_features[marker_name] = extract_marker_intensity(spectrum_df, mz_value)
        
        prediction_df = pd.DataFrame([extracted_features])
        
        if hasattr(model, 'feature_names_in_'):
            prediction_df = prediction_df[model.feature_names_in_]
        else:
            pass 

        st.write("Features extracted for prediction:")
        st.dataframe(prediction_df)

        if st.button("Predict Species"):
            try:
                prediction = model.predict(prediction_df)
                prediction_proba = model.predict_proba(prediction_df)

                predicted_species = prediction[0]
                confidence = prediction_proba[0][model.classes_ == predicted_species][0] * 100

                st.success(f"**Predicted Species:** {predicted_species}")
                st.info(f"**Confidence:** {confidence:.2f}%")

                st.subheader("All Class Probabilities:")
                prob_df = pd.DataFrame({
                    'Species': model.classes_,
                    'Probability': prediction_proba[0]
                }).sort_values(by='Probability', ascending=False)
                st.dataframe(prob_df)

            except Exception as e:
                st.error(f"Error during prediction: {e}. Please ensure your model was trained with features matching the extracted ones.")
                if hasattr(model, 'feature_names_in_'):
                    st.write("Model expected features:", model.feature_names_in_.tolist())
                st.write("Features provided for prediction:", prediction_df.columns.tolist())
    else:
        st.warning("No valid spectrum data was extracted from the uploaded file.")
