# Motor Sound Classifier (CNN)

This Streamlit app lets users:
- Upload `.wav` motor sound files to classify as **Normal** or **Abnormal**
- Train their own custom CNN model by uploading a ZIP file with labeled audio data
- Visualize Mel-spectrograms
- Download the trained model (`.h5`) for reuse

## How to Use

### 🔍 Predict
1. Go to the **Predict** tab
2. Upload a `.wav` file
3. View the classification and spectrogram

### ⚙️ Train Custom Model
1. Upload a ZIP with:
    - `/normal/` folder containing normal `.wav` files
    - `/abnormal/` folder containing abnormal `.wav` files
2. The app will train a CNN and use it for all future predictions
3. Option to download the trained model

## Requirements
See `requirements.txt`
