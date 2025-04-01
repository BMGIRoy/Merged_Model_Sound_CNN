
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import zipfile
import tempfile
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import io

st.set_page_config(page_title="Motor Sound CNN Classifier", layout="centered")

# ------------------ Branding ------------------
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Logo_BMGI.svg/2560px-Logo_BMGI.svg.png", width=200)
st.markdown("""
# 🧠 Motor Sound Classifier (CNN)
Upload `.wav` motor sounds to predict Normal or Abnormal conditions using a CNN.
You can also train a custom model using your own recordings.
""")

# ------------------ Utilities ------------------
def extract_mel_spectrogram(file, sr=22050):
    y, sr = librosa.load(file, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

def plot_spectrogram(S_DB):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_DB, x_axis='time', y_axis='mel', ax=ax)
    ax.set(title='Mel-frequency spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# ------------------ CNN Model ------------------
def build_cnn_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ------------------ App UI ------------------
tabs = st.tabs(["🔍 Predict", "⚙️ Train Custom Model"])

# Store model in session
if 'cnn_model' not in st.session_state:
    st.session_state['cnn_model'] = None
    st.session_state['input_shape'] = None

with tabs[0]:
    st.subheader("🔍 Predict Normal or Abnormal")
    uploaded_file = st.file_uploader("Upload a .wav file", type="wav")
    if uploaded_file and st.session_state['cnn_model'] is not None:
        try:
            S_DB = extract_mel_spectrogram(uploaded_file)
            spec_img = plot_spectrogram(S_DB)
            st.image(spec_img, caption="Generated Mel-Spectrogram", use_column_width=True)
            img = np.expand_dims(S_DB, axis=-1)
            img = tf.image.resize(img, st.session_state['input_shape'][:2])
            img = np.expand_dims(img, axis=0)
            pred = st.session_state['cnn_model'].predict(img)[0][0]
            label = "Abnormal" if pred > 0.5 else "Normal"
            confidence = float(pred if pred > 0.5 else 1 - pred)
            st.success(f"🧠 Prediction: **{label}**")
            st.write(f"Confidence: {confidence:.2f}")
        except Exception as e:
            st.error(f"Failed to process audio: {e}")
    elif uploaded_file:
        st.warning("Please train a model first using the Train tab.")

with tabs[1]:
    st.subheader("⚙️ Train Custom CNN Model")
    st.write("Upload a ZIP with folders `/normal/` and `/abnormal/` each containing `.wav` files.")
    training_zip = st.file_uploader("Upload Training ZIP", type="zip")

    if training_zip:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(training_zip, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            data = []
            labels = []
            for label_folder, class_label in [('normal', 0), ('abnormal', 1)]:
                folder_path = os.path.join(tmpdir, label_folder)
                if os.path.exists(folder_path):
                    for fname in os.listdir(folder_path):
                        if fname.endswith('.wav'):
                            file_path = os.path.join(folder_path, fname)
                            try:
                                spec = extract_mel_spectrogram(file_path)
                                data.append(spec)
                                labels.append(class_label)
                            except Exception as e:
                                st.warning(f"Skipped {fname}: {e}")

            if len(data) >= 10:
                X = np.array(data)
                y = np.array(labels)
                X = np.expand_dims(X, -1)
                input_shape = X.shape[1:]
                st.session_state['input_shape'] = input_shape

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = build_cnn_model(input_shape)
                model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

                loss, acc = model.evaluate(X_test, y_test, verbose=0)
                st.success(f"✅ Model trained with accuracy: {acc:.2f}")
                st.session_state['cnn_model'] = model

                model.save("cnn_model.h5")
                with open("cnn_model.h5", "rb") as f:
                    st.download_button("⬇️ Download Trained Model (.h5)", f, file_name="cnn_model.h5")
            else:
                st.warning("Need at least 10 samples (5 normal, 5 abnormal) to train a model.")
