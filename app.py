import streamlit as st
import os
import numpy as np
import librosa           # <--- This is the missing piece!
import soundfile as sf
import tensorflow as tf

# Fix for Mac blank screen bug
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 255
HOP_LEN = 125
CHUNK_WIDTH = 128

NOISE_CLASSES = [
    'AirConditioner', 'CopyMachine', 'Munching', 
    'NeighborSpeaking', 'SqueakyChair', 'Typing', 
    'VacuumCleaner', 'WasherDryer'
]

STRATEGIES = {
    'transient': 0.15,  
    'stationary': 0.02, 
    'standard': 0.05    
}

@st.cache_resource
def load_brains():
    try:
        classifier = tf.keras.models.load_model('best_classifier.keras')
        denoiser = tf.keras.models.load_model('best_mrdnn.keras')
        return classifier, denoiser
    except Exception as e:
        st.error(f"Error loading models: {e}. Make sure the .keras files are in the same folder.")
        st.stop()

classifier_model, denoiser_model = load_brains()

def predict_context(audio):
    """Brain 1: Analyzes noise type from the first second of audio."""
    target_len = SAMPLE_RATE
    
    # FIX: Take the FIRST second of the audio instead of the middle
    if len(audio) > target_len:
        snapshot = audio[:target_len] 
    else:
        snapshot = np.pad(audio, (0, target_len - len(audio)))

    melspec = librosa.feature.melspectrogram(y=snapshot, sr=SAMPLE_RATE, n_mels=N_MELS)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    norm_spec = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min() + 1e-10)
    
    input_tensor = norm_spec[np.newaxis, ..., np.newaxis]
    probs = classifier_model.predict(input_tensor, verbose=0)
    
    class_idx = np.argmax(probs)
    return NOISE_CLASSES[class_idx], np.max(probs) * 100

def clean_audio(audio, noise_class):
    """Brain 2: Removes noise using the Alpha Leash."""
    if noise_class in ['Typing', 'SqueakyChair', 'Munching']:
        alpha = STRATEGIES['transient']
    elif noise_class in ['AirConditioner', 'VacuumCleaner']:
        alpha = STRATEGIES['stationary']
    else:
        alpha = STRATEGIES['standard']

    chunk_samples = CHUNK_WIDTH * HOP_LEN
    padded_audio = np.pad(audio, (0, chunk_samples - (len(audio) % chunk_samples)))
    
    full_spec = librosa.stft(padded_audio, n_fft=N_FFT, hop_length=HOP_LEN)
    full_mag, full_phase = librosa.magphase(full_spec)
    
    num_chunks = full_mag.shape[1] // CHUNK_WIDTH
    predicted_mags = []
    
    for i in range(num_chunks):
        seg_mag = full_mag[:, i*CHUNK_WIDTH : (i+1)*CHUNK_WIDTH]
        log_mag = np.log1p(seg_mag)
        input_chunk = log_mag.T[np.newaxis, ..., np.newaxis]
        
        pred = denoiser_model.predict(input_chunk, verbose=0)
        predicted_mags.append(np.expm1(pred.squeeze().T))
        
    predicted_full_mag = np.hstack(predicted_mags)
    
    crop_len = predicted_full_mag.shape[1]
    noisy_mag_crop = full_mag[:, :crop_len]
    phase_crop = full_phase[:, :crop_len]
    
    mask = predicted_full_mag / (noisy_mag_crop + 1e-10)
    mask = np.clip(mask, alpha, 1.0) 
    
    clean_spec_complex = (noisy_mag_crop * mask) * phase_crop
    clean_audio = librosa.istft(clean_spec_complex, hop_length=HOP_LEN)
    
    return clean_audio, alpha

def plot_spectrogram(audio, title):
    """Helper to draw the spectrograms."""
    fig, ax = plt.subplots(figsize=(6, 2))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=SAMPLE_RATE, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    plt.title(title)
    st.pyplot(fig)

# --- 3. FRONTEND UI ---
st.title("🎙️ Context-Aware Audio Denoiser")
st.write("Upload a noisy audio file or record directly from your microphone.")

# Use tabs to organize the inputs
tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Record Microphone"])

raw_audio_bytes = None

with tab1:
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])
    if uploaded_file is not None:
        raw_audio_bytes = uploaded_file.read()

with tab2:
    st.info("Ensure your browser has permission to access the microphone.")
    recorded_audio = st.audio_input("Record your voice (with some background noise!)")
    if recorded_audio is not None:
        raw_audio_bytes = recorded_audio.read()

# --- 4. EXECUTION FLOW ---
if raw_audio_bytes is not None:
    st.divider()
    st.subheader("1. Original Audio")
    
    # Save temp file for librosa to read
    with open("temp_input.wav", "wb") as f:
        f.write(raw_audio_bytes)
        
    audio, sr = librosa.load("temp_input.wav", sr=SAMPLE_RATE, mono=True)
    st.audio("temp_input.wav")
    plot_spectrogram(audio, "Input Spectrogram (Noisy)")

    if st.button("🚀 Process & Clean Audio", type="primary"):
        with st.spinner("Brain 1: Analyzing context..."):
            noise_type, conf = predict_context(audio)
            st.success(f"**Detected Context:** {noise_type} ({conf:.1f}% confidence)")
            
        with st.spinner(f"Brain 2: Denoising using {noise_type} strategy..."):
            cleaned_audio, alpha_used = clean_audio(audio, noise_type)
            sf.write("temp_clean.wav", cleaned_audio, SAMPLE_RATE)
            
        st.subheader("2. Cleaned Audio")
        st.info(f"**Strategy Applied:** The U-Net mask was leashed to a minimum of {int(alpha_used * 100)}% noise retention ($\alpha$ = {alpha_used}) to protect speech.")
        st.audio("temp_clean.wav")
        plot_spectrogram(cleaned_audio, "Output Spectrogram (Clean)")
        
        with open("temp_clean.wav", "rb") as file:
            st.download_button("💾 Download Clean Audio", data=file, file_name=f"Cleaned_{noise_type}.wav", mime="audio/wav")
