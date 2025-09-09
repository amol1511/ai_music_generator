import streamlit as st
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import scipy.io.wavfile
import tempfile
import os

st.set_page_config(page_title="🎵 AI Music Generator", page_icon="🎶", layout="centered")
st.title("🎶 AI Music Generator")
st.write("Enter a prompt and generate AI music!")

# Load model
@st.cache_resource
def load_model():
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    return model, processor

model, processor = load_model()

# User Input
prompt = st.text_input("Enter your music description:", "A calm lo-fi beat with piano and soft drums")
duration = st.slider("Duration (seconds)", 5, 30, 10)

if st.button("🎵 Generate Music"):
    with st.spinner("Generating your music..."):
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        )

        audio_values = model.generate(**inputs, max_new_tokens=duration * 50)

        # Save audio
        sample_rate = model.config.audio_encoder.sampling_rate
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            scipy.io.wavfile.write(fp.name, rate=sample_rate, data=audio_values[0, 0].cpu().numpy())
            audio_path = fp.name

        st.success("✅ Music generated!")
        st.audio(audio_path, format="audio/wav")
        with open(audio_path, "rb") as f:
            st.download_button("Download Music", f, file_name="music.wav", mime="audio/wav")
