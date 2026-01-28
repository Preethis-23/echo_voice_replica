import sys
import os
import torch
import librosa
import numpy as np
import soundfile as sf
import simpleaudio as sa
from pathlib import Path
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# Print current working directory
print("Current Working Directory:", os.getcwd())

# Add paths to the Real-Time-Voice-Cloning repo
base_path = "C:/Users/preethis/PycharmProjects/echco_voice_replica/Real-Time-Voice-Cloning"
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "synthesizer"))
sys.path.append(os.path.join(base_path, "vocoder"))

# Verify sys.path and directories
print("sys.path:", sys.path)
print("Real-Time-Voice-Cloning contents:", os.listdir(base_path))
print("Synthesizer contents:", os.listdir(os.path.join(base_path, "synthesizer")))
print("Vocoder contents:", os.listdir(os.path.join(base_path, "vocoder")))

# Test synthesizer.hparams import
try:
    from synthesizer.hparams import hparams
    print("Successfully imported synthesizer.hparams")
except ImportError as e:
    print("Failed to import synthesizer.hparams:", e)
    sys.exit(1)

try:
    from encoder import inference as encoder
    from synthesizer.inference import Synthesizer
    from vocoder import inference as vocoder
except ImportError as e:
    print("Import error:", e)
    sys.exit(1)

# Load models with Path objects
encoder_path = Path("C:/Users/preethis/PycharmProjects/echco_voice_replica/Real-Time-Voice-Cloning/encoder/saved_models/default/encoder.pt")
synthesizer_path = Path("C:/Users/preethis/PycharmProjects/echco_voice_replica/Real-Time-Voice-Cloning/synthesizer/saved_models/default/synthesizer.pt")
vocoder_path = Path("C:/Users/preethis/PycharmProjects/echco_voice_replica/Real-Time-Voice-Cloning/vocoder/saved_models/default/vocoder.pt")

try:
    encoder.load_model(encoder_path)
    synthesizer = Synthesizer(synthesizer_path)
    vocoder.load_model(vocoder_path)
except Exception as e:
    print("Error loading models:", e)
    sys.exit(1)

# Load and preprocess the voice sample
audio_path = "dhoni.wav"
try:
    wav, sr = librosa.load(audio_path, sr=16000)
except Exception as e:
    print("Error loading dhoni.wav:", e)
    sys.exit(1)

# Preprocess input: Trim silence and normalize
wav_trimmed, _ = librosa.effects.trim(wav, top_db=20)  # Remove silence
wav_normalized = wav_trimmed / np.max(np.abs(wav_trimmed)) * 0.9 if np.max(np.abs(wav_trimmed)) > 0 else wav_trimmed

# Save preprocessed audio for reference
cleaned_path = "cleaned_dhoni.wav"
sf.write(cleaned_path, wav_normalized, 16000)
print(f"Saved preprocessed audio to: {cleaned_path}")

# Encode preprocessed audio
preprocessed_wav = encoder.preprocess_wav(wav_normalized, 16000)
embedding = encoder.embed_utterance(preprocessed_wav)

# Text to clone in the voice
text = "Hello Kishore, what's up? How is your life going? Haha, now the project's error has been solved. Now, to get some deep work done."

# Create spectrogram
spec = synthesizer.synthesize_spectrograms([text], [embedding])[0]

# Save spectrogram as image for debugging
plt.figure(figsize=(10, 4))
plt.imshow(spec, aspect='auto', origin='lower', interpolation='none')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-Spectrogram')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.savefig('spectrogram.png')
plt.close()
print(f"Saved spectrogram to: spectrogram.png")

# Generate waveform with adjusted parameters
waveform = vocoder.infer_waveform(spec, target=12000, overlap=600)

# Apply high-pass filter to reduce bass and boost speech frequencies
def highpass_filter(data, cutoff=800, fs=22050, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y

waveform_filtered = highpass_filter(waveform, cutoff=800, fs=22050)

# Slow down the waveform to natural tempo
waveform_slowed = librosa.effects.time_stretch(waveform_filtered, rate=0.3)

# Correct pitch to counteract time-stretching effect (shift up by ~1.5 octaves)
waveform_corrected = librosa.effects.pitch_shift(waveform_slowed, sr=22050, n_steps=18)

# Normalize waveform for audibility
waveform_normalized = waveform_corrected / np.max(np.abs(waveform_corrected)) * 0.8 if np.max(np.abs(waveform_corrected)) > 0 else waveform_corrected

# Save and play
output_path = "output.wav"
sf.write(output_path, waveform_normalized, 22050)
print(f"Saved output to: {output_path}")
try:
    wave_obj = sa.WaveObject.from_wave_file(output_path)
    wave_obj.play().wait_done()
except Exception as e:
    print("Error playing output.wav:", e)