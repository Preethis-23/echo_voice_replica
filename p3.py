import sys
import os
import torch
import librosa
import numpy as np
import soundfile as sf
import simpleaudio as sa
from pathlib import Path
import matplotlib.pyplot as plt

# Set device
device = torch.device("cpu")  # Use CPU for debugging

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
vocoder_path = Path("C:/Users/preethis/PycharmProjects/echco_voice_replica/Real-Time-Voice-Cloning/vocoder/savedAla_models/default/vocoder.pt")

encoder.load_model(encoder_path).to(device)
synthesizer = Synthesizer(synthesizer_path).to(device)
vocoder.load_model(vocoder_path).to(device)

# Load and preprocess the voice sample
audio_path = "ratheesh.wav"
original_wav, sampling_rate = librosa.load(audio_path, sr=16000)
preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
sf.write("preprocessed.wav", preprocessed_wav, 16000)  # Save for inspection
embedding = encoder.embed_utterance(preprocessed_wav)
print("Embedding shape:", embedding.shape)

# Text to clone in the voice
text = "Hello Kishore, what's up? How is your life going? Haha, now the project's error has been solved. Now, to get some deep work done."

# Create spectrogram
spec = synthesizer.synthesize_spectrograms([text], [embedding])[0]
plt.imshow(spec, aspect="auto", origin="lower")
plt.savefig("spectrogram.png")
print("Spectrogram saved as spectrogram.png")

# Generate waveform
waveform = vocoder.infer_waveform(spec)
sf.write("raw_waveform.wav", waveform, 16000)  # Save raw waveform

# Apply pitch correction (optional)
waveform_corrected = librosa.effects.pitch_shift(waveform, sr=16000, n_steps=-2)  # Lower pitch slightly

# Save and play
output_path = "output.wav"
sf.write(output_path, waveform_corrected, 16000)  # Consistent sampling rate
wave_obj = sa.WaveObject.from_wave_file(output_path)
wave_obj.play().wait_done()