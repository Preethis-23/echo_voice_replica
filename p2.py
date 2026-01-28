import sys
import os
import torch
import librosa
import numpy as np
import soundfile as sf
import simpleaudio as sa
from pathlib import Path

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

encoder.load_model(encoder_path)
synthesizer = Synthesizer(synthesizer_path)
vocoder.load_model(vocoder_path)

# Load and preprocess the voice sample
audio_path = "dhoni.wav"
original_wav, sampling_rate = librosa.load(audio_path, sr=16000)
preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
embedding = encoder.embed_utterance(preprocessed_wav)

# Text to clone in the voice
text = "Hello Kishore, what's up? How is your life going? Haha,now the project's error has been solved. Now, to get some deep work done."
#



# Create spectrogram
spec = synthesizer.synthesize_spectrograms([text], [embedding])[0]

# Generate waveform
waveform = vocoder.infer_waveform(spec)

# Slow down the waveform (speech is 4x too fast, so slow it to 1/4 speed)
waveform_slowed = librosa.effects.time_stretch(waveform, rate=0.5)

# Save and play
output_path = "output.wav"
sf.write(output_path, waveform_slowed, 22050)
wave_obj = sa.WaveObject.from_wave_file(output_path)
wave_obj.play().wait_done()














"""import sys
import os
import torch
import librosa
import numpy as np
import soundfile as sf
import simpleaudio as sa
from pathlib import Path  # Add this import

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

encoder.load_model(encoder_path)
synthesizer = Synthesizer(synthesizer_path)
vocoder.load_model(vocoder_path)

# Load and preprocess the voice sample
audio_path = "sampleaudio.wav"
original_wav, sampling_rate = librosa.load(audio_path, sr=16000)
preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
embedding = encoder.embed_utterance(preprocessed_wav)

# Text to clone in the voice
text = "Hello Kishore, what's up? How is your life going? Haha, now the project's error has been solved. Now, to get some deep work done."

# Create spectrogram
spec = synthesizer.synthesize_spectrograms([text], [embedding])[0]

# Generate waveform
waveform = vocoder.infer_waveform(spec)

# Save and play
output_path = "output.wav"
sf.write(output_path, waveform, 22050)
wave_obj = sa.WaveObject.from_wave_file(output_path)
wave_obj.play().wait_done()

"""



'''import sys
import os
import torch
import librosa
import numpy as np
import soundfile as sf
import simpleaudio as sa


import os
print(os.getcwd())

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'Real-Time-Voice-Cloning')))

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/Real-Time-Voice-Cloning")
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Real-Time-Voice-Cloning", "synthesizer"))


# Add paths to Real-Time-Voice-Cloning repo
sys.path.append("C:\\Users\\preethis\\PycharmProjects\\echco_voice_replica\\Real-Time-Voice-Cloning")
sys.path.append("C:\\Users\\preethis\\PycharmProjects\\echco_voice_replica\\Real-Time-Voice-Cloning\\synthesizer")
sys.path.append("C:\\Users\\preethis\\PycharmProjects\\echco_voice_replica\\Real-Time-Voice-Cloning\\vocoder")

from encoder import inference as encoder
from synthesizer.synthesizer import Synthesizer
#from synthesizer.hparams import hparams as _syn_hp
from vocoder import inference as vocoder
# Load models
#encoder.load_model("C:/Users/preethis/PycharmProjects/echco_voice_replica/Real-Time-Voice-Cloning/encoder/saved_models/pretrained.pt")
#synthesizer = Synthesizer("C:/Users/preethis/PycharmProjects/echco_voice_replica/Real-Time-Voice-Cloning/synthesizer/saved_models/synthesizer_model.pt")
#vocoder.load_model("C:/Users/preethis/PycharmProjects/echco_voice_replica/Real-Time-Voice-Cloning/vocoder/saved_models/hifigan_model.pt")
encoder.load_model("C:/Users/preethis/PycharmProjects/echco_voice_replica/Real-Time-Voice-Cloning/encoder/saved_models/default/encoder.pt")
synthesizer = Synthesizer("C:/Users/preethis/PycharmProjects/echco_voice_replica/Real-Time-Voice-Cloning/synthesizer/saved_models/default/synthesizer.pt")
vocoder.load_model("C:/Users/preethis/PycharmProjects/echco_voice_replica/Real-Time-Voice-Cloning/vocoder/saved_models/default/vocoder.pt")

# Load and preprocess the voice sample
audio_path = "sampleaudio.wav"
original_wav, sampling_rate = librosa.load(audio_path, sr=16000)
preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
embedding = encoder.embed_utterance(preprocessed_wav)

# Text to clone in the voice
text = "Hello, this is a voice clone speaking."

# Create spectrogram
spec = synthesizer.synthesize_spectrograms([text], [embedding])[0]

# Generate waveform
waveform = vocoder.infer_waveform(spec)

# Save and play
output_path = "output.wav"
sf.write(output_path, waveform, 22050)
wave_obj = sa.WaveObject.from_wave_file(output_path)
wave_obj.play().wait_done()
'''