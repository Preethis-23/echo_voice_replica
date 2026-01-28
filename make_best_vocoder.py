import numpy as np
import soundfile as sf
from pathlib import Path
import sys
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / 'Real-Time-Voice-Cloning'))
from synthesizer.inference import Synthesizer

out = Path('outputs')
mel = np.load(out / 'test_mel.npy')
models_dir = Path('Real-Time-Voice-Cloning') / 'saved_models' / 'default'
from vocoder import inference as vocoder
from encoder import inference as encoder
encoder.load_model(models_dir / 'encoder.pt')
synth = Synthesizer(models_dir / 'synthesizer.pt')
vocoder.load_model(models_dir / 'vocoder.pt')

# Best candidate from grid search
target = 8000
overlap = 200
wav = vocoder.infer_waveform(mel, target=target, overlap=overlap)
# apply light lowpass
try:
    from scipy.signal import butter, filtfilt
    b,a = butter(4, 7000/(0.5*synth.sample_rate), btype='low')
    wav = filtfilt(b,a,wav)
except Exception:
    pass
wav = wav / max(1e-9, abs(wav).max()) * 0.95
sf.write(out / 'test_vocoder_best.wav', wav.astype('float32'), synth.sample_rate)
print('Saved outputs/test_vocoder_best.wav')