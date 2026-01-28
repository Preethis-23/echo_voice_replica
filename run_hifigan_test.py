from pathlib import Path
import numpy as np
import soundfile as sf

repo = Path('Real-Time-Voice-Cloning')
import sys
sys.path.insert(0, str(repo))
mel = np.load('outputs/test_mel.npy')
print('mel shape', mel.shape)
from hifigan import inference as hifi
hifi_path = repo / 'saved_models' / 'default' / 'hifigan.pt'
print('hifi exists', hifi_path.exists())
try:
    hifi.load_model(hifi_path)
    y = hifi.infer_waveform(mel)
    y = y / max(1e-9, abs(y).max()) * 0.95
    sf.write('outputs/test_hifigan.wav', y.astype('float32'), 16000)
    print('Saved outputs/test_hifigan.wav')
except Exception as e:
    print('HiFi-GAN inference failed:', e)
