"""
Run a comparison between Griffin-Lim and WaveRNN (vocoder) outputs for a given reference audio and text.
Usage:
    python run_synthesis_test.py path/to/ref.wav "Text to synthesize"
Outputs:
    outputs/test_vocoder.wav
    outputs/test_griffin.wav
    outputs/test_mel.npy
"""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "Real-Time-Voice-Cloning"))

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

# paths
models_dir = repo_root / "Real-Time-Voice-Cloning" / "saved_models" / "default"
encoder_path = models_dir / "encoder.pt"
synth_path = models_dir / "synthesizer.pt"
voc_path = models_dir / "vocoder.pt"

out_dir = repo_root / "outputs"
out_dir.mkdir(exist_ok=True)

if len(sys.argv) < 3:
    print("Usage: python run_synthesis_test.py <ref.wav> <text>")
    sys.exit(1)

ref = Path(sys.argv[1])
text = sys.argv[2]

# Load models
encoder.load_model(encoder_path)
synthesizer = Synthesizer(synth_path)
vocoder.load_model(voc_path)

# Load reference and preprocess
import librosa
wav, sr = librosa.load(str(ref), sr=None)
print(f"Original sr={sr}, duration={len(wav)/sr:.2f}s")
preprocessed = encoder.preprocess_wav(wav, source_sr=sr, normalize=True, trim_silence=False)
print(f"Preprocessed duration: {len(preprocessed)/encoder.sampling_rate:.2f}s")

# Embedding
embed = encoder.embed_utterance(preprocessed)
print('Embedding computed, norm:', np.linalg.norm(embed))

# Synthesizer -> mel
mels = synthesizer.synthesize_spectrograms([text], [embed])
mel = mels[0]
print('Mel shape:', mel.shape, 'min/max:', mel.min(), mel.max())
np.save(out_dir / 'test_mel.npy', mel)

# Griffin-Lim (reference)
print('Generating Griffin-Lim audio...')
griffin_wav = synthesizer.griffin_lim(mel)
griffin_wav = griffin_wav / max(1e-9, np.abs(griffin_wav).max()) * 0.95
sf.write(out_dir / 'test_griffin.wav', griffin_wav.astype(np.float32), synthesizer.sample_rate)

# Vocoder
print('Generating WaveRNN vocoder audio from synthesized mel...')
voc_wav = vocoder.infer_waveform(mel)
voc_wav = voc_wav / max(1e-9, np.abs(voc_wav).max()) * 0.95
sf.write(out_dir / 'test_vocoder_from_synth.wav', voc_wav.astype(np.float32), synthesizer.sample_rate)

# Also test vocoder on real mel spectrogram derived from the reference audio
print('Generating WaveRNN vocoder audio from real mel (reference) ...')
mel_real = synthesizer.make_spectrogram(preprocessed)
print('Real mel shape:', mel_real.shape, 'min/max:', mel_real.min(), mel_real.max())
voc_wav_real = vocoder.infer_waveform(mel_real)
voc_wav_real = voc_wav_real / max(1e-9, np.abs(voc_wav_real).max()) * 0.95
sf.write(out_dir / 'test_vocoder_from_real.wav', voc_wav_real.astype(np.float32), synthesizer.sample_rate)

print('Saved outputs in', out_dir)
print('Done')