"""
Deep diagnostics for voice cloning pipeline.
Saves:
 - spectrogram image (synthesized mel)
 - alignment heatmap (if returned)
 - mel numpy
 - Griffin-Lim audio
 - WaveRNN audio with several parameter configs
 - Split audio segments (first/mid/last 2s)
 - Plots of waveform and short-time energy
Usage:
    python deep_diagnose.py <ref.wav> "Text to synth"
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "Real-Time-Voice-Cloning"))

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from encoder.params_data import sampling_rate as enc_sr

out_dir = repo_root / "diagnostics"
out_dir.mkdir(exist_ok=True)

if len(sys.argv) < 3:
    print("Usage: python deep_diagnose.py <ref.wav> <text>")
    sys.exit(1)

ref = Path(sys.argv[1])
text = sys.argv[2]

# load models
models_dir = repo_root / "Real-Time-Voice-Cloning" / "saved_models" / "default"
encoder.load_model(models_dir / 'encoder.pt')
synthesizer = Synthesizer(models_dir / 'synthesizer.pt')
vocoder.load_model(models_dir / 'vocoder.pt')

# load and preprocess reference
import librosa
wav, sr = librosa.load(str(ref), sr=None)
print('orig sr', sr, 'len', len(wav)/sr)
pre = encoder.preprocess_wav(wav, source_sr=sr, normalize=True, trim_silence=False)
print('pre len', len(pre)/enc_sr)

# embed
emb = encoder.embed_utterance(pre)

# synthesize (get alignments)
specs, alignments = synthesizer.synthesize_spectrograms([text], [emb], return_alignments=True)
mel = specs[0]
align = alignments[0] if alignments is not None and len(alignments)>0 else None
print('mel shape', mel.shape)
np.save(out_dir / 'mel.npy', mel)

# save mel image
plt.figure(figsize=(10,4))
plt.imshow(mel, aspect='auto', origin='lower')
plt.colorbar()
plt.title('Synthesized mel')
plt.tight_layout()
plt.savefig(out_dir / 'mel.png')
plt.close()

# save alignment heatmap
if align is not None:
    plt.figure(figsize=(8,6))
    plt.imshow(align, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Alignment')
    plt.tight_layout()
    plt.savefig(out_dir / 'alignment.png')
    plt.close()
else:
    print('No alignments returned')

# compute mel energy envelope
mel_energy = mel.mean(axis=0)
plt.figure()
plt.plot(mel_energy)
plt.title('Mel mean energy per frame')
plt.savefig(out_dir / 'mel_energy.png')
plt.close()

# Griffin-Lim audio
g = synthesizer.griffin_lim(mel)
g = g / max(1e-9, np.abs(g).max()) * 0.95
sf.write(out_dir / 'griffin.wav', g.astype(np.float32), synthesizer.sample_rate)

# Try WaveRNN with multiple (target, overlap) settings
params = [(8000,400),(4000,200),(16000,800)]
for target, overlap in params:
    w = vocoder.infer_waveform(mel, target=target, overlap=overlap)
    w = w / max(1e-9, np.abs(w).max()) * 0.95
    fname = out_dir / f'vocoder_t{target}_o{overlap}.wav'
    sf.write(fname, w.astype(np.float32), synthesizer.sample_rate)

# Also generate WaveRNN from "real" mel (from reference) to see generator fidelity on real mels
real_mel = synthesizer.make_spectrogram(pre)
rmel = real_mel
sf.write(out_dir / 'real_mel.npy', rmel.astype(np.float32))
vg = vocoder.infer_waveform(rmel)
vg = vg / max(1e-9, np.abs(vg).max()) * 0.95
sf.write(out_dir / 'vocoder_from_real.wav', vg.astype(np.float32), synthesizer.sample_rate)

# Save waveform plots and split into segments
for name in ['griffin.wav'] + [f'vocoder_t{t}_o{o}.wav' for t,o in params] + ['vocoder_from_real.wav']:
    p = out_dir / name
    y, sr = sf.read(p)
    t = np.arange(len(y))/sr
    plt.figure(figsize=(10,3))
    plt.plot(t, y)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(out_dir / f'{Path(name).stem}_wave.png')
    plt.close()
    # segments 0-2s, mid 2-4s, last 2s
    segs = []
    total = len(y)
    seg_samples = int(2*sr)
    segs.append(y[:seg_samples])
    if total > 4*sr:
        segs.append(y[2*sr:4*sr])
        segs.append(y[-seg_samples:])
    else:
        segs.append(y[seg_samples:seg_samples*2])
        segs.append(y[-seg_samples:])
    for i, s in enumerate(segs):
        sf.write(out_dir / f'{Path(name).stem}_seg{i+1}.wav', s.astype(np.float32), sr)

print('Diagnostics written to', out_dir)
print('Done')
