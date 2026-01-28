"""
Grid-search WaveRNN inference params and postprocessing to find the best settings.
Compares outputs to Griffin-Lim using psig/pnoise dB and spectral centroid match.
Usage:
    python grid_vocoder_search.py
Reads `outputs/test_mel.npy` (generated previously by run_synthesis_test.py).
Saves results to `outputs/grid_results.csv` and writes candidate wavs to `outputs/`.
"""
import numpy as np
import soundfile as sf
from pathlib import Path
# Ensure Real-Time-Voice-Cloning package is importable
import sys
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / 'Real-Time-Voice-Cloning'))
from vocoder import inference as vocoder
from synthesizer.inference import Synthesizer
from scipy.signal import stft

out = Path('outputs')
mel_path = out / 'test_mel.npy'
if not mel_path.exists():
    print('No mel found at', mel_path, 'Run run_synthesis_test.py first')
    raise SystemExit(1)

mel = np.load(mel_path)
# ensure mel shape (n_mels, time)
print('mel shape', mel.shape)

# Load models (vocoder must be explicitly loaded)
models_dir = Path('Real-Time-Voice-Cloning') / 'saved_models' / 'default'
from encoder import inference as encoder
encoder.load_model(models_dir / 'encoder.pt')
synth = Synthesizer(models_dir / 'synthesizer.pt')
from vocoder import inference as vocoder
vocoder.load_model(models_dir / 'vocoder.pt')

# Generate griffin for reference
griffin = synth.griffin_lim(mel)
griffin = griffin / max(1e-9, np.abs(griffin).max()) * 0.95
sf.write(out / 'grid_ref_griffin.wav', griffin.astype(np.float32), synth.sample_rate)

candidates = []
for target in [4000, 8000, 12000, 16000]:
    for overlap in [200, 400, 800]:
        for lowpass in [False, True]:
            candidates.append((target, overlap, lowpass))

results = []
for (target, overlap, lowpass) in candidates:
    print('Testing', target, overlap, 'lowpass=', lowpass)
    wav = vocoder.infer_waveform(mel, target=target, overlap=overlap)
    # optional lowpass
    if lowpass:
        from scipy.signal import butter, filtfilt
        b,a = butter(4, 7000/(0.5*synth.sample_rate), btype='low')
        try:
            wav = filtfilt(b,a,wav)
        except Exception:
            pass
    wav = wav / max(1e-9, np.abs(wav).max()) * 0.95
    fname = out / f'grid_voc_t{target}_o{overlap}_lp{int(lowpass)}.wav'
    sf.write(fname, wav.astype(np.float32), synth.sample_rate)

    # compute metrics vs griffin: correlation and pnoise
    L = max(len(griffin), len(wav))
    g = np.pad(griffin, (0, L-len(griffin)))
    v = np.pad(wav, (0, L-len(wav)))
    corr = np.corrcoef(g, v)[0,1]
    psig = np.mean(g**2)
    pnoise = np.mean((g - v)**2)
    snr_db = 10 * np.log10(psig / (pnoise + 1e-12))
    f, t, Z = stft(v, fs=synth.sample_rate, nperseg=1024)
    centroid = (np.sum((f[:,None]*np.abs(Z)), axis=0) / (np.sum(np.abs(Z), axis=0)+1e-12)).mean()
    results.append((target, overlap, lowpass, corr, snr_db, centroid))

# Save CSV
import csv
with open(out / 'grid_results.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['target','overlap','lowpass','corr','snr_db','centroid'])
    for r in results:
        w.writerow(r)

print('Done. Results written to outputs/grid_results.csv and candidate WAVs in outputs/')
