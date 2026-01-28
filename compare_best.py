import soundfile as sf
import numpy as np
from scipy.signal import stft

g, sr = sf.read('outputs/test_griffin.wav')
b, sr2 = sf.read('outputs/test_vocoder_best.wav')
L = max(len(g), len(b))
g2 = np.pad(g, (0, L - len(g)))
b2 = np.pad(b, (0, L - len(b)))
print('corr', np.corrcoef(g2, b2)[0,1])
psig = np.mean(g2**2)
pnoise = np.mean((g2 - b2)**2)
print('psig/pnoise (dB):', 10*np.log10(psig/(pnoise+1e-12)))
f, t, Zg = stft(g2, fs=sr)
f, t, Zb = stft(b2, fs=sr)
cent_g = (np.sum((f[:,None]*np.abs(Zg)), axis=0) / (np.sum(np.abs(Zg), axis=0)+1e-12)).mean()
cent_b = (np.sum((f[:,None]*np.abs(Zb)), axis=0) / (np.sum(np.abs(Zb), axis=0)+1e-12)).mean()
print('centroid (griffin,best):', cent_g, cent_b)
