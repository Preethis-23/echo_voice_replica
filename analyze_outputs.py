import numpy as np
import soundfile as sf

files = ['outputs/test_griffin.wav', 'outputs/test_vocoder_from_synth.wav', 'outputs/test_vocoder_from_real.wav']

from scipy.signal import stft

for f in files:
    w, sr = sf.read(f)
    print(f) 
    print('  len', len(w), 'peak', float(np.max(w)), 'min', float(np.min(w)), 'std', float(np.std(w)))

# Compare vocoder_from_real to griffin
wg, _ = sf.read(files[0])
wv_real, _ = sf.read(files[2])
L = max(len(wg), len(wv_real))
wg2 = np.pad(wg, (0,L-len(wg)))
wv2 = np.pad(wv_real, (0,L-len(wv_real)))
print('\nComparison: griffin vs vocoder_from_real')
print('  corr', np.corrcoef(wg2, wv2)[0,1])
psig = np.mean(wg2**2)
pnoise = np.mean((wg2-wv2)**2)
print('  psig/pnoise (dB):', 10*np.log10(psig/(pnoise+1e-12)))

# Compare vocoder_from_synth to griffin
wv_synth, _ = sf.read(files[1])
wv2s = np.pad(wv_synth, (0,L-len(wv_synth)))
print('\nComparison: griffin vs vocoder_from_synth')
print('  corr', np.corrcoef(wg2, wv2s)[0,1])
psig = np.mean(wg2**2)
pnoise = np.mean((wg2-wv2s)**2)
print('  psig/pnoise (dB):', 10*np.log10(psig/(pnoise+1e-12)))

# Spectral centroid means
f_g, t_g, Zg = stft(wg2, fs=sr)
centroid_g = (np.sum((f_g[:,None]*np.abs(Zg)), axis=0) / (np.sum(np.abs(Zg), axis=0)+1e-12)).mean()
f_vr, t_vr, Zvr = stft(wv2, fs=sr)
centroid_vr = (np.sum((f_vr[:,None]*np.abs(Zvr)), axis=0) / (np.sum(np.abs(Zvr), axis=0)+1e-12)).mean()
print('\nSpectral centroid mean (griffin / vocoder_from_real):', centroid_g, centroid_vr)
