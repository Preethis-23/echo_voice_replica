import numpy as np
mel = np.load('outputs/test_mel.npy')
print('mel shape', mel.shape)
energy = mel.mean(axis=0)
print('energy min/max', energy.min(), energy.max())
low = np.where(energy < (energy.max()*0.05))[0]
print('low energy indices (first 20):', low[:20])
print('energy tail mean (last 10 frames):', energy[-10:].mean())
print('energy first 10', energy[:10])
