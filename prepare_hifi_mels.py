import numpy as np, os
m = np.load('outputs/test_mel.npy')
print('orig shape', m.shape, 'dtype', m.dtype)
# ensure shape is (1, num_mels, T)
if m.ndim == 2 and m.shape[0] == 80:
    x = m[np.newaxis, :, :]
elif m.ndim == 2 and m.shape[1] == 80:
    x = m.T[np.newaxis, :, :]
else:
    x = m
os.makedirs('Real-Time-Voice-Cloning/hifi_mels', exist_ok=True)
np.save('Real-Time-Voice-Cloning/hifi_mels/test_mel.npy', x)
print('saved to Real-Time-Voice-Cloning/hifi_mels/test_mel.npy, shape', x.shape)
