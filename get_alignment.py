import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "Real-Time-Voice-Cloning"))

from encoder import inference as encoder
from synthesizer.inference import Synthesizer

if len(sys.argv) < 3:
    print('Usage: python get_alignment.py <ref.wav> <text>')
    sys.exit(1)

ref = sys.argv[1]
text = sys.argv[2]

models_dir = repo_root / 'Real-Time-Voice-Cloning' / 'saved_models' / 'default'
encoder.load_model(models_dir / 'encoder.pt')
synth = Synthesizer(models_dir / 'synthesizer.pt')

import librosa
wav, sr = librosa.load(ref, sr=None)
pre = encoder.preprocess_wav(wav, source_sr=sr, trim_silence=False)
emb = encoder.embed_utterance(pre)

specs, aligns = synth.synthesize_spectrograms([text], [emb], return_alignments=True)
mel = specs[0]
align = aligns
print('mel shape', mel.shape)
if align is not None:
    print('align shape', align.shape)
    align_np = align.detach().cpu().numpy()
    if align_np.ndim == 3:
        align_np = align_np[0]
    np.save('alignment.npy', align_np)
    plt.figure(figsize=(8,6))
    plt.imshow(align_np, origin='lower', aspect='auto')
    plt.colorbar()
    plt.title('Alignment')
    plt.savefig('alignment.png')
    print('Saved alignment.png and alignment.npy')
else:
    print('No alignment returned')
