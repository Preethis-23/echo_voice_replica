"""
Diagnostic script to inspect reference audio preprocessing and encoder behavior.
Usage:
    python diagnose_audio.py path/to/file.wav

Prints:
- Original sampling rate and duration
- Post-preprocess duration (after VAD/trimming)
- Number of partial slices used to compute embedding
- Embedding shape and L2 norm
- Recommendation whether audio is too short
"""

import sys
from pathlib import Path

# Ensure we can import the Real-Time-Voice-Cloning package
repo_dir = Path(__file__).resolve().parent
rtvc_dir = repo_dir / "Real-Time-Voice-Cloning"
if str(rtvc_dir) not in sys.path:
    sys.path.insert(0, str(rtvc_dir))

import numpy as np
import librosa
from encoder import inference as encoder
from encoder.params_data import sampling_rate


def seconds(samples, sr):
    return float(samples) / float(sr)


def diagnose(wpath):
    wpath = Path(wpath)
    if not wpath.exists():
        print(f"File not found: {wpath}")
        return

    # Load original file
    orig_wav, orig_sr = librosa.load(str(wpath), sr=None)
    print(f"Original sr: {orig_sr}, duration: {seconds(len(orig_wav), orig_sr):.3f}s, frames: {len(orig_wav)}")

    # Resample to encoder sampling_rate for comparison
    try:
        wav = librosa.resample(orig_wav, orig_sr=orig_sr, target_sr=sampling_rate)
    except TypeError:
        # fallback for older versions
        wav = librosa.resample(orig_wav, orig_sr, sampling_rate)
    print(f"Resampled to {sampling_rate} Hz, duration: {seconds(len(wav), sampling_rate):.3f}s")

    # Preprocess using the encoder's function
    preprocessed = encoder.preprocess_wav(wav, source_sr=sampling_rate, normalize=True, trim_silence=True)
    print(f"After encoder.preprocess_wav (normalize=True, trim_silence=True): duration: {seconds(len(preprocessed), sampling_rate):.3f}s, samples: {len(preprocessed)}")

    preprocessed_raw = encoder.preprocess_wav(wav, source_sr=sampling_rate, normalize=True, trim_silence=False)
    print(f"After encoder.preprocess_wav (normalize=True, trim_silence=False): duration: {seconds(len(preprocessed_raw), sampling_rate):.3f}s, samples: {len(preprocessed_raw)}")

    # Compute partials
    from encoder.inference import compute_partial_slices
    wave_slices, mel_slices = compute_partial_slices(len(preprocessed))
    print(f"Computed partials: {len(wave_slices)} partial(s), partial samples: {wave_slices[0].stop - wave_slices[0].start if wave_slices else 0}")

    # Compute embedding
    if len(preprocessed) == 0:
        print("Preprocessed audio is empty after trimming. No embedding can be produced.")
        return

    if not encoder.is_loaded():
        # Load default encoder if available
        try:
            encoder.load_model(Path('Real-Time-Voice-Cloning/saved_models/default/encoder.pt'))
        except Exception as e:
            print("Could not load encoder default model. Please supply encoder model file.", e)

    try:
        emb = encoder.embed_utterance(preprocessed)
        print(f"Embedding shape: {emb.shape}, L2 norm: {np.linalg.norm(emb):.6f}")
    except Exception as e:
        print("Skipping embedding step due to error while computing embedding:", e)

    # Suggest minimum recommended duration
    recommended = 1.6  # seconds; encoder partials length expressed in defaults
    if seconds(len(preprocessed), sampling_rate) < recommended:
        print(f"Warning: preprocessed reference is shorter than recommended ({recommended}s). Consider using a longer sample (>2-3s).\n")
    else:
        print("Reference audio meets recommended length for encoder partials.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python diagnose_audio.py <path_to_audio_file>")
    else:
        diagnose(sys.argv[1])
