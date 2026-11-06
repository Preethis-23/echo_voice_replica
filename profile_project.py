import time
import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

repo_root = Path(__file__).resolve().parent
base_path = repo_root / "Real-Time-Voice-Cloning"
sys.path.insert(0, str(base_path))

from encoder import inference as encoder
from encoder.params_data import sampling_rate as encoder_sr
from synthesizer.inference import Synthesizer

models_dir = base_path / "saved_models" / "default"
if not (models_dir / "encoder.pt").exists():
    models_dir = repo_root / "saved_models" / "default"

print("==================================================")
print("       ECHO VOICE REPLICA - EMPIRICAL BENCHMARK   ")
print("==================================================")

# 1. Model Loading Benchmark
t0 = time.perf_counter()
encoder.load_model(models_dir / "encoder.pt")
t_enc_load = time.perf_counter() - t0

t0 = time.perf_counter()
synthesizer = Synthesizer(models_dir / "synthesizer.pt")
synthesizer.load()
t_syn_load = time.perf_counter() - t0

print(f"[MODEL LOAD] Encoder: {t_enc_load:.3f} s")
print(f"[MODEL LOAD] Synthesizer (Tacotron 2): {t_syn_load:.3f} s")
print(f"[MODEL LOAD] Total Initialization: {t_enc_load + t_syn_load:.3f} s")

# 2. Test Audio Benchmark across short, medium, long prompts
test_files = ["cleaned_ratheesh.wav", "sampleaudio.wav", "cleaned_dhoni.wav"]
valid_files = [f for f in test_files if os.path.exists(f)]

prompts = [
    ("Short (7 words)", "Hello! This is a quick voice cloning test."),
    ("Medium (17 words)", "Echo Voice Replica extracts speaker embeddings in real time and synthesizes natural-sounding speech."),
    ("Long (34 words)", "Artificial intelligence speech synthesis enables zero-shot voice replication by conditioning an autoregressive Tacotron two neural network on a 256 dimensional speaker d-vector extracted from a short reference audio sample.")
]

for test_f in valid_files:
    print(f"\n--------------------------------------------------")
    print(f"REFERENCE AUDIO: {test_f}")
    print(f"--------------------------------------------------")
    raw_wav, sr = librosa.load(test_f, sr=encoder_sr)
    ref_duration = len(raw_wav) / encoder_sr
    print(f"Reference Audio Length: {ref_duration:.2f} s ({len(raw_wav)} samples at {encoder_sr} Hz)")

    # Step A: Encoder Preprocessing & Embedding
    t0 = time.perf_counter()
    preprocessed_wav = encoder.preprocess_wav(raw_wav, source_sr=encoder_sr)
    t_prep = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        emb = encoder.embed_utterance(preprocessed_wav, using_partials=True)
    t_embed = time.perf_counter() - t0

    print(f" -> Encoder Preprocessing: {t_prep*1000:.1f} ms")
    print(f" -> Speaker Embedding Extraction: {t_embed*1000:.1f} ms (d-vector shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f})")

    for p_name, text in prompts:
        print(f"\n  * PROMPT: {p_name} [{len(text.split())} words, {len(text)} chars]")
        
        # Step B: Synthesizer (Mel Spectrogram generation)
        t0 = time.perf_counter()
        with torch.no_grad():
            specs = synthesizer.synthesize_spectrograms([text], [emb])
        t_synth = time.perf_counter() - t0
        mel = specs[0]

        # Step C: Vocoder (Griffin-Lim)
        t0 = time.perf_counter()
        wav_out = synthesizer.griffin_lim(mel)
        t_voc = time.perf_counter() - t0

        t_pipeline = t_prep + t_embed + t_synth + t_voc
        gen_duration = len(wav_out) / synthesizer.sample_rate
        rtf = t_pipeline / gen_duration

        peak_amp = float(np.max(np.abs(wav_out)))
        rms_val = float(np.sqrt(np.mean(wav_out**2)))

        print(f"    - Tacotron 2 Synthesis: {t_synth:.3f} s")
        print(f"    - Griffin-Lim Vocoding: {t_voc*1000:.1f} ms")
        print(f"    - Total Latency: {t_pipeline:.3f} s")
        print(f"    - Generated Duration: {gen_duration:.2f} s")
        print(f"    - Real-Time Factor (RTF): {rtf:.3f}x ({'Faster than real-time' if rtf < 1.0 else 'Slower than real-time'})")
        print(f"    - Audio Metrics: Peak = {peak_amp:.4f}, RMS = {rms_val:.4f}, Sample Rate = {synthesizer.sample_rate} Hz")

print("\n==================================================")
print("            BENCHMARK COMPLETE                    ")
print("==================================================")
