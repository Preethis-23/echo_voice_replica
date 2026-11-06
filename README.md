# Echo Voice Replica 🎙️🔊

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Web Framework](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Audio Sampling](https://img.shields.io/badge/Audio-16kHz%20%2F%2024kHz-blueviolet)](https://github.com/Preethis-23/echo_voice_replica)
[![Real-Time Factor](https://img.shields.io/badge/RTF-0.54x%20(CPU)-success)](https://github.com/Preethis-23/echo_voice_replica)

> **Real-Time Zero-Shot Voice Cloning & Neural Speech Synthesis Web Application**  
> Clone any human voice from just 3–10 seconds of reference audio and synthesize expressive, natural speech in real time on standard consumer hardware.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Empirical Benchmarks](#-empirical-benchmarks)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Pretrained Checkpoints](#pretrained-checkpoints)
  - [Running the Application](#running-the-application)
- [API Reference](#-api-reference)
- [What Makes Echo Voice Replica Unique?](#-what-makes-echo-voice-replica-unique)
- [License & Acknowledgements](#-license--acknowledgements)

---

## 🌟 Overview

**Echo Voice Replica** is an end-to-end voice cloning system and web application that reproduces a speaker's unique vocal identity—including timbre, pitch contours, and accent nuances—conditioned on a brief audio reference. 

Rather than requiring hours of studio recordings or expensive GPU fine-tuning cycles, Echo Voice Replica utilizes an SV2TTS (Speaker Verification to Multispeaker Text-to-Speech) zero-shot framework:

1. **Speaker Encoder**: Extracts a compact, robust 256-dimensional speaker embedding ($d\text{-vector}$) from unprocessed reference speech.
2. **Acoustic Synthesizer**: An autoregressive sequence-to-sequence neural network (Tacotron 2) that translates input text into an 80-channel mel spectrogram, guided by the target speaker's $d\text{-vector}$.
3. **Neural & DSP Vocoder**: Inverts the mel spectrogram into a crisp 16-bit PCM waveform using optimized phase reconstruction (Griffin-Lim) or neural vocoders (WaveRNN / HiFi-GAN), enhanced with digital high-pass filtering and peak loudness normalization.

The system comes with an interactive, dark-themed **Glassmorphism Web UI** featuring real-time speech quality analysis, live audio preview, and one-click audio export.

---

## 🏗️ System Architecture

```
                    +------------------------------------+
                    |   User Reference Audio (3-10 sec)  |
                    +-----------------+------------------+
                                      |
                                      v
                        [ Audio Preprocessing & VAD ]
                        (Silence Trimming / RMS Scaling)
                                      |
                                      v
                     +----------------------------------+
                     |    GE2E Speaker Encoder (LSTM)   |
                     |  Outputs 256-dim Speaker Vector  |
                     +----------------+-----------------+
                                      |
   +----------------------+           | (Conditioning Vector)
   |  Input Text Prompt   |           |
   +----------+-----------+           |
              |                       v
              +------------> [ Tacotron 2 Synthesizer ]
                             (Seq2Seq Attention Model)
                                      |
                                      v
                         [ Target Mel Spectrogram ]
                               (80 Mel Channels)
                                      |
                                      v
                         [ Vocoder & Audio Pipeline ]
                         * Griffin-Lim / HiFi-GAN Vocoder
                         * 60 Hz Butterworth High-Pass Filter
                         * 0.95 Peak Amplitude Normalization
                                      |
                                      v
                    +-----------------------------------+
                    |   High-Fidelity Cloned WAV Audio  |
                    +-----------------------------------+
```

---

## ✨ Key Features

- 🎯 **True Zero-Shot Cloning**: Clone any voice instantaneously from an arbitrary 3–10 second reference clip without fine-tuning, retraining, or dataset preparation.
- ⚡ **Faster-Than-Real-Time on CPU**: Achieves a **$0.54\times - 0.70\times$ Real-Time Factor (RTF)** on standard consumer CPU hardware—generating audio faster than playback duration.
- 🧹 **Signal-Level Cleanliness**: Incorporates a 2nd-order 60 Hz Butterworth high-pass filter and $0.95$ peak loudness normalization, completely removing low-frequency sub-bass hum, clipping, and "air-frequency" noise.
- 🧩 **Syntax-Aware Sentence Chunking**: Automatically breaks long passages at 120-character punctuation boundaries with gentle 150 ms inter-phrase pauses, eliminating Tacotron attention drift and skipped words.
- 🎨 **Modern Glassmorphic Web UI**: Sleek, responsive interface with drag-and-drop audio uploading, dynamic speech duration & quality analysis badges, character counter, and in-browser waveform playback.
- 🔌 **RESTful API Architecture**: Decoupled Flask backend exposing clean endpoints (`/generate_voice`, `/audio_info`) for seamless integration into downstream services.

---

## 📊 Empirical Benchmarks

The following benchmarks were measured empirically on standard x86 CPU hardware:

| Prompt Category | Text Length | Generation Time | Audio Duration | Real-Time Factor (RTF) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Short Utterance** | 8 words (42 chars) | **1.37 s** | 1.93 s | **0.71×** | ⚡ Faster than real-time |
| **Medium Utterance** | 13 words (100 chars) | **2.61 s** | 4.31 s | **0.60×** | ⚡ Faster than real-time |
| **Long Utterance** | 29 words (223 chars) | **10.41 s** | 19.30 s | **0.54×** | ⚡ Faster than real-time |

### Pipeline Latency Breakdown
- **Encoder Preprocessing**: `0.7 ms – 4.3 ms`
- **Speaker Embedding Extraction**: `41.3 ms` (for standard 6s reference audio)
- **Cold-Start Model Initialization**: `593 ms` total (32 ms Encoder + 560 ms Synthesizer)
- **Memory Footprint**: `~590 MB – 750 MB` total process RAM on CPU
- **Output Quality**: Fixed peak amplitude of `0.9500` (zero digital clipping) with an average RMS of `0.0716`.

---

## 💻 Technology Stack

| Layer | Component | Description |
| :--- | :--- | :--- |
| **Core AI / ML** | [PyTorch](https://pytorch.org/) | Neural network inference engine for Encoder & Tacotron 2 |
| **Speaker Encoder** | GE2E LSTM (`encoder.pt`) | 3-layer LSTM extracting 256-dimensional speaker $d\text{-vectors}$ |
| **Acoustic Model** | Tacotron 2 (`synthesizer.pt`) | 30.87M parameter autoregressive sequence-to-sequence model |
| **Neural Vocoding** | Griffin-Lim / WaveRNN / HiFi-GAN | Multi-mode mel-to-waveform audio synthesis |
| **Digital Signal Processing** | [Librosa](https://librosa.org/) & [SciPy](https://scipy.org/) | Spectrogram derivation, STFT, 60 Hz filtering, volume normalization |
| **Audio I/O** | [SoundFile](https://python-soundfile.readthedocs.io/) | Lossless 16-bit PCM WAV reading and writing |
| **Backend API** | [Flask](https://flask.palletsprojects.com/) + Flask-CORS | Lightweight REST API server with multipart upload handling |
| **Frontend** | HTML5, CSS3, Vanilla ES6 | Dark-themed glassmorphic interface with audio inspection badges |

---

## 📁 Project Structure

```
echo_voice_replica/
├── app.py                      # Flask REST API server & web application entrypoint
├── index.html                  # Interactive Glassmorphic web user interface
├── requirements.txt            # Production Python package dependencies
├── profile_project.py          # Latency, RTF, and audio metric benchmark suite
├── test_synthesis.py           # Automated integration test script for API endpoints
├── download_hifigan.py         # Helper script to download pretrained vocoder weights
├── diagnose_audio.py           # Utility to inspect reference audio quality and VAD
├── run_synthesis_test.py       # Standalone CLI voice synthesis testing tool
├── Real-Time-Voice-Cloning/    # Core SV2TTS model implementation (Encoder, Synthesizer, Vocoder)
│   ├── encoder/                # GE2E speaker encoder architecture & audio processing
│   ├── synthesizer/            # Tacotron 2 acoustic model & text cleaners
│   └── vocoder/                # WaveRNN and Griffin-Lim vocoder inference
├── saved_models/               # Pretrained neural network checkpoints
│   └── default/
│       ├── encoder.pt          # Trained speaker encoder weights
│       ├── synthesizer.pt      # Trained Tacotron 2 synthesizer weights
│       └── vocoder.pt          # Trained WaveRNN vocoder weights
├── uploads/                    # Temporary storage for uploaded reference samples (auto-cleaned)
├── outputs/                    # Generated cloned audio outputs
└── LICENSE                     # MIT Open Source License
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: `3.10` or `3.11` recommended
- **Operating System**: Windows 10/11, macOS, or Linux
- **Hardware**: Standard multi-core CPU (GPU with CUDA optional for even faster inference)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Preethis-23/echo_voice_replica.git
   cd echo_voice_replica
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # On Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # On Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Pretrained Checkpoints

Ensure the pretrained model checkpoints are present in `Real-Time-Voice-Cloning/saved_models/default/` (or `saved_models/default/`):
- `encoder.pt` (~15 MB)
- `synthesizer.pt` (~123 MB)
- `vocoder.pt` (~18 MB)

*(Optional) To download HiFi-GAN vocoder weights, run:*
```bash
python download_hifigan.py
```

### Running the Application

1. **Start the Flask web server**:
   ```bash
   python app.py
   ```

2. **Open your browser**:
   Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

3. **Clone a voice**:
   - Drag and drop or upload a 3–10 second voice recording (`.wav`, `.mp3`, `.m4a`, `.flac`).
   - Enter your desired text.
   - Click **Generate Cloned Voice** to preview and download the output WAV.

---

## 🔌 API Reference

### 1. Generate Cloned Voice
- **Endpoint**: `POST /generate_voice`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `audio`: Audio file containing reference speech.
  - `text`: Text prompt to synthesize (up to 2,000 characters).
- **Response**: `200 OK` streaming `audio/wav` attachment (`cloned_voice_output.wav`).

### 2. Inspect Reference Audio
- **Endpoint**: `POST /audio_info`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `audio`: Audio file to analyze.
- **Response**: `200 OK` JSON:
  ```json
  {
    "duration": 6.38,
    "speech_duration": 6.25,
    "sample_rate": 16000,
    "quality": "good"
  }
  ```

---

## 💎 What Makes Echo Voice Replica Unique?

- 🎙️ **Zero-Shot Adaptation Without Retraining**: Unlike traditional voice cloning models that require gradient descent fine-tuning or minutes of clean data, Echo Voice Replica generalizes to completely unseen speakers in **under 50 milliseconds** using a single feed-forward pass of the speaker encoder.
- 🚀 **CPU-First Real-Time Efficiency**: Engineered for extreme accessibility. While modern diffusion-based voice clones demand high-end GPUs, Echo Voice Replica delivers an impressive **$0.54\times$ Real-Time Factor on standard CPUs**, making natural speech synthesis deployable on low-cost servers, edge systems, or local developer laptops.
- 🎚️ **Hybrid Neural-DSP Signal Conditioning**: Solves the common "robotic hum" and "air frequency static" plaguing synthetic audio by combining deep autoregressive representations with precision digital signal processing (active 60 Hz zero-phase high-pass filtering + dynamic peak normalization to 0.95).
- 🧱 **Decoupled Three-Stage Architecture**: The modular separation of speaker encoding, acoustic synthesis, and vocoding enables independent component upgrades (e.g., swapping Griffin-Lim for HiFi-GAN) without breaking pipeline cohesion.
- 🌐 **Clean, Self-Contained Web Experience**: Complete out-of-the-box solution—no complicated command-line setups or external cloud API keys required. Just run `python app.py` and enjoy an intuitive, private voice cloning workstation.

---

## 📄 License & Acknowledgements

This project is licensed under the [MIT License](LICENSE).

### Acknowledgements & References
- **SV2TTS Architecture**: Based on the seminal research paper *"Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis"* (Jia et al., 2018).
- **Real-Time-Voice-Cloning**: Framework inspired by [CorentinJ/Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning).
- **HiFi-GAN**: Vocoder models based on [jik876/hifi-gan](https://github.com/jik876/hifi-gan).

---

<p align="center">
  <b>Echo Voice Replica</b> · Built by <a href="https://github.com/Preethis-23">Preethis</a>
</p>
