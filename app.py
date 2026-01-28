#works only first few words - hi, iam cricket
'''import sys
import os
import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename

# Add paths to the Real-Time-Voice-Cloning repo
base_path = "C:/Users/preethis/PycharmProjects/echco_voice_replica/Real-Time-Voice-Cloning"
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "synthesizer"))
sys.path.append(os.path.join(base_path, "vocoder"))

# Import voice cloning modules
try:
    from synthesizer.hparams import hparams
    from encoder import inference as encoder
    from synthesizer.inference import Synthesizer
    from vocoder import inference as vocoder
except ImportError as e:
    print("Import error:", e)
    sys.exit(1)

# Initialize Flask app
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models
encoder_path = Path(base_path) / "encoder/saved_models/default/encoder.pt"
synthesizer_path = Path(base_path) / "synthesizer/saved_models/default/synthesizer.pt"
vocoder_path = Path(base_path) / "vocoder/saved_models/default/vocoder.pt"

encoder.load_model(encoder_path)
synthesizer = Synthesizer(synthesizer_path)
vocoder.load_model(vocoder_path)

# Directory to save uploaded and generated files
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/generate_voice', methods=['POST'])
def generate_voice():
    try:
        if 'audio' not in request.files or 'text' not in request.form:
            return jsonify({'error': 'Audio file and text are required'}), 400

        audio_file = request.files['audio']
        text = request.form['text']

        if audio_file.filename == '' or not audio_file.filename.endswith('.wav'):
            return jsonify({'error': 'Please upload a valid WAV file'}), 400

        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(audio_path)

        # Load and preprocess the voice sample
        preprocessed_wav = encoder.preprocess_wav(audio_path)
        embedding = encoder.embed_utterance(preprocessed_wav)

        # Create spectrogram and generate waveform
        spec = synthesizer.synthesize_spectrograms([text], [embedding])[0]
        waveform = vocoder.infer_waveform(spec)

        # Save output
        output_path = os.path.join(OUTPUT_FOLDER, "output.wav")
        sf.write(output_path, waveform, synthesizer.sample_rate)

        # Clean up
        os.remove(audio_path)

        return send_file(output_path, mimetype='audio/wav', as_attachment=True, download_name='output.wav')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_frontend():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)'''

'''
import sys
import os
import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename

# Add paths to the Real-Time-Voice-Cloning repo
base_path = "C:/Users/preethis/PycharmProjects/echco_voice_replica/Real-Time-Voice-Cloning"
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "synthesizer"))
sys.path.append(os.path.join(base_path, "vocoder"))

# Import voice cloning modules
try:
    from synthesizer.hparams import hparams
    from encoder import inference as encoder
    from synthesizer.inference import Synthesizer
    from vocoder import inference as vocoder
except ImportError as e:
    print("Import error:", e)
    sys.exit(1)

# Initialize Flask app
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models
encoder_path = Path(base_path) / "encoder/saved_models/default/encoder.pt"
synthesizer_path = Path(base_path) / "synthesizer/saved_models/default/synthesizer.pt"
vocoder_path = Path(base_path) / "vocoder/saved_models/default/vocoder.pt"

encoder.load_model(encoder_path)
synthesizer = Synthesizer(synthesizer_path)
vocoder.load_model(vocoder_path)

# Directory to save uploaded and generated files
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/generate_voice', methods=['POST'])
def generate_voice():
    try:
        if 'audio' not in request.files or 'text' not in request.form:
            return jsonify({'error': 'Audio file and text are required'}), 400

        audio_file = request.files['audio']
        text = request.form['text']

        if audio_file.filename == '' or not audio_file.filename.endswith('.wav'):
            return jsonify({'error': 'Please upload a valid WAV file'}), 400

        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(audio_path)

        # Load and preprocess the voice sample
        preprocessed_wav = encoder.preprocess_wav(audio_path)
        embedding = encoder.embed_utterance(preprocessed_wav)

        # Create spectrogram and generate waveform
        spec = synthesizer.synthesize_spectrograms([text], [embedding])[0]
        waveform = vocoder.infer_waveform(spec)

        # Save output without stretching (to maintain clarity)
        output_path = os.path.join(OUTPUT_FOLDER, "output.wav")
        sf.write(output_path, waveform, synthesizer.sample_rate)

        # Clean up
        os.remove(audio_path)

        return send_file(output_path, mimetype='audio/wav', as_attachment=True, download_name='output.wav')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_frontend():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
'''


import sys
import os
import numpy as np
import soundfile as sf
from pathlib import Path
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import re
from flask_cors import CORS
import torch

# Add paths to the Real-Time-Voice-Cloning repo
repo_root = Path(__file__).resolve().parent
base_path = repo_root / "Real-Time-Voice-Cloning"
sys.path.append(str(base_path))
sys.path.append(str(base_path / "synthesizer"))
sys.path.append(str(base_path / "vocoder"))

# Import voice cloning modules
try:
    from synthesizer.hparams import hparams
    from encoder import inference as encoder
    from synthesizer.inference import Synthesizer
    from vocoder import inference as vocoder
    from utils.default_models import ensure_default_models
except ImportError as e:
    print("Import error:", e)
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models
# Use the canonical RTVC model location: Real-Time-Voice-Cloning/saved_models/default/*.pt
# This avoids accidentally loading placeholder or mismatched weights elsewhere in the repo.
ensure_default_models(base_path / "saved_models")

encoder_path = base_path / "saved_models" / "default" / "encoder.pt"
synthesizer_path = base_path / "saved_models" / "default" / "synthesizer.pt"
vocoder_path = base_path / "saved_models" / "default" / "vocoder.pt"

encoder.load_model(encoder_path)
synthesizer = Synthesizer(synthesizer_path)
vocoder.load_model(vocoder_path)

# --- audio utilities -------------------------------------------------------
def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    """Peak-normalize safely to avoid clipping or exploding gain."""
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 1e-6:
        waveform = (waveform / peak) * 0.95
    return np.clip(waveform, -1.0, 1.0)

# Directory to save uploaded and generated files
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Enhance input text for better rhythm and natural pauses
def enhance_text_with_pauses(text):
    text = re.sub(r'\b(and|but|so|because|then|when)\b', r'\1,', text, flags=re.IGNORECASE)
    text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)
    words = text.split()
    result = []
    for i in range(0, len(words), 15):
        chunk = words[i:i+15]
        if len(chunk) == 15:
            result.append(' '.join(chunk) + ',')
        else:
            result.append(' '.join(chunk))
    return ' '.join(result)

@app.route('/generate_voice', methods=['POST'])
def generate_voice():
    try:
        if 'audio' not in request.files or 'text' not in request.form:
            return jsonify({'error': 'Audio file and text are required'}), 400

        audio_file = request.files['audio']
        raw_text = request.form['text']

        if audio_file.filename == '' or not audio_file.filename.endswith('.wav'):
            return jsonify({'error': 'Please upload a valid WAV file'}), 400

        # Enhance text
        text = enhance_text_with_pauses(raw_text)

        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(audio_path)

        # Load and preprocess the voice sample
        # Note: encoder expects 16 kHz; preprocess_wav handles resampling.
        preprocessed_wav = encoder.preprocess_wav(audio_path, trim_silence=False)
        duration_sec = len(preprocessed_wav) / float(encoder.sampling_rate)
        if duration_sec < 1.6:
            return jsonify({'error': 'Reference audio too short; please provide at least 2 seconds of clean speech.'}), 400

        with torch.no_grad():
            embedding = encoder.embed_utterance(preprocessed_wav)
            # Create spectrogram and generate waveform
            spec = synthesizer.synthesize_spectrograms([text], [embedding])[0]

            # Choose vocoder mode: 'wave' uses WaveRNN, 'griffin' uses Griffin-Lim
            voc_mode = request.form.get('vocoder', 'wave')
            if voc_mode not in ('wave', 'griffin'):
                voc_mode = 'wave'

                    # allow overriding WaveRNN inference params via form
            voc_target = int(request.form.get('voc_target', 8000))
            voc_overlap = int(request.form.get('voc_overlap', 200))

            if voc_mode == 'griffin':
                waveform = synthesizer.griffin_lim(spec)
            elif voc_mode == 'hifigan':
                # Try HiFi-GAN if available
                try:
                    from hifigan.inference import infer_waveform as hifi_infer, load_model as hifi_load, is_loaded as hifi_is_loaded
                    hifi_path = base_path / 'saved_models' / 'default' / 'hifigan.pt'
                    if not hifi_is_loaded() and hifi_path.exists():
                        hifi_load(hifi_path)
                    waveform = hifi_infer(spec)
                except Exception as e:
                    print('HiFi-GAN not available or failed to run:', e)
                    waveform = vocoder.infer_waveform(spec, target=voc_target, overlap=voc_overlap)
            else:
                waveform = vocoder.infer_waveform(spec, target=voc_target, overlap=voc_overlap)

        # Optional lightweight lowpass to suppress high-frequency noise
        try:
            from scipy.signal import butter, filtfilt
            def lowpass(wav, sr, cutoff=7000):
                b, a = butter(4, cutoff / (0.5 * sr), btype='low')
                return filtfilt(b, a, wav)
            apply_lowpass = request.form.get('lowpass', 'true').lower() not in ('0','false','no')
            if apply_lowpass:
                waveform = lowpass(waveform, synthesizer.sample_rate, cutoff=7000)
        except Exception:
            # If scipy not available, skip lowpass
            pass

        waveform = normalize_audio(waveform)

        # Auto-fallback: if using WaveRNN and output is likely noise, fall back to Griffin-Lim
        if voc_mode == 'wave':
            try:
                # High-frequency energy ratio heuristic
                from scipy.signal import stft
                f, t, Z = stft(waveform, fs=synthesizer.sample_rate, nperseg=1024)
                hf_ratio = np.sum(np.abs(Z[f >= 6000])) / (np.sum(np.abs(Z)) + 1e-9)
                if hf_ratio > 0.35:
                    print('Detected high HF energy in WaveRNN output (hf_ratio=', hf_ratio, '), falling back to Griffin-Lim')
                    waveform = synthesizer.griffin_lim(spec)
                    waveform = normalize_audio(waveform)
            except Exception:
                pass

        # Save output
        output_path = os.path.join(OUTPUT_FOLDER, "output.wav")
        sf.write(output_path, waveform, synthesizer.sample_rate, subtype="PCM_16")

        # Clean up
        os.remove(audio_path)

        return send_file(output_path, mimetype='audio/wav', as_attachment=True, download_name='output.wav')

    except Exception as e:
        print("Error generating audio:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_frontend():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)





