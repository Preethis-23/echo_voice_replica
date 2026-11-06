"""Echo Voice Replica — XTTS v2 zero-shot voice cloning service.

XTTS extracts speaker conditioning directly from the uploaded reference audio
and generates the requested text with its own neural acoustic model and
vocoder. The reference audio is not copied; it conditions a new utterance.
"""

from pathlib import Path
import os
import uuid

import librosa
import soundfile as sf
import requests as http_requests
import torch
import torchaudio
from flask import Flask, after_this_request, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TTS_HOME", str(REPO_ROOT / ".cache" / "tts"))

# Coqui's current torchaudio path uses TorchCodec, which requires a separate
# FFmpeg shared-DLL installation on Windows. XTTS only needs decoded reference
# samples here, so use the already-installed SoundFile backend instead.
def _soundfile_load(path, *args, **kwargs):
    del args, kwargs
    samples, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    return torch.from_numpy(samples.T.copy()), sample_rate


torchaudio.load = _soundfile_load



UPLOAD_FOLDER = REPO_ROOT / "uploads"
OUTPUT_FOLDER = REPO_ROOT / "outputs"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

CHATTERBOX_WORKER_URL = "http://127.0.0.1:5001/generate"
OUTPUT_SAMPLE_RATE = 24000
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_TEXT_LENGTH = 2000



app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


def remove_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def validate_reference_audio(path: Path) -> tuple[float, int]:
    """Validate that a reference contains enough audio for speaker conditioning."""
    try:
        audio, sample_rate = librosa.load(str(path), sr=None, mono=True)
    except Exception as exc:
        raise ValueError(f"Could not decode the reference audio: {exc}") from exc

    if audio.size == 0:
        raise ValueError("The reference audio is empty or could not be decoded.")

    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    duration = len(trimmed) / float(sample_rate)
    if duration < 3.0:
        raise ValueError(
            "Reference audio must contain at least 3 seconds of clear speech."
        )
    return duration, sample_rate


@app.route("/generate_voice", methods=["POST"])
def generate_voice():
    reference_path: Path | None = None
    try:
        if "audio" not in request.files or "text" not in request.form:
            return jsonify({"error": "Audio file and text are required"}), 400

        audio_file = request.files["audio"]
        text = request.form["text"].strip()
        language = request.form.get("language", "en").strip().lower()

        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        if len(text.split()) < 4:
            return jsonify({"error": "Please enter at least 4 words so the voice cloning model has enough context."}), 400
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({"error": f"Text cannot exceed {MAX_TEXT_LENGTH} characters"}), 400
        if not audio_file.filename:
            return jsonify({"error": "No audio file selected"}), 400

        extension = Path(audio_file.filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            return jsonify({
                "error": f"Unsupported format '{extension}'. Use: "
                + ", ".join(sorted(ALLOWED_EXTENSIONS))
            }), 400

        filename = f"{uuid.uuid4().hex}_{secure_filename(audio_file.filename)}"
        reference_path = UPLOAD_FOLDER / filename
        audio_file.save(reference_path)
        speech_duration, source_sample_rate = validate_reference_audio(reference_path)
        print(
            f"[INFO] Reference: {speech_duration:.1f}s speech, "
            f"{source_sample_rate} Hz, language={language}"
        )

        output_path = OUTPUT_FOLDER / "chatterbox_output.wav"

        # Retry the Chatterbox worker up to 3 times.
        # On first inference after startup the model's alignment analyser can
        # throw a TypeError / IndexError before its buffer is populated; the
        # worker itself also retries internally, but this outer retry ensures
        # the browser never sees a 500 even when both collide on a cold start.
        worker_response = None
        last_worker_error = None
        for attempt in range(3):
            try:
                with reference_path.open("rb") as ref_stream:
                    worker_response = http_requests.post(
                        CHATTERBOX_WORKER_URL,
                        data={"text": text, "language": language},
                        files={"audio": (reference_path.name, ref_stream, "audio/wav")},
                        timeout=600,
                    )
                if worker_response.ok:
                    break
                last_worker_error = worker_response.text
                print(f"[WARN] Worker attempt {attempt+1} returned {worker_response.status_code}. Retrying...")
            except Exception as exc:
                last_worker_error = str(exc)
                print(f"[WARN] Worker attempt {attempt+1} raised {type(exc).__name__}: {exc}. Retrying...")

        if worker_response is None or not worker_response.ok:
            raise RuntimeError(
                f"Voice generation failed after 3 attempts. "
                f"Last error: {last_worker_error}"
            )

        output_path.write_bytes(worker_response.content)

        # Verify the model produced a non-empty audio file.
        generated_audio, generated_sr = sf.read(output_path)
        if generated_audio.size == 0:
            raise RuntimeError("Chatterbox produced an empty audio file")
        print(
            f"[INFO] Chatterbox output: {len(generated_audio) / generated_sr:.2f}s, "
            f"{generated_sr} Hz"
        )

        @after_this_request
        def cleanup_reference(response):
            if reference_path is not None:
                remove_file(reference_path)
            return response

        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="cloned_voice_output.wav",
        )

    except ValueError as exc:
        if reference_path is not None:
            remove_file(reference_path)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        if reference_path is not None:
            remove_file(reference_path)
        print(f"[ERROR] XTTS generation failed: {exc}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/audio_info", methods=["POST"])
def audio_info():
    temporary_path: Path | None = None
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        safe_name = secure_filename(audio_file.filename or "reference_audio")
        temporary_path = UPLOAD_FOLDER / f"_info_{uuid.uuid4().hex}_{safe_name}"
        audio_file.save(temporary_path)

        wav, sample_rate = librosa.load(str(temporary_path), sr=None, mono=True)
        duration = len(wav) / float(sample_rate)
        trimmed, _ = librosa.effects.trim(wav, top_db=30)
        speech_duration = len(trimmed) / float(sample_rate)
        quality = "good" if speech_duration >= 4 else "fair" if speech_duration >= 3 else "poor"

        return jsonify({
            "duration": round(duration, 2),
            "speech_duration": round(speech_duration, 2),
            "sample_rate": sample_rate,
            "quality": quality,
            "engine": "XTTS v2",
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if temporary_path is not None:
            remove_file(temporary_path)


@app.route("/")
def serve_frontend():
    return send_file(REPO_ROOT / "index.html")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
