"""
Voice cloning worker — two-stage pipeline
==========================================
Stage 1 : ChatterboxTTS  — generates clean, neutral English speech from text
Stage 2 : ChatterboxVC   — voice-converts that speech to exactly match the
                            reference speaker's timbre, pitch, and accent

Why two stages instead of conditioned TTS?
  Zero-shot TTS (conditioned on a reference) must invent phonemes, prosody,
  and voice simultaneously from scratch — and always drifts from the reference.
  Voice Conversion takes *existing* audio and transforms ONLY the voice
  characteristics, leaving the linguistic content intact.  The result is a
  much closer match to the reference speaker.
"""

from pathlib import Path
import tempfile
import uuid

import numpy as np
import soundfile as sf
import torch
import torchaudio
from flask import Flask, jsonify, request, send_file
from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC

ROOT   = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "chatterbox_worker.wav"
UPLOAD = ROOT / "uploads"
UPLOAD.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load both models ────────────────────────────────────────────────────────
print(f"[INFO] Loading ChatterboxTTS on {DEVICE}...")
tts_model = ChatterboxTTS.from_pretrained(DEVICE)
print("[INFO] ChatterboxTTS ready.")

print(f"[INFO] Loading ChatterboxVC on {DEVICE}...")
vc_model = ChatterboxVC.from_pretrained(DEVICE)
print("[INFO] ChatterboxVC ready.")

# ── Warm-up: prime both models so the first real request is fast ─────────────
print("[INFO] Warming up TTS model...")
_warmup_wav = UPLOAD / "_warmup.wav"
try:
    _warmup = tts_model.generate(
        "Warm up.",
        exaggeration=0.1,
        cfg_weight=0.5,
        temperature=0.7,
    )
    torchaudio.save(str(_warmup_wav), _warmup, tts_model.sr)
    print("[INFO] TTS warm-up done. Warming up VC model...")
    vc_model.generate(audio=str(_warmup_wav), target_voice_path=str(_warmup_wav))
    print("[INFO] VC warm-up done — worker ready.")
except Exception as _e:
    print(f"[INFO] Warm-up completed with: {_e}")
finally:
    _warmup_wav.unlink(missing_ok=True)

app = Flask(__name__)


@app.post("/generate")
def generate():
    if "audio" not in request.files or "text" not in request.form:
        return jsonify({"error": "audio and text are required"}), 400

    reference_file = request.files["audio"]
    text           = request.form["text"].strip()

    if not text:
        return jsonify({"error": "text cannot be empty"}), 400

    ref_path  = UPLOAD / f"ref_{uuid.uuid4().hex}.wav"
    tts_path  = UPLOAD / f"tts_{uuid.uuid4().hex}.wav"

    try:
        # ── Save reference audio ────────────────────────────────────────────
        reference_file.save(ref_path)

        # ── Stage 1: TTS — generate clean speech in Chatterbox's own voice ─
        print(f"[INFO] Stage 1 TTS: '{text[:60]}'")
        last_err = None
        tts_wav  = None
        for attempt in range(3):
            try:
                tts_wav = tts_model.generate(
                    text,
                    exaggeration=0.1,      # very neutral; VC handles the voice
                    cfg_weight=0.5,
                    temperature=0.7,
                    repetition_penalty=1.3,
                )
                break
            except Exception as exc:
                last_err = exc
                print(f"[WARN] TTS attempt {attempt+1} failed ({type(exc).__name__}): {exc}")

        if tts_wav is None:
            return jsonify({"error": f"TTS failed: {last_err}"}), 500

        torchaudio.save(str(tts_path), tts_wav, tts_model.sr)
        print(f"[INFO] Stage 1 done: {tts_wav.shape[-1]/tts_model.sr:.2f}s at {tts_model.sr} Hz")

        # ── Stage 2: VC — transform TTS audio into the reference speaker ───
        print(f"[INFO] Stage 2 VC: converting to reference speaker voice...")
        last_err = None
        vc_wav   = None
        for attempt in range(3):
            try:
                vc_wav = vc_model.generate(
                    audio=str(tts_path),
                    target_voice_path=str(ref_path),
                )
                break
            except Exception as exc:
                last_err = exc
                print(f"[WARN] VC attempt {attempt+1} failed ({type(exc).__name__}): {exc}")

        if vc_wav is None:
            return jsonify({"error": f"Voice conversion failed: {last_err}"}), 500

        # ── Save final output ───────────────────────────────────────────────
        final_audio = vc_wav.squeeze().detach().cpu().numpy()
        sf.write(str(OUTPUT), final_audio, vc_model.sr, subtype="PCM_16")
        print(f"[INFO] Stage 2 done: {len(final_audio)/vc_model.sr:.2f}s at {vc_model.sr} Hz")

        return send_file(
            OUTPUT,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="cloned_voice_output.wav",
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    finally:
        ref_path.unlink(missing_ok=True)
        tts_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
