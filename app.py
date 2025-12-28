"""
Flask backend for Stuttering Detection + Exercises (extended from previous).
Replace original app.py with this file. Keeps original model loading and predict pipeline.
"""

import os
import tempfile
import logging
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import noisereduce as nr
from flask_cors import CORS

from flask import Flask, request, jsonify, send_from_directory
app = Flask(__name__)
CORS(app)   

# ---------- Config ----------
# ---------- Config ----------
# Priority: 1. Environment variable 2. Current script directory 3. Hardcoded default
SCRIPT_DIR = Path(__file__).parent.absolute()
# Explicit path requested by user
REQUESTED_MODELS_DIR = Path(r"D:/fluenta_fresh/backend/backend/models_retrained")
DEFAULT_MODELS_DIR = REQUESTED_MODELS_DIR if REQUESTED_MODELS_DIR.exists() else (SCRIPT_DIR / "models_retrained")

MODELS_DIR = Path(os.environ.get("FLUENTA_MODELS_DIR", DEFAULT_MODELS_DIR))


# candidate filenames (keeps compatibility)
PREFERRED_MODEL_FILENAMES = ("stutter_model_retrained.pkl", "stutter_model_resaved.pkl", "stutter_model_final.pkl", "stutter_model.pkl")
PREFERRED_SCALER_FILENAMES = ("scaler_retrained.pkl", "scaler_final.pkl", "scaler.pkl")
PREFERRED_FCOUNT_FILENAMES = ("feature_count_final.pkl", "feature_count.pkl", "feature_count_val.pkl")

# directories for audio templates and uploads
EX_AUDIO_DIR = SCRIPT_DIR / "ex_audio"
UPLOADS_DIR = SCRIPT_DIR / "uploads"
LOGS_DIR = SCRIPT_DIR / "logs"

EX_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_LOG_FILE = LOGS_DIR / "predictions_log.json"
SUBMISSIONS_LOG_FILE = LOGS_DIR / "submissions_log.json"
RETRAIN_QUEUE_FILE = LOGS_DIR / "retrain_queue.jsonl"

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fluenta_api")

# ---------- Globals ----------
MODEL = None
SCALER = None
FEATURE_COUNT = None
MODEL_PATH_LOADED = None
SCALER_PATH_LOADED = None
FCOUNT_PATH_LOADED = None

PREDICTIONS_LOG = []   # list of dicts saved to predictions_log.json
SUBMISSIONS_LOG = []   # list of dicts saved to submissions_log.json
LOG_LOCK = threading.Lock()

# ---------- Exercise templates (short, precise) ----------
# Put the referenced audio files in ./ex_audio/
EXERCISES = [
    {
        "exercise_id": "breath_01",
        "title": "Breathing Control",
        "paragraph": "Focus on slow, even breaths. Inhale quietly for 4 seconds; exhale for 6 seconds. Keep shoulders relaxed.",
        "instructions": "Listen to the guide audio, repeat cycle 10 times, then read a short sentence while keeping steady breath. Record the whole minute.",
        "audio_file": "breath_guide_4_6.mp3",
        "type": "breathing",
        "difficulty": "easy"
    },
    {
        "exercise_id": "slowread_01",
        "title": "Slow Reading",
        "paragraph": "Read slowly and clearly. Pause slightly between phrases. Aim for smooth transitions.",
        "instructions": "Open the script, speak at ~60% normal speed, record one minute reading. Play template audio first for pace.",
        "audio_file": "slow_read_demo.mp3",
        "type": "reading",
        "difficulty": "easy"
    },
    {
        "exercise_id": "rhythm_01",
        "title": "Rhythmic/Pacing Speech",
        "paragraph": "Use a steady beat to pace words. One word per beat keeps rhythm and reduces repetition.",
        "instructions": "Play the metronome audio, speak each word on the beat for 1 minute. Record and submit.",
        "audio_file": "metronome_60bpm.mp3",
        "type": "rhythmic",
        "difficulty": "medium"
    },
    {
        "exercise_id": "voice_mod_01",
        "title": "Voice Modulation",
        "paragraph": "Practice gentle changes in pitch and volume to gain control over your tone.",
        "instructions": "Follow the demo: say the sentence with low pitch, neutral, then higher pitch. Repeat 5 times and record.",
        "audio_file": "voice_mod_demo.mp3",
        "type": "modulation",
        "difficulty": "medium"
    }
]

# ---------- Utilities ----------
def pick_existing_file(models_dir: Path, candidates: Tuple[str, ...]) -> Optional[Path]:
    for fn in candidates:
        p = models_dir / fn
        if p.exists():
            return p
    return None

def safe_load_models(models_dir: Path) -> Tuple[Optional[object], Optional[object], Optional[int]]:
    """
    Load model, scaler, and feature_count from models_dir using preferred filenames.
    """
    global MODEL_PATH_LOADED, SCALER_PATH_LOADED, FCOUNT_PATH_LOADED
    try:
        models_dir = Path(models_dir)
        model_path = pick_existing_file(models_dir, PREFERRED_MODEL_FILENAMES)
        scaler_path = pick_existing_file(models_dir, PREFERRED_SCALER_FILENAMES)
        fc_path = pick_existing_file(models_dir, PREFERRED_FCOUNT_FILENAMES)

        if model_path is None or scaler_path is None or fc_path is None:
            logger.error("Missing required artifact(s) in %s", models_dir)
            logger.debug("Files present: %s", [p.name for p in models_dir.iterdir() if p.is_file()])
            return None, None, None

        logger.info("Loading model from: %s", model_path)
        model = joblib.load(model_path)
        logger.info("Loading scaler from: %s", scaler_path)
        scaler = joblib.load(scaler_path)
        logger.info("Loading feature_count from: %s", fc_path)
        fc_raw = joblib.load(fc_path)

        try:
            feature_count = int(fc_raw)
        except Exception:
            try:
                feature_count = int(np.asarray(fc_raw).item())
            except Exception:
                feature_count = fc_raw

        MODEL_PATH_LOADED = model_path
        SCALER_PATH_LOADED = scaler_path
        FCOUNT_PATH_LOADED = fc_path

        logger.info("Model type: %s", type(model))
        logger.info("Feature count: %s", feature_count)
        return model, scaler, feature_count
    except Exception:
        logger.exception("Error loading artifacts")
        return None, None, None

def convert_to_wav(src_bytes: bytes, src_filename: str) -> str:
    tmp_src = None
    tmp_wav = None
    try:
        tmp_src = tempfile.NamedTemporaryFile(delete=False, suffix=Path(src_filename).suffix)
        tmp_src.write(src_bytes); tmp_src.flush(); tmp_src.close()

        # Save permanently in uploads folder for retraining
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in src_filename if c.isalnum() or c in "._-").rstrip()
        out_name = f"mobile_{timestamp}_{safe_name}.wav" if not safe_name.endswith(".wav") else f"mobile_{timestamp}_{safe_name}"
        if not out_name.endswith(".wav"): out_name += ".wav"
        
        final_wav_path = str(UPLOADS_DIR / out_name)

        audio = AudioSegment.from_file(tmp_src.name)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(final_wav_path, format="wav")
        return final_wav_path

    finally:
        try:
            if tmp_src is not None:
                os.remove(tmp_src.name)
        except Exception:
            pass

def preprocess_audio(y: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    # 1. Basic energy check: if signal is effectively zero, abort or return error
    if len(y) == 0:
        raise ValueError("Audio buffer is empty")
    
    rms_energy = np.sqrt(np.mean(y**2))
    duration = len(y) / sr
    logger.info("Preprocess: signal length=%d (%.3fs), RMS energy=%.8f", len(y), duration, rms_energy)
    
    if duration < 1.0 or rms_energy < 0.01:
        # Increased thresholds: 1s minimum and 0.01 RMS to filter quiet room noise
        logger.warning("Audio too short (%.3fs) or quiet (RMS: %.8f). Analysis may be rejected.", duration, rms_energy)





    # 2. Trim silence with a slightly more relaxed threshold (30db instead of 20)
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        # If trimming removed everything or left too little, it was likely silence/noise
        if len(y_trimmed) < int(0.1 * sr):
             logger.warning("Trimming removed most of the audio; likely silent or noise.")
        else:
            y = y_trimmed
    except Exception:
        pass

    # 3. Normalize
    try:
        y = librosa.util.normalize(y)
    except Exception:
        pass

    # 4. Noise reduction
    try:
        y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
        # Fix possible NaNs/Infs from noise reduction
        if not np.all(np.isfinite(y_denoised)):
            y_denoised = np.nan_to_num(y_denoised)
        y = y_denoised
    except Exception:
        pass

    # 5. Resample
    if sr != target_sr:
        try:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception:
            pass

    # 6. Minimum length padding
    min_length = int(0.5 * sr)
    if len(y) < min_length:
        y = np.pad(y, (0, max(0, min_length - len(y))), mode="constant")
    
    return y, sr


def extract_features_from_file(file_path: str) -> Tuple[Optional[np.ndarray], float]:
    try:
        # Use librosa.load instead of sf.read for much better format support (ffmpeg/audioread)
        y, sr = librosa.load(file_path, sr=16000)
        duration = len(y) / sr
        y, sr = preprocess_audio(y, sr, target_sr=16000)


        feats = []
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        feats.extend(np.mean(mfccs.T, axis=0).tolist())
        feats.extend(np.std(mfccs.T, axis=0).tolist())
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feats.append(float(np.mean(spectral_centroids)))
        feats.append(float(np.std(spectral_centroids)))
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        feats.append(float(np.mean(spectral_rolloff)))
        feats.append(float(np.std(spectral_rolloff)))
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feats.append(float(np.mean(zcr)))
        feats.append(float(np.std(zcr)))
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats.extend(np.mean(chroma.T, axis=0).tolist())
        rms = librosa.feature.rms(y=y)[0]
        feats.append(float(np.mean(rms)))
        feats.append(float(np.std(rms)))
        return np.array(feats, dtype=np.float32), duration
    except Exception as e:
        logger.exception("Feature extraction failed for %s: %s", file_path, e)
        return None, 0.0


def predict_from_features(features: np.ndarray, thresh: float = 0.7, debug: bool = False):
    if SCALER is None or MODEL is None:
        raise RuntimeError("Model or scaler not loaded")
    features = np.asarray(features).flatten()
    try:
        expected = int(FEATURE_COUNT)
    except Exception:
        expected = FEATURE_COUNT
    if features.size != expected:
        raise ValueError(f"Observed feature length: {features.size} Expected: {expected}")
    X_scaled = SCALER.transform([features])
    try:
        raw_pred = int(MODEL.predict(X_scaled)[0])
    except Exception as e:
        raise RuntimeError(f"MODEL.predict failed: {e}")
    p_stutter = None
    raw_proba = None
    idx_used = None
    try:
        if hasattr(MODEL, "predict_proba"):
            raw_proba = MODEL.predict_proba(X_scaled)[0]
            if hasattr(MODEL, "classes_"):
                classes_list = list(MODEL.classes_)
                if 1 in classes_list:
                    idx_used = classes_list.index(1)
                else:
                    idx_used = 1 if len(raw_proba) > 1 else 0
            else:
                idx_used = 1 if len(raw_proba) > 1 else 0
            if 0 <= idx_used < len(raw_proba):
                p_stutter = float(raw_proba[idx_used])
                logger.info("Raw model p_stutter: %.4f (threshold: %s)", p_stutter, thresh)

            else:

                p_stutter = None
    except Exception as e:
        logger.warning("predict_proba failed or unusable: %s", e)
        raw_proba = None
        p_stutter = None
        idx_used = None
    if p_stutter is not None and isinstance(p_stutter, (float, int)) and not np.isnan(p_stutter):
        if p_stutter >= thresh:
            final_label = "Stutter"
            final_confidence = float(p_stutter)
        else:
            final_label = "Non-Stutter"
            final_confidence = float(1.0 - p_stutter)
    else:
        final_label = "Stutter" if raw_pred == 1 else "Non-Stutter"
        final_confidence = None
    if debug:
        logger.info("=== PREDICTION DEBUG INFO ===")
        logger.info("Observed feature length: %s Expected: %s", features.size, expected)
        logger.info("Raw predict class: %s", raw_pred)
        logger.info("MODEL.classes_: %s", getattr(MODEL, "classes_", None))
        logger.info("Raw proba array: %s", raw_proba)
        logger.info("Index used for class '1': %s", idx_used)
        logger.info("p_stutter: %s", p_stutter)
        logger.info("THRESH: %s", thresh)
        logger.info("Final label: %s", final_label)
        logger.info("Final confidence: %s", final_confidence)
        logger.info("=== END DEBUG INFO ===")
    return raw_pred, final_confidence, final_label

# ---------- Persistence helpers ----------
def _load_json_file(path):
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load %s", path)
        return []

def _save_json_file(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed to save %s", path)

# ---------- Flask app ----------
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    ok = MODEL is not None and SCALER is not None and FEATURE_COUNT is not None
    return jsonify({
        "status": "ok" if ok else "error",
        "model_loaded": MODEL is not None,
        "model_path": str(MODEL_PATH_LOADED) if MODEL_PATH_LOADED else None,
        "scaler_loaded": SCALER is not None,
        "scaler_path": str(SCALER_PATH_LOADED) if SCALER_PATH_LOADED else None,
        "feature_count": FEATURE_COUNT
    })

# Keep original predict endpoint behavior (file -> features -> model)
@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None or SCALER is None or FEATURE_COUNT is None:
        return jsonify({"error": "Model not loaded on server"}), 500
    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded (use key 'file')"}), 400
    uploaded = request.files["file"]
    filename = uploaded.filename or "uploaded_audio"
    file_bytes = uploaded.read()
    header_hex = file_bytes[:16].hex(' ')
    logger.info("Predict: received file '%s' size=%d bytes, header: %s", filename, len(file_bytes), header_hex)

    
    # convert or write wav

    if filename.lower().endswith(".wav"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"mobile_raw_{timestamp}_{filename}"
        wav_path = str(UPLOADS_DIR / out_name)
        try:
            with open(wav_path, "wb") as f:
                f.write(file_bytes)
            logger.info("Saved raw WAV for retraining: %s", wav_path)
        except Exception as e:
            logger.exception("Failed to write WAV to uploads: %s", e)
            return jsonify({"error": "Failed to write WAV file"}), 500

    else:
        try:
            wav_path = convert_to_wav(file_bytes, filename)
        except Exception as e:
            logger.exception("Conversion to wav failed: %s", e)
            return jsonify({"error": f"Audio conversion failed: {e}"}), 500
    try:
        features, duration_sec = extract_features_from_file(wav_path)
        
        # Check for essentially empty or too short audio
        is_empty = False
        if features is None or duration_sec < 1.0:
            is_empty = True
        else:
            # Check RMS energy in the feature vector
            if features[-2] < 0.01: # RMS mean
                is_empty = True

        if is_empty:
            return jsonify({
                "prediction": "No Speech",
                "confidence": 0,
                "fluency_score": 100,
                "severity": 0,
                "stutter_severity": 0,
                "feedback": "No clear speech detected. Please speak closer to the mic for 3-5s.",
                "duration_received": float(duration_sec),
                "model_path": str(MODEL_PATH_LOADED) if MODEL_PATH_LOADED else None
            })




        observed = len(features)

        try: expected = int(FEATURE_COUNT)
        except Exception: expected = FEATURE_COUNT
        logger.info("Feature vector length observed=%s expected=%s", observed, expected)
        if observed != expected:
            return jsonify({
                "error": "Feature count mismatch",
                "expected_feature_count": expected,
                "observed_feature_count": observed
            }), 500
        # Use a more conservative threshold (0.8) to reduce false positives on mobile
        pred_int, confidence, label = predict_from_features(features, thresh=0.8, debug=False)
        
        if confidence is None:
            fluency_score = 50.0
            stutter_severity = 50.0
        else:
            # Calculation logic:
            # confidence reflects probability of the chosen 'label'.
            # If label is Non-Stutter, confidence is the probability of being fluent.
            # However, for mobile users, if p_stutter is high but below thresh (e.g. 0.7),
            # we should still treat them as mostly fluent to avoid confusing "Moderate Stutter" results.
            
            if label == "Non-Stutter":
                # Scale fluency so that 1.0 confidence -> 100%, and even 0.2 confidence (near stutter)
                # doesn't drop below a reasonable flueny baseline if it was still labeled fluent.
                fluency_score = round(confidence * 100, 2)
                stutter_severity = round((1 - confidence) * 100, 2)
            else:
                stutter_severity = round(confidence * 100, 2)
                fluency_score = round((1 - confidence) * 100, 2)

        # Force severe reduction of severity if labeled Non-Stutter and it's close to fluent
        if label == "Non-Stutter" and stutter_severity < 40:
             stutter_severity = round(stutter_severity / 2, 2)
             fluency_score = 100 - stutter_severity


        # Ensure fluency_score is never None for feedback
        f_score = fluency_score if fluency_score is not None else 50.0

        if f_score <= 30:
            feedback = "Stuttering detected. Try using prolonged speech or gentle onsets."
        elif f_score <= 80:
            feedback = "Moderate stuttering detected. Keep practicing your techniques!"
        else:
            feedback = "Excellent! You are sounding very fluent today."

        result = {
            "prediction": label,
            "confidence": round(float(confidence) * 100, 2) if confidence is not None else None,
            "fluency_score": fluency_score,
            "stutter_severity": stutter_severity,
            "severity": stutter_severity, # alias for compatibility
            "feedback": feedback,
            "model_path": str(MODEL_PATH_LOADED) if MODEL_PATH_LOADED else None
        }

        # log prediction (persist to disk)
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "filename": filename,
            "prediction": label,
            "confidence": float(confidence) if confidence is not None else None,
            "model_path": str(MODEL_PATH_LOADED) if MODEL_PATH_LOADED else None
        }
        with LOG_LOCK:
            PREDICTIONS_LOG.append(entry)
            _save_json_file(PREDICTIONS_LOG_FILE, PREDICTIONS_LOG)
        return jsonify(result)
    finally:
        # Commented out to allow the user to collect mobile recordings for retraining
        # try: os.remove(wav_path)
        # except Exception: pass
        pass


# ---------- New endpoints: exercises, audio, recommend, submit, progress, flag ----------

@app.route("/exercises", methods=["GET"])
def list_exercises():
    # returns short list of templates
    return jsonify(EXERCISES)

@app.route("/ex_audio/<path:filename>", methods=["GET"])
def serve_ex_audio(filename):
    # serve static audio from ex_audio folder
    p = EX_AUDIO_DIR / filename
    if not p.exists():
        return jsonify({"error": "audio not found"}), 404
    return send_from_directory(str(EX_AUDIO_DIR.resolve()), filename, as_attachment=False)

@app.route("/recommend", methods=["GET"])
def recommend():
    """
    Query params:
      - label (optional): "Stutter" or "Non-Stutter"
      - confidence (optional): float 0..100 (percent) OR 0..1
    If not provided, uses last prediction in predictions_log.
    """
    label = request.args.get("label")
    conf_raw = request.args.get("confidence")
    confidence = None
    if conf_raw is not None:
        try:
            confidence = float(conf_raw)
            # allow percent => normalize to 0..1
            if confidence > 1.0:
                confidence = confidence / 100.0
        except Exception:
            confidence = None

    if label is None:
        with LOG_LOCK:
            if PREDICTIONS_LOG:
                last = PREDICTIONS_LOG[-1]
                label = last.get("prediction")
                confidence = last.get("confidence")
                if confidence is not None and confidence > 1.0:
                    confidence = float(confidence) / 100.0

    if label is None:
        return jsonify({"error": "No label provided and no predictions available"}), 400

    # simple rule-based mapping:
    recs = []
    if label in ("Non-Stutter", "NoStutter", "No-Stutter"):
        recs.append({"id": "tips_maintain", "title": "Maintenance Tip", "desc": "Keep monitoring and do Slow Reading weekly."})
    elif label == "Stutter":
        if confidence is None:
            recs = [{"id":"retest","title":"Uncertain","desc":"Confidence unknown. Please record a test."}]
        else:
            if confidence >= 0.8:
                # strong stutter detection
                recs = [e for e in EXERCISES if e["exercise_id"] in ("breath_01","slowread_01","rhythm_01")]
            elif confidence >= 0.5:
                recs = [e for e in EXERCISES if e["exercise_id"] in ("breath_01","slowread_01")]
            else:
                recs = [{"id":"retest","title":"Uncertain","desc":"Confidence low. Please record a longer sample or re-test."}]
    else:
        recs = [{"id":"retest","title":"Unknown label","desc":"Label not recognized. Provide label=Stutter or label=Non-Stutter"}]

    return jsonify({"recommendations": recs})

def compute_score_from_prediction(label, confidence, events_count, duration_sec=60.0):
    """
    Produce a numeric score 0..100 (higher -> more fluent).
    Uses confidence and events/min penalty.
    """
    if confidence is None:
        base = 50.0
    else:
        # ensure 0..1
        if confidence > 1.0:
            confidence = confidence / 100.0
        base = confidence * 100.0 if label != "Stutter" else (1.0 - confidence) * 100.0

    dpm = (events_count / duration_sec) * 60.0 if duration_sec > 0 else 0.0
    penalty = min(dpm / 10.0, 1.0)
    score = base * (1.0 - penalty)
    return max(0.0, min(100.0, round(float(score), 2))), {"dpm": round(dpm, 3), "events_count": events_count, "base": round(base,2)}

@app.route("/exercise/submit", methods=["POST"])
def submit_exercise():
    """
    Form fields:
      - exercise_id (required)
      - file (required) multipart file
      - user_id (optional) string
      - duration_sec (optional) float for scoring; if missing we approximate as 60s
    """
    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded (use key 'file')"}), 400
    exercise_id = request.form.get("exercise_id")
    user_id = request.form.get("user_id", "anonymous")
    duration_sec = request.form.get("duration_sec")
    try:
        duration_sec = float(duration_sec) if duration_sec is not None else 60.0
    except Exception:
        duration_sec = 60.0

    if not exercise_id:
        return jsonify({"error": "exercise_id is required"}), 400

    uploaded = request.files["file"]
    filename = uploaded.filename or "exercise_upload"
    file_bytes = uploaded.read()
    logger.info("Submit Exercise: received file '%s' size=%d bytes", filename, len(file_bytes))


    # convert to wav if necessary
    if filename.lower().endswith(".wav"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"mobile_ex_{timestamp}_{filename}"
        wav_path = str(UPLOADS_DIR / out_name)
        try:
            with open(wav_path, "wb") as f:
                f.write(file_bytes)
            logger.info("Saved raw Exercise WAV for retraining: %s", wav_path)
        except Exception as e:
            logger.exception("Failed to write Exercise WAV to uploads: %s", e)
            return jsonify({"error": "Failed to write WAV file"}), 500

    else:
        try:
            wav_path = convert_to_wav(file_bytes, filename)
        except Exception as e:
            logger.exception("Conversion to wav failed: %s", e)
            return jsonify({"error": f"Audio conversion failed: {e}"}), 500

    try:
        features, duration_sec = extract_features_from_file(wav_path)
        
        # Graceful handling for empty/tiny audio
        is_empty = False
        if features is None or duration_sec < 1.0:
            is_empty = True
        elif features[-2] < 0.01: # Strict RMS check
            is_empty = True
            
        if is_empty:
             return jsonify({
                "ok": False,
                "error": "Recording too short or quiet.",
                "result": {
                    "prediction": "No Speech",
                    "score": 100,
                    "fluency_score": 100,
                    "stutter_severity": 0,
                    "severity": 0,
                    "feedback": "No clear speech detected.",
                    "duration_received": float(duration_sec)
                }
            })





        # If feature length mismatch, return helpful error
        observed = len(features)
        try: expected = int(FEATURE_COUNT)
        except Exception: expected = FEATURE_COUNT
        if observed != expected:
            return jsonify({
                "error": "Feature count mismatch",
                "expected_feature_count": expected,
                "observed_feature_count": observed
            }), 500

        pred_int, confidence, label = predict_from_features(features, thresh=0.7, debug=False)
        # We don't produce detailed event timestamps here; estimate events_count from model output absence (0) -> 0 else 1
        # If your model can output events, replace the logic below.
        events_count = 1 if label == "Stutter" else 0

        score, metrics = compute_score_from_prediction(label, confidence, events_count, duration_sec=duration_sec)

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "exercise_id": exercise_id,
            "filename": filename,
            "label": label,
            "confidence": float(confidence) if confidence is not None else None,
            "score": score,
            "metrics": metrics
        }
        with LOG_LOCK:
            SUBMISSIONS_LOG.append(entry)
            _save_json_file(SUBMISSIONS_LOG_FILE, SUBMISSIONS_LOG)

        return jsonify({"ok": True, "result": entry})
    finally:
        # try: os.remove(wav_path)
        # except Exception: pass
        pass


@app.route("/progress", methods=["GET"])
def progress():
    """
    Optional query param: user_id to filter results.
    Returns recent predictions and exercise submissions.
    """
    user_id = request.args.get("user_id")
    with LOG_LOCK:
        preds = PREDICTIONS_LOG[-50:]
        subs = SUBMISSIONS_LOG[-50:]
    if user_id:
        preds = [p for p in preds if p.get("user_id") == user_id or p.get("filename", "").startswith(user_id)]
        subs = [s for s in subs if s.get("user_id") == user_id]

    return jsonify({"predictions": preds, "submissions": subs})

@app.route("/flag", methods=["POST"])
def flag_for_retrain():
    """
    Body (json):
      - type: "prediction" or "submission"
      - id: filename or item identifier
      - reason: text
      - user_id: optional
    Appends to retrain queue file for manual review.
    """
    body = request.get_json() or {}
    typ = body.get("type")
    item_id = body.get("id")
    reason = body.get("reason", "")
    user_id = body.get("user_id", "anonymous")
    if not typ or not item_id:
        return jsonify({"error": "type and id required"}), 400
    rec = {"timestamp": datetime.utcnow().isoformat(), "type": typ, "id": item_id, "reason": reason, "user_id": user_id}
    try:
        with open(RETRAIN_QUEUE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return jsonify({"ok": True})
    except Exception:
        logger.exception("Failed to write retrain queue")
        return jsonify({"error": "Failed to flag"}), 500

@app.route("/stats", methods=["GET"])
def stats():
    with LOG_LOCK:
        total = len(PREDICTIONS_LOG)
        stutter_count = sum(1 for p in PREDICTIONS_LOG if p.get("prediction") == "Stutter")
    if total == 0:
        return jsonify({"message": "No predictions yet"})
    return jsonify({
        "total_predictions": total,
        "stutter_predictions": stutter_count,
        "fluent_predictions": total - stutter_count,
        "stutter_percentage": round(stutter_count / total * 100, 2)
    })

# ---------- Startup load and run ----------
def startup_load():
    global MODEL, SCALER, FEATURE_COUNT, MODELS_DIR, PREDICTIONS_LOG, SUBMISSIONS_LOG
    
    # Priority Strategy:
    # 1. Environment Variable (Standard for Cloud/Hugging Face)
    # 2. Local relative models_retrained (Dev/Fallback)
    
    env_dir = os.environ.get("FLUENTA_MODELS_DIR")
    # In Hugging Face, app is at root /app, so models are at /app/models_retrained
    RELATIVE_PATH = SCRIPT_DIR / "models_retrained"

    if env_dir and Path(env_dir).exists():
        MODELS_DIR = Path(env_dir)
        logger.info("Using models from ENVIRONMENT variable: %s", MODELS_DIR)
    elif RELATIVE_PATH.exists() and pick_existing_file(RELATIVE_PATH, PREFERRED_MODEL_FILENAMES):
        MODELS_DIR = RELATIVE_PATH
        logger.info("Using models from PROJECT RELATIVE path: %s", MODELS_DIR)
    else:
        # Fallback for local testing if needed
        D_DRIVE_PATH = Path(r"D:/fluenta_fresh/backend/backend/models_retrained")
        if D_DRIVE_PATH.exists() and pick_existing_file(D_DRIVE_PATH, PREFERRED_MODEL_FILENAMES):
             MODELS_DIR = D_DRIVE_PATH
             logger.info("Using models from hardcoded D: drive: %s", MODELS_DIR)
        else:
             logger.warning("Could not find models directory!")
        
    logger.info("Final MODELS_DIR determined as: %s", MODELS_DIR)




    m, s, fc = safe_load_models(MODELS_DIR)
    globals_ = globals()
    globals_['MODEL'], globals_['SCALER'], globals_['FEATURE_COUNT'] = m, s, fc

    if MODEL is None:
        logger.error("Model not loaded. Check FLUENTA_MODELS_DIR or artifact placement.")
    else:
        logger.info("Models loaded and API ready.")
        logger.info("Loaded model file: %s", MODEL_PATH_LOADED)
        logger.info("Loaded scaler file: %s", SCALER_PATH_LOADED)
        logger.info("Loaded feature_count file: %s", FCOUNT_PATH_LOADED)

    # load persisted logs
    try:
        PREDICTIONS_LOG = _load_json_file(PREDICTIONS_LOG_FILE)
    except Exception:
        PREDICTIONS_LOG = []
    try:
        SUBMISSIONS_LOG = _load_json_file(SUBMISSIONS_LOG_FILE)
    except Exception:
        SUBMISSIONS_LOG = []

if __name__ == "__main__":
    startup_load()
    app.run(host="0.0.0.0", port=5000, debug=False)
