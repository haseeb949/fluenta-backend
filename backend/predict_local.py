"""
predict_local.py

Standalone script to predict a single WAV file using retrained artifacts.

Usage:
  python predict_local.py               # uses default test file (/mnt/data/fluent_1.wav)
  python predict_local.py path/to/file.wav
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import soundfile as sf
import librosa

# Configuration: path to retrained artifacts (matches retrain output)
MODEL_DIR = Path(r"D:/fluenta_fresh/backend/backend/models_retrained")
MODEL_FILE = "stutter_model_retrained.pkl"
SCALER_FILE = "scaler_retrained.pkl"
FCOUNT_FILE = "feature_count.pkl"

THRESH = 0.8  # threshold for labeling stutter

# ---------- Feature extraction (same as app.py) ----------
def preprocess_audio(y: np.ndarray, sr: int, target_sr: int = 16000):
    # 1. Energy check
    if len(y) == 0: return y, sr
    rms = np.sqrt(np.mean(y**2))
    duration = len(y) / sr
    if duration < 1.0 or rms < 0.01: return np.array([]), sr




    # 2. Trim
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        if len(y_trimmed) >= int(0.1 * sr): y = y_trimmed
    except Exception: pass

    # 3. Normalize
    try: y = librosa.util.normalize(y)
    except Exception: pass

    # 4. Resample
    if sr != target_sr:
        try:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception: pass

    # 5. Padding
    min_length = int(0.5 * sr)
    if 0 < len(y) < min_length:
        y = np.pad(y, (0, max(0, min_length - len(y))), mode="constant")
    return y, sr


def extract_60_features(path: str):
    # Use librosa.load for better robustness (supports more formats than soundfile)
    y, sr = librosa.load(path, sr=16000)
    y, sr = preprocess_audio(y, sr, target_sr=16000)

    if len(y) == 0:
        raise ValueError("Audio is too quiet or silent. Please provide a clearer recording.")

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
    arr = np.asarray(feats, dtype=np.float32)
    if arr.size != 60:
        raise ValueError(f"Extracted feature length {arr.size} != expected 60")
    return arr

# ---------- Load artifacts ----------
def load_artifacts(model_dir: Path = MODEL_DIR):
    model_path = model_dir / MODEL_FILE
    scaler_path = model_dir / SCALER_FILE
    fcount_path = model_dir / FCOUNT_FILE
    if not model_path.exists() or not scaler_path.exists() or not fcount_path.exists():
        raise FileNotFoundError(f"Missing model/scaler/feature_count in {model_dir}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    fcount = joblib.load(fcount_path)
    try:
        fcount = int(fcount)
    except Exception:
        fcount = int(np.asarray(fcount).item())
    return model, scaler, fcount

def predict_file(model, scaler, features: np.ndarray, thresh: float = THRESH):
    X_scaled = scaler.transform([features])
    raw_pred = int(model.predict(X_scaled)[0])
    p_stutter = None
    if hasattr(model, "predict_proba"):
        raw_proba = model.predict_proba(X_scaled)[0]
        idx = 1 if len(raw_proba) > 1 else 0
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if 1 in classes:
                idx = classes.index(1)
        if 0 <= idx < len(raw_proba):
            p_stutter = float(raw_proba[idx])
    # decide label
    if p_stutter is not None:
        if p_stutter >= thresh:
            label = "Stutter"
            confidence = p_stutter
        else:
            label = "Non-Stutter"
            confidence = 1.0 - p_stutter
    else:
        label = "Stutter" if raw_pred == 1 else "Non-Stutter"
        confidence = None
    return raw_pred, confidence, label, p_stutter

# ---------- Main ----------
def main():
    # default test file (uploaded earlier)
    default_fp = Path("/mnt/data/fluent_1.wav")
    fp = Path(sys.argv[1]) if len(sys.argv) > 1 else default_fp
    if not fp.exists():
        print("File not found:", fp)
        return
    model, scaler, fcount = load_artifacts(MODEL_DIR)
    print("Loaded model from:", MODEL_DIR)
    feats = extract_60_features(str(fp))
    raw_pred, confidence, label, p_stutter = predict_file(model, scaler, feats, thresh=THRESH)
    print("=== Prediction ===")
    print("File:", fp)
    print("Raw predicted class:", raw_pred)
    if p_stutter is not None:
        print(f"Probability(stutter): {p_stutter:.4f}")
        print(f"Used threshold: {THRESH}")
    print("Final label:", label)
    if confidence is not None:
        print(f"Confidence (used for label): {confidence*100:.2f}%")
    else:
        print("Confidence: not available (model.predict_proba missing)")
    print("==================")

if __name__ == "__main__":
    main()
