"""
retrain_small.py

Usage (Windows PowerShell / CMD):
1) Place your custom clips:
   Dataset/custom/fluent/*.wav
   Dataset/custom/stutter/*.wav

2) Activate your venv (if needed) and run:
   python retrain_small.py

Options:
 - If you already have a training CSV or existing model, put its path into EXISTING_MODEL_PATH or EXISTING_TRAIN_CSV variables below.
 - The script will create: stutter_model_retrained.pkl, scaler_retrained.pkl, feature_count.pkl in the current directory (or in --out-dir).
"""

import os
import argparse
import logging
import joblib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

# Optional XGBoost (if installed) — faster and often better for tabular features
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# ---------------- Config ----------------
EXPECTED_FEATURE_LEN = 60
RANDOM_SEED = 42
N_ESTIMATORS = 200
OUT_MODEL = "stutter_model_retrained.pkl"
OUT_SCALER = "scaler_retrained.pkl"
OUT_FCOUNT = "feature_count.pkl"
CALIBRATED_MODEL = "stutter_model_retrained_calibrated.pkl"

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retrain_small")

# ---------------- Audio utils (match your app.py) ----------------
def preprocess_audio(y: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    # Keep minimal/resilient preprocessing: trim, normalize, resample
    try:
        y, _ = librosa.effects.trim(y, top_db=30)
    except Exception:
        pass

    try:
        y = librosa.util.normalize(y)
    except Exception:
        pass
    if sr != target_sr:
        try:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception:
            pass
    min_length = int(0.5 * sr)
    if len(y) < min_length:
        y = np.pad(y, (0, max(0, min_length - len(y))), mode="constant")
    return y, sr

def extract_60_features(path: str) -> np.ndarray:
    """
    Extract the same 60 features used by your app.py:
    - 20 MFCC means + 20 MFCC stds = 40
    - spectral centroid mean/std = 2
    - spectral rolloff mean/std = 2
    - zcr mean/std = 2
    - chroma mean (12) = 12
    - rms mean/std = 2
    Total = 60
    """
    try:
        y, sr = librosa.load(path, sr=16000)
        y, sr = preprocess_audio(y, sr, target_sr=16000)


        feats: List[float] = []

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
        if arr.size != EXPECTED_FEATURE_LEN:
            raise ValueError(f"Extracted feature length {arr.size} != expected {EXPECTED_FEATURE_LEN}")
        return arr
    except Exception as e:
        logger.exception("Failed to extract features from %s: %s", path, e)
        raise

# ---------------- Data loader ----------------
def load_custom_dataset(base_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Looks for:
      base_dir/fluent/*.wav  -> label 0
      base_dir/stutter/*.wav -> label 1
    Returns X (n_samples, 60), y (n_samples,), files list
    """
    X = []
    y = []
    files = []

    fluent_dir = base_dir / "fluent"
    stutter_dir = base_dir / "stutter"

    for p in (fluent_dir.glob("*.wav") if fluent_dir.exists() else []):
        try:
            feats = extract_60_features(str(p))
            X.append(feats)
            y.append(0)
            files.append(str(p))
        except Exception:
            logger.warning("Skipping file (failed extract): %s", p)

    for p in (stutter_dir.glob("*.wav") if stutter_dir.exists() else []):
        try:
            feats = extract_60_features(str(p))
            X.append(feats)
            y.append(1)
            files.append(str(p))
        except Exception:
            logger.warning("Skipping file (failed extract): %s", p)

    if not X:
        raise RuntimeError(f"No valid audio files found in {base_dir}. Place wavs under fluent/ and stutter/")

    X = np.stack(X)
    y = np.asarray(y, dtype=int)
    return X, y, files

# ---------------- Training routine ----------------
def train_and_save(X: np.ndarray, y: np.ndarray, out_dir: Path, use_xgb: bool = False):
    rng = np.random.RandomState(RANDOM_SEED)

    # split for quick eval
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rng, stratify=y)

    # scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # choose model
    if use_xgb and XGBOOST_AVAILABLE:
        logger.info("Using XGBoost classifier")
        clf = XGBClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED, use_label_encoder=False, eval_metric="logloss")
    else:
        logger.info("Using RandomForest (class_weight='balanced')")
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1)

    # basic CV (optional)
    logger.info("Cross-validating (5-fold) on training set...")
    try:
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring="f1")
        logger.info("CV F1 scores: %s | mean=%.4f", cv_scores, float(np.mean(cv_scores)))
    except Exception:
        logger.warning("Cross-validation failed (proceeding to train anyway)")

    # fit
    clf.fit(X_train_scaled, y_train)
    logger.info("Model training complete.")

    # calibrate probabilities
    try:
        logger.info("Calibrating classifier probabilities (sigmoid/Platt)...")
        calibrator = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv="prefit")

        calibrator.fit(X_test_scaled, y_test)
        model_to_save = calibrator
        logger.info("Calibration complete.")
    except Exception as e:
        logger.warning("Calibration failed: %s. Saving uncalibrated model.", e)
        model_to_save = clf

    # evaluate quickly
    try:
        preds = model_to_save.predict(X_test_scaled)
        probs = model_to_save.predict_proba(X_test_scaled)[:, 1] if hasattr(model_to_save, "predict_proba") else None
        logger.info("Test classification report:\n%s", classification_report(y_test, preds))
        logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, preds))
    except Exception:
        logger.warning("Evaluation step failed.")

    # save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / OUT_MODEL
    scaler_path = out_dir / OUT_SCALER
    fcount_path = out_dir / OUT_FCOUNT

    joblib.dump(model_to_save, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(int(EXPECTED_FEATURE_LEN), fcount_path)

    logger.info("Saved model -> %s", model_path)
    logger.info("Saved scaler -> %s", scaler_path)
    logger.info("Saved feature count -> %s", fcount_path)

    return model_path, scaler_path, fcount_path

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="Dataset/custom", help="Folder containing fluent/ and stutter/ subfolders")
    parser.add_argument("--out-dir", type=str, default="models_retrained", help="Output folder to save model/scaler")
    parser.add_argument("--use-xgb", action="store_true", help="Use XGBoost if available")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    logger.info("Loading custom dataset from: %s", data_dir)
    X, y, files = load_custom_dataset(data_dir)
    logger.info("Loaded %d samples (files shown below):", len(files))
    for f in files:
        logger.info(" - %s", f)

    # train and save
    model_path, scaler_path, fcount_path = train_and_save(X, y, out_dir, use_xgb=args.use_xgb)
    logger.info("Retraining finished. Artifacts at: %s", out_dir.resolve())

if __name__ == "__main__":
    main()
