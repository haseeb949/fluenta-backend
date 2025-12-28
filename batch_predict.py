# batch_predict.py
import os, csv, joblib, numpy as np
from pathlib import Path
from feature_extraction import extract_features

MODELS_DIR = Path("C:/Users/X/Downloads/models")  # adjust if needed
CLIPS_DIR = Path("C:/Users/X/Downloads/models/Dataset/clips/clips")
OUT_CSV = Path("C:/Users/X/Downloads/models/backend/results.csv")

model = joblib.load(MODELS_DIR / "stutter_model_final.pkl")
scaler = joblib.load(MODELS_DIR / "scaler_final.pkl")
fc = joblib.load(MODELS_DIR / "feature_count.pkl")
try:
    expected = int(fc)
except Exception:
    expected = int(np.asarray(fc).item())

files = sorted([p for p in CLIPS_DIR.glob("*") if p.suffix.lower() in [".wav",".flac",".mp3",".ogg"]])
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename","prediction","confidence"])
    for p in files:
        feats = extract_features(str(p))
        if feats is None:
            writer.writerow([str(p.name),"ERROR","-"])
            continue
        if len(feats) != expected:
            writer.writerow([str(p.name),"FEATURE_MISMATCH",f"{len(feats)}/{expected}"])
            continue
        X = scaler.transform([feats])
        pred = model.predict(X)[0]
        conf = None
        if hasattr(model, "predict_proba"):
            conf = float(model.predict_proba(X)[0][int(pred)])
        label = "Stutter" if int(pred)==1 else "Non-Stutter"
        writer.writerow([str(p.name), label, "" if conf is None else round(conf,4)])
print("Saved results to", OUT_CSV)
