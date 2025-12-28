
import joblib
import numpy as np
from pathlib import Path
import glob

MODEL_DIR = Path(r"C:/Users/X/Downloads/models/backend/models_retrained")
MODEL_FILE = "stutter_model_retrained.pkl"
SCALER_FILE = "scaler_retrained.pkl"

def test_model():
    model_path = MODEL_DIR / MODEL_FILE
    scaler_path = MODEL_DIR / SCALER_FILE
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"Loaded model type: {type(model)}")
    
    # helper
    def _print_pred(feats, label):
        X_scaled = scaler.transform([feats])
        pred = model.predict(X_scaled)
        proba = model.predict_proba(X_scaled) if hasattr(model, "predict_proba") else "N/A"
        print(f"{label}: Pred={pred}, Proba={proba}")

    print("\n--- Testing Fluent Samples ---")
    fluent_files = glob.glob("Dataset/custom/fluent/*.wav")[:3]
    for wav_f in fluent_files:
        try:
            from feature_extraction import extract_features
            feats = extract_features(wav_f)
            if feats is not None:
                _print_pred(feats, f"File: {wav_f}")
        except Exception as e:
            print(f"Error testing file {wav_f}: {e}")

    print("\n--- Testing Stutter Samples ---")
    stutter_files = glob.glob("Dataset/custom/stutter/*.wav")[:3]
    for wav_f in stutter_files:
        try:
            from feature_extraction import extract_features
            feats = extract_features(wav_f)
            if feats is not None:
                _print_pred(feats, f"File: {wav_f}")
        except Exception as e:
            print(f"Error testing file {wav_f}: {e}")

    print("\n--- Testing with Random Features ---")
    for i in range(2):
        random_features = np.random.randn(60)
        _print_pred(random_features, f"Random {i}")

    print("\n--- Testing with ZERO features ---")
    zero_features = np.zeros(60)
    _print_pred(zero_features, "Zeros")

    print("\n--- Testing with silent.wav ---")
    if Path("silent.wav").exists():
        try:
            from feature_extraction import extract_features
            feats = extract_features("silent.wav")
            if feats is not None:
                _print_pred(feats, "File: silent.wav")
        except Exception as e:
            print(f"Error testing silent.wav: {e}")


if __name__ == "__main__":
    test_model()
