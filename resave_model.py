# resave_model.py
import joblib
from pathlib import Path

MODELS_DIR = Path("C:/Users/X/Downloads/models")
m = joblib.load(MODELS_DIR / "stutter_model_final.pkl")
joblib.dump(m, MODELS_DIR / "stutter_model_resaved.pkl")
print("Resaved model to stutter_model_resaved.pkl")
