# ml_utils.py
import os
import numpy as np
from pydub import AudioSegment
import librosa
import soundfile as sf

SAMPLE_RATE = 16000

def to_wav_mono16(in_path, out_path, target_sr=SAMPLE_RATE):
    """Convert to mono WAV 16-bit PCM and resample to SAMPLE_RATE."""
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_frame_rate(target_sr).set_channels(1).set_sample_width(2)
    audio.export(out_path, format="wav")
    return out_path

def load_audio_mono(path, sr=SAMPLE_RATE):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def trim_silence(y, top_db=30):
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return y
    parts = [y[start:end] for start, end in intervals]
    return np.concatenate(parts)

def extract_mfcc(y, sr=SAMPLE_RATE, n_mfcc=20, hop_length=512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    # shape: (n_mfcc, t). We return mean+std for simple vector input
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feat.astype(np.float32)

class ModelWrapper:
    """Replace predict() with your model code. Keep interface same: returns dict."""
    def __init__(self, model_path=None):
        self.model_path = model_path
        # TODO: load Torch/TF/ONNX model here if available
        self.model = None

    def predict_from_features(self, features):
        # placeholder: if mean MFCC amplitude > threshold say "Stutter"
        score = float(np.tanh(np.abs(features).mean()) )  # dummy 0..1-ish
        label = "Stutter" if score > 0.4 else "NoStutter"
        events = []  # e.g. [{"t":1.2,"type":"repeat"}]
        return {"label": label, "confidence": score, "events": events}

    def predict_from_file(self, audio_path):
        # full pipeline: convert -> load -> trim -> extract -> predict
        tmp = audio_path
        if not audio_path.lower().endswith(".wav"):
            tmp = audio_path + ".wav"
            to_wav_mono16(audio_path, tmp)
        y, sr = load_audio_mono(tmp)
        y = trim_silence(y)
        feats = extract_mfcc(y, sr)
        return self.predict_from_features(feats)
