# feature_extraction.py
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
import logging
from typing import Optional

logger = logging.getLogger("feature_extraction")

def preprocess_audio(y, sr, target_sr=16000):
    # 1. Basic energy check
    if len(y) == 0:
        return y, sr
    
    rms_energy = np.sqrt(np.mean(y**2))
    duration = len(y) / sr
    if duration < 1.0 or rms_energy < 0.01:

        # Return empty for too short or quiet audio
        return np.array([]), sr


    # 2. Trim silence with a slightly more relaxed threshold (30db instead of 20)
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        if len(y_trimmed) >= int(0.1 * sr):
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
    if 0 < len(y) < min_length:
        y = np.pad(y, (0, max(0, min_length - len(y))), mode="constant")
    
    return y, sr


def extract_features(path: str) -> Optional[np.ndarray]:
    """
    Extract exactly the 60 features used by the model:
    - MFCC (20) mean + std  => 40
    - spectral centroid mean/std => 2
    - spectral rolloff mean/std => 2
    - zcr mean/std => 2
    - chroma mean (12) => 12
    - rms mean/std => 2
    Total = 60
    """
    try:
        # Use librosa.load for better robustness
        y, sr = librosa.load(path, sr=16000)
        y, sr = preprocess_audio(y, sr, target_sr=16000)

        if len(y) == 0:
            logger.error("Audio is too quiet or empty after preprocessing.")
            return None


        feats = []
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        feats.extend(np.mean(mfccs.T, axis=0).tolist())
        feats.extend(np.std(mfccs.T, axis=0).tolist())

        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feats.append(float(np.mean(spectral_centroids))); feats.append(float(np.std(spectral_centroids)))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        feats.append(float(np.mean(spectral_rolloff))); feats.append(float(np.std(spectral_rolloff)))

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feats.append(float(np.mean(zcr))); feats.append(float(np.std(zcr)))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats.extend(np.mean(chroma.T, axis=0).tolist())

        rms = librosa.feature.rms(y=y)[0]
        feats.append(float(np.mean(rms))); feats.append(float(np.std(rms)))

        return np.array(feats, dtype=np.float32)
    except Exception as e:
        logger.exception("Failed to extract features for %s: %s", path, e)
        return None
