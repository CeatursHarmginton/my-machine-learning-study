import numpy as np
import torch.nn as nn
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
from transformers import AutoModelForCTC, AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor

# 1. Đọc WAV & Trích xuất phổ STFT bằng scipy chuẩn của OlpAI'25 (không cần cài đặt torchaudio!):
def load_wav_16k(path: str) -> tuple[np.ndarray, np.ndarray]:
    sr, wav = wavfile.read(path)
    wav = wav.astype(np.float32) / (32768.0 if wav.dtype == np.int16 else 1.0)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = resample_poly(wav, 16000, sr).astype(np.float32)
    _, _, spec = stft(wav, fs=16000, nperseg=400, noverlap=240)
    return wav, np.log1p(np.abs(spec))

# 2. AutoProcessor của Wav2Vec2 & Whisper tự trích xuất đặc trưng âm thanh trực tiếp từ mảng NumPy 16kHz:
ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True)

def load_asr_models(w2v_ckpt: str = "facebook/wav2vec2-base-960h", wh_ckpt: str = "openai/whisper-small"):
    w2v = (AutoProcessor.from_pretrained(w2v_ckpt), AutoModelForCTC.from_pretrained(w2v_ckpt))
    wh = (WhisperProcessor.from_pretrained(wh_ckpt), WhisperForConditionalGeneration.from_pretrained(wh_ckpt))
    return w2v, wh
