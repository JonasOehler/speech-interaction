import argparse
import requests
import csv
import itertools
import io
import librosa
import re
import time
import os
import numpy as np
import soundfile as sf
from corpus import create_corpus
from jiwer import wer

parser=argparse.ArgumentParser()
parser.add_argument(
    "--speech_corpus",
    help="define speech corpus",
    type=str,
    default="fleurs",
    required=False
)

parser.add_argument(
    "--dataset_samples",
    help="number of audio samples to evaluate",
    type=int, 
    default=10
)

parser.add_argument(
    "--model",
    type=str,
    default="default",
    help="ASR model identifier (for logging only)"
)

parser.add_argument(
    "--filename",
    type=str,
    default="asr_results.csv",
    help="name of the resulting csv file",
    required=False
)

args=parser.parse_args()

ASR_URL = "http://127.0.0.1:8080/inference"
RESULTS_CSV = "results/" + args.filename
RATE = 16000

def encode_wav(audio_array: np.ndarray, orig_sr: int, target_sr: int = RATE) -> bytes:
    if orig_sr != target_sr:
        audio_array = librosa.resample(audio_array.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)

    buf = io.BytesIO()
    sf.write(buf, audio_array, target_sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def send_to_asr(url: str, wav_bytes: bytes) -> dict:
    files = {"file": ("sample.wav", wav_bytes, "audio/wav")}
    r = requests.post(url, files=files, timeout=60)
    r.raise_for_status()
    return r.json()

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\[.*?\]", "", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

corpus = create_corpus(args)
print('Set speech corpus')
print(corpus.get_speech_corpus_details())
ds = corpus.load_speech_corpus()

fieldnames = [
    "index",
    "model",
    "reference",
    "hypothesis",
    "audio_sec",
    "infer_sec",
    "x_real_time",
    "wer"
]

write_header = not os.path.exists(RESULTS_CSV)

with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for i, ex in enumerate(itertools.islice(ds, args.dataset_samples)):
        ref = ex.get("sentence") or ex.get("transcription") or ex.get("text") or ""
        aud = ex["audio"]
        audio_array = aud["array"]
        sr = aud["sampling_rate"]

        # Audio Duration 
        audio_sec = len(audio_array) / sr

        # ASR Inference
        wav_bytes = encode_wav(audio_array, orig_sr=sr, target_sr=RATE)
        t0 = time.perf_counter()
        result = send_to_asr(ASR_URL, wav_bytes)
        infer_sec = time.perf_counter() - t0

        hyp = result.get("text", "")

        # ASR Speed
        x_real_time = audio_sec / infer_sec

        # WER
        ref_n = normalize_text(ref)
        hyp_n = normalize_text(hyp)
        sample_wer = wer(ref_n, hyp_n)

        print(f"\n[{i+1:03d}/{args.dataset_samples}] sr={sr} -> {RATE}")
        print("REFERENCE:", ref)
        print("ASR :", hyp)

        writer.writerow({
                "index": i,
                "model": args.model,
                "reference": ref,
                "hypothesis": hyp,
                "audio_sec": audio_sec,
                "infer_sec": infer_sec,
                "x_real_time": x_real_time,
                "wer": sample_wer
        })