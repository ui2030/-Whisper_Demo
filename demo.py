# Whisper STT 데모 — Hugging Face에서 openai/whisper-small을 받아 한국어 의료 문장의 말을 글자로 받아써보는 프로젝트(음성→텍스트)
#음성 테스트는 나의 음성이 아닌 TTS의 음성으로 하였다.
# 실행: C:\Users\ui2030\anaconda3\python.exe demo.py   (아나콘다 가상환경에서 사용함.)
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import pipeline

HERE = Path(__file__).parent
MODEL = "openai/whisper-small"  # 약 967MB, 첫 실행 때 자동 다운로드

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"장치: {device} / 모델: {MODEL}")

t0 = time.time()
asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL,
    device=device,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
print(f"모델 로드: {time.time() - t0:.1f}초")

audio, sr = sf.read(HERE / "sample.wav")
print(f"음성 길이: {len(audio) / sr:.1f}초 (샘플레이트 {sr}Hz)")

t0 = time.time()
result = asr(
    {"array": audio, "sampling_rate": sr},
    generate_kwargs={"language": "korean", "task": "transcribe"},
    return_timestamps=True,  # 30초 넘는 음성도 처리 가능하게
)
elapsed = time.time() - t0

original = (HERE / "original.txt").read_text(encoding="utf-8-sig").strip()
print(f"\n말을 글자로 옮기는 시간: {elapsed:.1f}초")
print(f"\n[원문]\n{original}")
print(f"\n[Whisper 전사 결과]\n{result['text'].strip()}")
# 의료 용어(메트포르민, 상복부, 오심 등)에서 틀리는 부분이 있다. 의료 분야의 특화 파인튜닝이 필요한 이유라고 생각한다
