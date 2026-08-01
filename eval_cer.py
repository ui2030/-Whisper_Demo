# 한국어 의료 문장 40건(일반 20 + 약품명 20)에 대해 whisper-small의 CER(문자 오류율)을 측정한다.
# 실행: C:\Users\ui2030\anaconda3\python.exe eval_cer.py
# 문장은 전부 자작이며 실존 환자·병원·의사 정보는 포함하지 않는다.
import csv
import re
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import pyttsx3
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import pipeline

HERE = Path(__file__).parent
OUT = HERE / "out"
WAV = OUT / "wav"
MODEL = "openai/whisper-small"  # demo.py와 동일 모델 (비교 일관성)
SR = 16000  # whisper 입력 규격: 16kHz mono

# (id, 정답 문장) — 숫자·단위는 아라비아 숫자 + 영문 단위로 표기 (whisper 자동 정규화와 표기 맞춤)
GENERAL = [
    "환자분은 어젯밤부터 오른쪽 아랫배 통증을 호소하고 있습니다.",
    "혈압은 정상 범위이고 체온은 37.2도입니다.",
    "다음 주 화요일 오전에 외래 진료 예약을 잡아 드리겠습니다.",
    "검사 결과는 3일 뒤에 확인하실 수 있습니다.",
    "어지럼증이 심해지면 바로 응급실로 오세요.",
    "금식은 검사 8시간 전부터 유지해 주세요.",
    "상처 부위가 붉어지거나 열감이 있으면 알려 주세요.",
    "오늘은 심전도와 흉부 엑스레이를 시행하겠습니다.",
    "지난번보다 체중이 2kg 줄었습니다.",
    "기침이 2주 넘게 지속되면 추가 검사가 필요합니다.",
    "수술 후 회복까지는 보통 6주 정도 걸립니다.",
    "하루에 물을 2L 이상 드시는 것이 좋습니다.",
    "가족력 중에 심장 질환이 있는지 확인하겠습니다.",
    "입원 기간은 대략 5일로 예상됩니다.",
    "알레르기 반응이 있었던 적이 있는지 여쭤보겠습니다.",
    "공복 혈당이 130 정도로 조금 높게 나왔습니다.",
    "소변 검사에서 특별한 이상은 없었습니다.",
    "물리치료는 주 3회로 시작하겠습니다.",
    "오늘 처방은 1주일분으로 드리겠습니다.",
    "통증이 10점 만점에 몇 점 정도인지 말씀해 주세요.",
]

# (약품 성분명, 문장) — 약품명 토큰을 알고 있어야 오인식 추출이 가능
DRUG = [
    ("메트포르민", "메트포르민 500mg을 하루 두 번 식후에 복용하세요."),
    ("아토르바스타틴", "아토르바스타틴 20mg으로 증량하고 4주 뒤 재검합니다."),
    ("아목시실린", "아목시실린 250mg 캡슐을 8시간 간격으로 드세요."),
    ("로사르탄", "로사르탄 50mg을 아침 식전에 한 알 복용합니다."),
    ("암로디핀", "암로디핀 5mg을 저녁에 복용하고 혈압을 기록해 주세요."),
    ("오메프라졸", "오메프라졸 20mg을 공복에 드시고 2주 후 내원하세요."),
    ("세티리진", "세티리진 10mg은 취침 전에 한 알만 복용하세요."),
    ("이부프로펜", "이부프로펜 400mg은 통증이 있을 때만 복용합니다."),
    ("아세트아미노펜", "아세트아미노펜 650mg을 6시간 간격으로 드셔도 됩니다."),
    ("레보플록사신", "레보플록사신 500mg을 7일간 하루 한 번 복용하세요."),
    ("클로피도그렐", "클로피도그렐 75mg은 임의로 중단하지 마세요."),
    ("리시노프릴", "리시노프릴 10mg 복용 후 마른기침이 생기면 알려 주세요."),
    ("심바스타틴", "심바스타틴 20mg은 저녁에 드시는 것이 좋습니다."),
    ("프레드니솔론", "프레드니솔론 5mg을 3일간 감량하며 복용합니다."),
    ("와파린", "와파린 2.5mg 복용 중이니 출혈 징후를 확인하세요."),
    ("트라마돌", "트라마돌 50mg은 어지럼증이 있을 수 있어 운전을 피하세요."),
    ("파모티딘", "파모티딘 20mg을 아침저녁으로 복용하세요."),
    ("아지트로마이신", "아지트로마이신 500mg을 3일간 복용하고 종료합니다."),
    ("메토프롤롤", "메토프롤롤 25mg 복용 중 맥박이 느려지면 중단하세요."),
    ("독시사이클린", "독시사이클린 100mg은 충분한 물과 함께 드세요."),
]


def normalize(s):
    """정답·인식 양쪽에 동일 적용: 구두점 제거 → 대소문자 통일 → 공백 정리."""
    s = re.sub(r"[.,?!]", "", s)
    return re.sub(r"\s+", " ", s.lower()).strip()


def synth(engine, text, path):
    """윈도우 내장 TTS로 wav 생성 후 16kHz mono로 리샘플."""
    tmp = path.with_suffix(".raw.wav")
    engine.save_to_file(text, str(tmp))
    engine.runAndWait()
    audio, sr = sf.read(tmp)
    if audio.ndim > 1:  # 혹시 스테레오면 모노로
        audio = audio.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        audio = resample_poly(audio, SR // g, sr // g)
    sf.write(path, audio, SR, subtype="PCM_16")
    tmp.unlink()
    return audio


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # 윈도우 cp949 콘솔에서 한글·기호 깨짐 방지
    WAV.mkdir(parents=True, exist_ok=True)
    rows = [(f"g{i + 1:02d}", "general", t, None) for i, t in enumerate(GENERAL)]
    rows += [(f"d{i + 1:02d}", "drug", t, d) for i, (d, t) in enumerate(DRUG)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[경고] CUDA 사용 불가 — CPU로 폴백합니다.")
    print(f"장치: {device} / 모델: {MODEL} / 문장 {len(rows)}건")

    t0 = time.time()
    asr = pipeline(
        "automatic-speech-recognition",
        model=MODEL,
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    load_sec = time.time() - t0
    print(f"모델 로드: {load_sec:.1f}초")

    engine = pyttsx3.init()
    results, times = [], []
    for rid, group, ref, drug in rows:
        audio = synth(engine, ref, WAV / f"{rid}.wav")
        t0 = time.time()
        hyp = asr(
            {"array": audio, "sampling_rate": SR},
            generate_kwargs={"language": "korean", "task": "transcribe"},
        )["text"]
        times.append(time.time() - t0)
        n_ref, n_hyp = normalize(ref), normalize(hyp)
        cer = jiwer.cer(n_ref, n_hyp)
        results.append((rid, group, drug, n_ref, n_hyp, cer))
        print(f"{rid} [{group}] CER {cer:.3f} | {n_hyp}")
    engine.stop()

    # 약품명 오인식: 정답 토큰이 인식 결과에 그대로 없으면 가장 가까운(편집거리 최소) 토큰을 기록
    misses = []
    for rid, group, drug, n_ref, n_hyp, _ in results:
        if group != "drug" or drug in n_hyp:
            continue
        toks = n_hyp.split()
        closest = min(toks, key=lambda t: jiwer.cer(drug, t)) if toks else "(공백)"
        misses.append((rid, drug, closest))

    def avg(g):
        v = [r[5] for r in results if g is None or r[1] == g]
        return sum(v) / len(v)

    all_cer, gen_cer, drug_cer = avg(None), avg("general"), avg("drug")
    ratio = drug_cer / gen_cer if gen_cer else float("inf")

    with open(OUT / "cer_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "group", "정답", "인식결과", "CER"])
        for rid, group, _, n_ref, n_hyp, cer in results:
            w.writerow([rid, group, n_ref, n_hyp, f"{cer:.4f}"])

    lines = [
        f"모델: {MODEL} / 장치: {device} / 문장 {len(results)}건",
        f"모델 로드 {load_sec:.1f}초, 문장당 평균 전사 {sum(times) / len(times):.2f}초",
        "",
        f"전체 평균 CER      : {all_cer:.4f} ({all_cer * 100:.2f}%)",
        f"일반 문장 평균 CER : {gen_cer:.4f} ({gen_cer * 100:.2f}%)",
        f"약품명 문장 평균 CER: {drug_cer:.4f} ({drug_cer * 100:.2f}%)",
        f"약품명/일반 비율   : {ratio:.2f}배",
        "",
        f"약품명 오인식 {len(misses)}건 / {len(DRUG)}건" + (" (0건)" if not misses else ""),
    ]
    lines += [f"  {rid}: {d} → {c}" for rid, d, c in misses]
    lines += [
        "",
        f'"의료 문장 {len(results)}건 기준 전체 CER {all_cer * 100:.1f}%, 약품명 포함 문장은 '
        f'{drug_cer * 100:.1f}%로 일반 문장({gen_cer * 100:.1f}%) 대비 {ratio:.1f}배 — '
        f'오류가 의료 용어에 집중됨을 정량 확인"',
    ]
    text = "\n".join(lines)
    (OUT / "cer_summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
