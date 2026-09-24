"""
breakeven.py — 6단계: 손익분기 계산

AI 도입이 언제부터 이득인가를 구한다.
두 모델 중 무엇을 고를지가 아니다 (둘 다 Apache 2.0 무료).

브리핑 6절 수식
  절감 = N×d×r×R − N×d×r×t_f − N×(1-d)×p×t_v − C
  N*   = C / [ d×r×R − d×r×t_f − (1-d)×p×t_v ]

d, r, p 외의 값은 전부 입력이다.
개별 기업의 신용장 건수·운영비는 공개 자료가 없으므로
추정치를 만들지 않고 R 단위로 정규화한 표로 제시한다.

R(재제출 1회 비용)에 대하여
  시간이 아니라 비용이다. 다음을 모두 포함한다.
    · 서류 재작성 인건비
    · 은행 재심사 대기 5영업일(UCP 600)의 기회비용  ← 지배적
        대금 회수 지연 자금조달 비용, 납기 위약금,
        신용장 유효기일 초과 위험, 거래처 신뢰도 손상
  브리핑 6절이 "R이 t_f·t_v와 자릿수가 다르다"고 한 이유가 이것이다.
"""

import json
import sys
from pathlib import Path

import pandas as pd

here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next((p for p in [here, *here.parents] if (p / "src").is_dir()), here)
OUT = ROOT / "outputs"

# ---------- 고정값 ----------
D = 0.70          # 하자 발생률. ICC 1차 제시 거절률 65~80% 중 브리핑 기본값
                  # 주의: 최종 지급거절률이 아니라 1차 제시 거절률
                  #       이미 사람이 점검한 뒤의 값이므로 점검 항을 따로 빼면 이중 차감

# ---------- 5단계 실험값 ----------
R_DETECT = 1.000  # 탐지율 r  (granite/paddle 동일)
P_FALSE  = 0.025  # 오탐률 p  (granite/paddle 동일)


def saving_per_doc(d=D, r=R_DETECT, p=P_FALSE, R=1.0, t_f=0.0, t_v=0.0):
    """건당 순절감액. 단위는 R·t_f·t_v와 같다."""
    return d * r * R - d * r * t_f - (1 - d) * p * t_v


def breakeven_n(C, **kw):
    """손익분기 건수. 건당 순절감이 0 이하면 None (손익분기 없음)."""
    s = saving_per_doc(**kw)
    return (C / s) if s > 0 else None


# ---------- 1. 전제 ----------
print("[1] 전제")
print(f"    하자 발생률 d = {D:.2f}   (ICC 1차 제시 거절률 65~80%)")
print(f"    탐지율     r = {R_DETECT:.3f}  (5단계 실험, 두 모델 동일)")
print(f"    오탐률     p = {P_FALSE:.3f}  (5단계 실험, 두 모델 동일)")
print(f"    나머지(N, R, t_f, t_v, C)는 입력값")
print(f"    모든 금액은 R(재제출 1회 비용) 단위로 정규화한다.")

# ---------- 2. 건당 순절감 ----------
print(f"\n[2] 건당 순절감  (R = 1.0 으로 정규화)")
print(f"{'t_f/R':>8}{'t_v/R':>8}{'건당 순절감':>14}   해석")
print("-" * 58)
SCEN = [(0.00, 0.00), (0.02, 0.02), (0.05, 0.05), (0.10, 0.10), (0.20, 0.20)]
base = {}
for tf, tv in SCEN:
    s = saving_per_doc(R=1.0, t_f=tf, t_v=tv)
    base[(tf, tv)] = s
    note = "수정·확인 비용 무시" if tf == 0 else f"수정·확인이 R의 {tf:.0%} 수준"
    print(f"{tf:>8.2f}{tv:>8.2f}{s:>14.4f}   {note}")
print(f"    → 수정·확인 비용을 R의 20%로 크게 잡아도 순절감이 0.56 남는다.")

# ---------- 3. 손익분기 표 ----------
print(f"\n[3] 손익분기 N  (C를 '재제출 M건 상당'으로 표현)")
print(f"{'C (R 단위)':>12}" + "".join(f"{f't_f=t_v={tf:.0%}':>14}" for tf, _ in SCEN))
print("-" * (12 + 14 * len(SCEN)))

C_LIST = [1, 5, 10, 50, 100, 500]
rows = []
for C in C_LIST:
    cells = []
    for tf, tv in SCEN:
        n = breakeven_n(C, R=1.0, t_f=tf, t_v=tv)
        cells.append(f"{n:>13.0f}건" if n else f"{'없음':>14}")
        rows.append({"C_in_R": C, "t_f_ratio": tf, "t_v_ratio": tv,
                     "saving_per_doc": round(base[(tf, tv)], 4),
                     "breakeven_N": round(n) if n else None})
    print(f"{C:>12}" + "".join(cells))

n_ref = breakeven_n(10, R=1.0, t_f=.05, t_v=.05)
print(f"\n    읽는 법: 월 고정비가 '재제출 10건 값'이고 수정·확인 비용이")
print(f"             재제출의 5% 수준이면, 월 {n_ref:.0f}건 이상부터 이득")
print(f"    관계: N* ≈ C / 0.66 ≈ C × 1.5  (거의 비례)")

# ---------- 4. R 민감도 ----------
# C를 고정하고 R만 바꾼다.
# C도 R에 비례시키면 분자·분모가 같이 움직여 손익분기가 불변이 되는데,
# 그것은 수식의 동어반복이지 결과가 아니다.
print(f"\n[4] R 민감도  (C 고정, R만 변동)")
print(f"    브리핑 6절: R이 지배 변수")
print(f"    R = 재작성 인건비 + 재심사 대기 5영업일의 기회비용")
print(f"    t_f=t_v=R의 5%, C = 기준 R의 10배로 고정")
print(f"{'R (기준 대비)':>14}{'손익분기 N':>14}   해석")
print("-" * 58)
C_FIX = 10.0
for r_mult, note in [(0.25, "자금 여유 큰 대기업"),
                     (0.50, ""),
                     (1.00, "기준"),
                     (2.00, ""),
                     (4.00, "자금 회전 빡빡한 중소기업"),
                     (8.00, "납기 위약금 있는 계약")]:
    n = breakeven_n(C_FIX, R=r_mult, t_f=0.05 * r_mult, t_v=0.05 * r_mult)
    print(f"{r_mult:>13.2f}배{n:>13.0f}건   {note}")
print(f"    → 재제출 비용이 클수록 손익분기가 낮아진다. R이 4배면 필요 건수는 1/4.")
print(f"      5영업일 대기가 자금 회전에 치명적인 기업이 가장 빨리 본전을 뽑는다.")

# ---------- 5. 탐지율 민감도 ----------
print(f"\n[5] 탐지율 민감도")
print(f"    C = 재제출 10건 상당, t_f=t_v=5%")
print(f"{'r (탐지율)':>12}{'손익분기 N':>14}")
print("-" * 30)
for r in [0.60, 0.70, 0.80, 0.90, 0.95, 1.00]:
    n = breakeven_n(10, r=r, R=1.0, t_f=0.05, t_v=0.05)
    print(f"{r:>11.0%}{n:>13.0f}건")
n80 = breakeven_n(10, r=0.8, R=1.0, t_f=.05, t_v=.05)
n100 = breakeven_n(10, r=1.0, R=1.0, t_f=.05, t_v=.05)
print(f"    → 탐지율이 20%p 떨어져도 손익분기는 {n100:.0f}건 → {n80:.0f}건, "
      f"{n80-n100:.0f}건 차이에 그친다.")
print(f"      모델 성능보다 'C가 재제출 몇 건에 해당하는가'가 결정적이다.")

# ---------- 6. 오탐률 민감도 ----------
print(f"\n[6] 오탐률 민감도")
print(f"    C = 재제출 10건 상당, t_f=t_v=5%, r=100%")
print(f"{'p (오탐률)':>12}{'손익분기 N':>14}")
print("-" * 30)
for p in [0.025, 0.05, 0.10, 0.30, 0.50]:
    n = breakeven_n(10, p=p, R=1.0, t_f=0.05, t_v=0.05)
    print(f"{p:>11.1%}{n:>13.0f}건")
print(f"    → 오탐이 20배(2.5%→50%)로 늘어도 손익분기는 거의 변하지 않는다.")
print(f"      오탐 확인 비용(t_v)이 R에 비해 작기 때문이다.")

# ---------- 7. 모델 비교 ----------
print(f"\n[7] 모델 비교")
print(f"    r, p가 동일하므로 손익분기 N도 동일하다.")
print(f"    차이는 비용 수식이 아니라 배포 부담에서 나온다.")
print(f"{'':>16}{'Granite':>12}{'PaddleOCR':>12}")
print("-" * 40)
for label, g, p in [("탐지율 r", "100.0%", "100.0%"),
                    ("오탐률 p", "2.5%", "2.5%"),
                    ("손익분기 N", "동일", "동일"),
                    ("모델 크기", "515MB", "1.92GB"),
                    ("장당 소요(T4)", "26.3초", "18.0초")]:
    print(f"{label:>16}{g:>12}{p:>12}")

# ---------- 8. 한계 ----------
print(f"\n[8] 한계")
print(f"    R(재제출 1회 비용)과 C(월 고정 운영비)의 절대 금액은 알 수 없다.")
print(f"    이 계산은 'C가 R의 몇 배인가'라는 비율 하나로 답을 내는 틀이다.")
print(f"    모르는 값을 둘에서 하나로 줄였을 뿐, 없앤 것은 아니다.")

# ---------- 9. 저장 ----------
pd.DataFrame(rows).to_csv(OUT / "breakeven.csv", index=False, encoding="utf-8-sig")
summary = {
    "fixed": {"d": D, "d_source": "ICC 1차 제시 거절률 65~80%, 브리핑 기본값 0.7"},
    "experiment": {"r": R_DETECT, "p": P_FALSE,
                   "note": "5단계, granite/paddle 동일"},
    "inputs": ["N", "R", "t_f", "t_v", "C"],
    "normalization": "모든 금액을 R(재제출 1회 비용) 단위로 정규화",
    "R_definition": "재작성 인건비 + 재심사 대기 5영업일(UCP 600)의 기회비용. "
                    "대금 회수 지연 자금조달 비용, 납기 위약금, "
                    "신용장 유효기일 초과 위험, 거래처 신뢰도 손상 포함",
    "key_finding": "탐지율이 20%p 떨어져도 손익분기는 소폭만 변한다. "
                   "모델 성능보다 C/R 비율이 결정적이다.",
    "model_comparison": {
        "granite": {"r": 1.0, "p": 0.025, "size_mb": 515, "sec_per_doc_t4": 26.3},
        "paddle":  {"r": 1.0, "p": 0.025, "size_mb": 1920, "sec_per_doc_t4": 18.0}},
    "limitation": "R과 C의 절대 금액은 공개 자료가 없어 알 수 없다. "
                  "비율로 환원했을 뿐 불확실성이 사라진 것은 아니다.",
}
(OUT / "breakeven.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[9] 저장 -> outputs/breakeven.csv, outputs/breakeven.json")