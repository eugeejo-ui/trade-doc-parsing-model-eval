"""
detect.py — 5단계: 규칙 적용 및 탐지율 환산

모델이 읽은 값에 규칙 3종을 적용해 하자를 판정하고,
1.5단계 정답지와 대조해 탐지율 r, 오탐률 p를 구한다.

rules.py의 함수를 그대로 쓴다.
amount_matches는 0단계에서 BPI 8,564건으로 검증된 함수다.
date_before / name_matches는 미검증 (BPI에 대응 필드 없음).
"""

import json
import sys
from pathlib import Path

import pandas as pd

here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next((p for p in [here, *here.parents] if (p / "src").is_dir()), here)
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))

from rules import amount_matches, date_before, name_matches, DEFAULT_EPS

MODELS = ["granite", "paddle"]

# 조건 → (정답지 파일, 규칙 종류, extracted.csv의 추출값 컬럼)
CONDS = [
    ("amount_0.1", "truth_amount_0.1.csv", "amount", "amt_got"),
    ("amount_1",   "truth_amount_1.csv",   "amount", "amt_got"),
    ("amount_10",  "truth_amount_10.csv",  "amount", "amt_got"),
    ("amount_50",  "truth_amount_50.csv",  "amount", "amt_got"),
    ("date_1",     "truth_date_1.csv",     "date",   "date_got"),
    ("date_3",     "truth_date_3.csv",     "date",   "date_got"),
    ("date_7",     "truth_date_7.csv",     "date",   "date_got"),
    ("date_30",    "truth_date_30.csv",    "date",   "date_got"),
    ("name",       "truth_name.csv",       "name",   "name_got"),
]


def to_date(s):
    return pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")


def judge(kind, got, cond):
    """모델이 읽은 값과 조건값을 대조해 하자 여부를 판정한다."""
    if got is None or (isinstance(got, float) and pd.isna(got)):
        return True          # 못 읽으면 대조 불가 → 하자로 올림 (보수적)
    if kind == "amount":
        return not amount_matches(float(got), float(cond), eps=DEFAULT_EPS)
    if kind == "date":
        d1, d2 = to_date(got), to_date(cond)
        if pd.isna(d1) or pd.isna(d2):
            return True
        return not date_before(d1, d2)
    return not name_matches(str(got), str(cond), strict=True)


# ---------- 1. 추출 결과 ----------
E = pd.read_csv(OUT / "extracted.csv", encoding="utf-8-sig")
print(f"[1] 추출 결과 {len(E):,}행 ({E.model.nunique()}모델)")

rows = []
for cond_name, fname, kind, got_col in CONDS:
    T = pd.read_csv(OUT / fname, encoding="utf-8-sig")
    for model in MODELS:
        m = E[E.model == model][["file_name", got_col]]
        j = m.merge(T[["file_name", "cond_value", "is_defect"]], on="file_name")
        for _, r in j.iterrows():
            rows.append({
                "cond": cond_name, "kind": kind, "model": model,
                "file_name": r["file_name"],
                "got": r[got_col], "cond_value": r["cond_value"],
                "truth": bool(r["is_defect"]),
                "pred": judge(kind, r[got_col], r["cond_value"]),
            })

d = pd.DataFrame(rows)
print(f"[2] 판정 {len(d):,}건 ({len(CONDS)}조건 × {len(MODELS)}모델 × {len(d)//18:,}건)")

# ---------- 3. 지표 ----------
res = []
for (cond, model), g in d.groupby(["cond", "model"], sort=False):
    tp = int(( g.truth &  g.pred).sum())
    fn = int(( g.truth & ~g.pred).sum())
    fp = int((~g.truth &  g.pred).sum())
    tn = int((~g.truth & ~g.pred).sum())
    res.append({
        "cond": cond, "model": model, "n": len(g),
        "defects": tp + fn, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "r": round(tp / (tp + fn) * 100, 2) if tp + fn else None,
        "p": round(fp / (fp + tn) * 100, 2) if fp + tn else None,
        "acc": round((tp + tn) / len(g) * 100, 2),
    })
R = pd.DataFrame(res)

print(f"\n[3] 조건별 탐지율 r / 오탐률 p")
print(f"{'조건':<12}{'모델':<10}{'하자':>6}{'r(탐지)':>10}{'p(오탐)':>10}{'정확도':>10}")
print("-" * 58)
for _, r in R.iterrows():
    print(f"{r['cond']:<12}{r['model']:<10}{r['defects']:>6}"
          f"{r['r']:>9.1f}%{r['p']:>9.1f}%{r['acc']:>9.1f}%")

# ---------- 4. 주입 폭 민감도 ----------
print(f"\n[4] 주입 폭 민감도  (동일하면 '탐지율은 하자 크기가 아니라 파싱 정확도가 결정')")
for kind, conds in [("금액", ["amount_0.1", "amount_1", "amount_10", "amount_50"]),
                    ("기한", ["date_1", "date_3", "date_7", "date_30"])]:
    print(f"  {kind}")
    for model in MODELS:
        vals = [f"{R[(R.cond==c)&(R.model==model)]['r'].iloc[0]:.1f}" for c in conds]
        print(f"    {model:<10}{' / '.join(vals)}")

# ---------- 5. 모델 비교 ----------
print(f"\n[5] 모델 간 차이 (r)")
for cond in R.cond.unique():
    g = R[(R.cond == cond) & (R.model == "granite")]["r"].iloc[0]
    p = R[(R.cond == cond) & (R.model == "paddle")]["r"].iloc[0]
    print(f"  {cond:<12} granite {g:>6.1f}%   paddle {p:>6.1f}%   차이 {g-p:>+5.1f}p")

# ---------- 6. 저장 ----------
d.to_csv(OUT / "detection.csv", index=False, encoding="utf-8-sig")
summary = {
    "n_cases": int(len(d) // 18),
    "eps": DEFAULT_EPS,
    "conditions": R.to_dict("records"),
    "note": "amount_matches는 0단계 BPI 검증 완료. date_before/name_matches는 미검증.",
}
(OUT / "detection_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[6] 저장 -> outputs/detection.csv, outputs/detection_summary.json")