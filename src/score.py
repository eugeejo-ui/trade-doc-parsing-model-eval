"""
score.py — 4단계 채점

두 모델의 추출값을 정답과 대조해 필드별 정확도를 낸다.
브리핑 9절 축 1에 해당한다.

extracted : 값을 뽑아냈는가
correct   : 뽑은 값이 정답과 같은가
둘을 구분해야 '못 읽은 것'과 '잘못 읽은 것'이 갈린다.
"""

import json
import sys
from pathlib import Path

import pandas as pd

here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next((p for p in [here, *here.parents] if (p / "src").is_dir()), here)
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))

from extract_fields import extract          # 두 모델 공통 추출기
from rules import amount_matches, name_matches, DEFAULT_EPS

MODELS = {"granite": OUT / "granite_raw.jsonl",
          "paddle":  OUT / "paddle_raw.jsonl"}

# ---------- 1. 정답지 ----------
# 조건별 정답지는 doc_value가 동일하므로 아무거나 하나에서 참값을 읽는다
t_amt  = pd.read_csv(OUT / "truth_amount_1.csv",  encoding="utf-8-sig")
t_date = pd.read_csv(OUT / "truth_date_1.csv",    encoding="utf-8-sig")
t_name = pd.read_csv(OUT / "truth_name.csv",      encoding="utf-8-sig")

truth = (t_amt[["file_name", "doc_value"]].rename(columns={"doc_value": "total"})
         .merge(t_date[["file_name", "doc_value"]].rename(columns={"doc_value": "invoice_date"}),
                on="file_name")
         .merge(t_name[["file_name", "doc_value"]].rename(columns={"doc_value": "seller_name"}),
                on="file_name"))
truth["total"] = pd.to_numeric(truth["total"], errors="coerce")
T = truth.set_index("file_name").to_dict("index")
print(f"[1] 정답지 {len(T):,}건")

# ---------- 2. 추출 + 채점 ----------
rows = []
for model, path in MODELS.items():
    n = 0
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        fn = r["file_name"]
        if fn not in T:
            continue
        n += 1
        got = extract(r.get("output", ""))
        ans = T[fn]

        rows.append({
            "model": model, "file_name": fn,
            # 금액
            "amt_got": got["total"], "amt_ans": ans["total"],
            "amt_extracted": got["total"] is not None,
            "amt_correct": (got["total"] is not None
                            and amount_matches(got["total"], ans["total"],
                                               eps=DEFAULT_EPS)),
            # 날짜
            "date_got": got["invoice_date"], "date_ans": ans["invoice_date"],
            "date_extracted": got["invoice_date"] is not None,
            "date_correct": got["invoice_date"] == ans["invoice_date"],
            # 상호
            "name_got": got["seller_name"], "name_ans": ans["seller_name"],
            "name_extracted": got["seller_name"] is not None,
            "name_correct": (got["seller_name"] is not None
                             and name_matches(got["seller_name"], ans["seller_name"],
                                              strict=True)),
        })
    print(f"[2] {model:<8} 채점 {n:,}건")

d = pd.DataFrame(rows)

# ---------- 3. 정확도 ----------
FIELDS = [("금액", "amt"), ("날짜", "date"), ("상호", "name")]

print(f"\n[3] 필드별 정확도  (n={len(d)//2})")
print(f"{'':<10}" + "".join(f"{k:>18}" for k, _ in FIELDS))

acc = {}
for model in MODELS:
    m = d[d.model == model]
    acc[model] = {}
    cells = []
    for label, f in FIELDS:
        ext = m[f"{f}_extracted"].mean() * 100
        cor = m[f"{f}_correct"].mean() * 100
        acc[model][label] = {"extracted_pct": round(ext, 2),
                             "correct_pct": round(cor, 2)}
        cells.append(f"{cor:>10.1f}% ({ext:>4.0f})")
    print(f"{model:<10}" + "".join(f"{c:>18}" for c in cells))

print(f"{'차이':<10}", end="")
for label, f in FIELDS:
    g = d[d.model == "granite"][f"{f}_correct"].mean() * 100
    p = d[d.model == "paddle"][f"{f}_correct"].mean() * 100
    print(f"{g-p:>+17.1f}p", end="")
print("\n            * 괄호 안은 추출 성공률(%)")

# ---------- 4. 오류 예시 ----------
print(f"\n[4] 오답 예시")
for model in MODELS:
    m = d[d.model == model]
    for label, f in FIELDS:
        bad = m[~m[f"{f}_correct"]]
        if len(bad) == 0:
            continue
        r = bad.iloc[0]
        print(f"  {model:<8} {label}  {len(bad):>3}건 오답 | "
              f"추출 {str(r[f'{f}_got'])[:32]!r} vs 정답 {str(r[f'{f}_ans'])[:32]!r}")

# ---------- 5. 저장 ----------
d.to_csv(OUT / "extracted.csv", index=False, encoding="utf-8-sig")
summary = {"n_cases": len(d) // 2, "eps": DEFAULT_EPS, "accuracy": acc,
           "diff_pp": {label: round(
               d[d.model == "granite"][f"{f}_correct"].mean() * 100
               - d[d.model == "paddle"][f"{f}_correct"].mean() * 100, 2)
               for label, f in FIELDS}}
(OUT / "accuracy.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
print(f"\n[5] 저장 -> outputs/extracted.csv, outputs/accuracy.json")