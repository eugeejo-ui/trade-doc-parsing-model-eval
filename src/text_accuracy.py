"""
text_accuracy.py — 5b단계: 문자 단위 정확도 (보조 지표)

세 필드가 아니라 출력 전체를 정답 OCR 텍스트와 대조한다.
평탄화는 extract_fields.to_rows()를 그대로 써서 두 모델에 동일 조건을 적용한다.

두 가지 기준으로 잰다.
  순서 반영 (prf)  : 읽기 순서까지 포함한 일치도
  순서 무시 (bag)  : 내용만의 일치도

두 모델의 출력 순서가 다르므로 둘을 나누어 봐야 한다.
  Granite : 헤더 → 판매자 → ITEMS → SUMMARY → 고객 주소   (고객이 맨 끝)
  Paddle  : 헤더 → 판매자·고객 → ITEMS → SUMMARY          (시각적 순서)
정답 OCR 텍스트는 시각적 읽기 순서이므로,
순서 기반 비교만 쓰면 Granite가 일방적으로 불리해진다.
"""

import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next((p for p in [here, *here.parents] if (p / "src").is_dir()), here)
DATA, OUT = ROOT / "data", ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))

from extract_fields import to_rows, flat      # 두 모델 공통 평탄화

MODELS = {"granite": OUT / "granite_raw.jsonl",
          "paddle":  OUT / "paddle_raw.jsonl"}

_NUM = re.compile(r"\d[\d\s.,]*\d|\d")


def norm(s: str) -> str:
    """대소문자·공백·통화기호를 통일한다. 두 모델에 동일 적용."""
    t = str(s).lower()
    t = re.sub(r"[$€£₩]", " ", t)
    t = re.sub(r"[\s\u00a0\u202f]+", " ", t)
    return t.strip()


def tokens(s: str) -> list[str]:
    return [w for w in re.split(r"[^\w.,/-]+", norm(s)) if w]


def numbers(s: str) -> list[str]:
    """숫자 토큰만. 구분자를 제거해 표기 차이를 흡수한다."""
    return [re.sub(r"[\s,.]", "", m) for m in _NUM.findall(norm(s))
            if re.sub(r"[\s,.]", "", m)]


def prf(got: list[str], ans: list[str]) -> tuple[float, float]:
    """순서를 고려한 재현율·정밀도 (difflib 기반).

    같은 내용이라도 배치가 다르면 점수가 깎인다.
    읽기 순서 충실도를 재는 지표다.
    """
    if not ans:
        return 0.0, 0.0
    match = sum(b.size for b in SequenceMatcher(None, ans, got,
                                                autojunk=False).get_matching_blocks())
    rec = match / len(ans) * 100
    pre = match / len(got) * 100 if got else 0.0
    return rec, pre


def bag(got: list[str], ans: list[str]) -> tuple[float, float]:
    """순서를 무시한 재현율·정밀도 (다중집합 교집합).

    내용 인식 정확도를 재는 지표다.
    출력 배치 차이에 영향받지 않는다.
    """
    if not ans:
        return 0.0, 0.0
    inter = sum((Counter(got) & Counter(ans)).values())
    return inter / len(ans) * 100, (inter / len(got) * 100 if got else 0.0)


# ---------- 1. 정답 OCR 텍스트 ----------
base = next(p for p in DATA.iterdir() if p.is_dir() and list(p.rglob("*.csv")))
frames = []
for c in sorted(base.rglob("*.csv")):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            frames.append(pd.read_csv(c, encoding=enc)); break
        except UnicodeDecodeError:
            continue
raw = pd.concat(frames, ignore_index=True)
raw["fn"] = raw["File Name"].astype(str).str.replace(".jpg", "", regex=False)
GT = dict(zip(raw["fn"], raw["OCRed Text"].astype(str)))
print(f"[1] 정답 OCR 텍스트 {len(GT):,}건")

# ---------- 2. 측정 ----------
rows = []
for model, path in MODELS.items():
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        fn = r["file_name"]
        if fn not in GT:
            continue
        got_txt = " ".join(flat(to_rows(r.get("output", ""))))
        ans_txt = GT[fn]

        gt_tok, an_tok = tokens(got_txt), tokens(ans_txt)
        gt_num, an_num = numbers(got_txt), numbers(ans_txt)

        t_rec, t_pre   = prf(gt_tok, an_tok)
        n_rec, n_pre   = prf(gt_num, an_num)
        bt_rec, _      = bag(gt_tok, an_tok)
        bn_rec, bn_pre = bag(gt_num, an_num)
        char = SequenceMatcher(None, norm(ans_txt), norm(got_txt),
                               autojunk=False).ratio() * 100

        rows.append({"model": model, "file_name": fn,
                     "char_sim": char,
                     "tok_recall": t_rec, "tok_precision": t_pre,
                     "num_recall": n_rec, "num_precision": n_pre,
                     "bag_tok_recall": bt_rec,
                     "bag_num_recall": bn_rec, "bag_num_precision": bn_pre,
                     "n_tok_ans": len(an_tok), "n_num_ans": len(an_num)})
    print(f"[2] {model:<8} {sum(1 for x in rows if x['model'] == model):,}건")

d = pd.DataFrame(rows)

# ---------- 3. 요약 ----------
COLS = [("문자유사도", "char_sim"),
        ("토큰재현(순서)", "tok_recall"),
        ("숫자재현(순서)", "num_recall"),
        ("토큰재현(무순)", "bag_tok_recall"),
        ("숫자재현(무순)", "bag_num_recall"),
        ("숫자정밀(무순)", "bag_num_precision")]

print(f"\n[3] 모델별 평균  (n={len(d)//2})")
print(f"{'':<10}" + "".join(f"{k:>16}" for k, _ in COLS))
for model in MODELS:
    m = d[d.model == model]
    print(f"{model:<10}" + "".join(f"{m[c].mean():>15.2f}%" for _, c in COLS))
print(f"{'차이':<10}", end="")
for _, c in COLS:
    g = d[d.model == "granite"][c].mean()
    p = d[d.model == "paddle"][c].mean()
    print(f"{g-p:>+15.2f}p", end="")
print()

# ---------- 4. 순서 효과 ----------
print(f"\n[4] 읽기 순서가 점수에 미치는 영향  (숫자 재현율)")
for model in MODELS:
    m = d[d.model == model]
    o, b = m["num_recall"].mean(), m["bag_num_recall"].mean()
    print(f"    {model:<10} 순서반영 {o:>6.2f}%   순서무시 {b:>6.2f}%   손실 {b-o:>6.2f}p")

go = d[d.model == "granite"]["num_recall"].mean() - d[d.model == "paddle"]["num_recall"].mean()
gb = d[d.model == "granite"]["bag_num_recall"].mean() - d[d.model == "paddle"]["bag_num_recall"].mean()
print(f"    모델 간 차이  순서반영 {go:>+6.2f}p   순서무시 {gb:>+6.2f}p"
      f"   (차이의 {abs(go)-abs(gb):.2f}p 가 순서 때문)")

# ---------- 5. 건별 승패 ----------
print(f"\n[5] 건별 승패  (순서 무시 숫자 재현율 기준)")
g = d[d.model == "granite"].set_index("file_name")["bag_num_recall"]
p = d[d.model == "paddle"].set_index("file_name")["bag_num_recall"]
i = g.index.intersection(p.index)
print(f"    granite 우세 {(g[i] > p[i]).sum():>4}건")
print(f"    paddle  우세 {(g[i] < p[i]).sum():>4}건")
print(f"    동점        {(g[i] == p[i]).sum():>4}건")

# ---------- 6. 차이가 큰 건 ----------
diff = (g[i] - p[i]).sort_values()
print(f"\n[6] 숫자 재현율 차이 (순서 무시)")
print(f"    paddle 우세 상위 3건")
for fn in diff.head(3).index:
    print(f"      {fn}  granite {g[fn]:>6.1f}%   paddle {p[fn]:>6.1f}%")
print(f"    granite 우세 상위 3건")
for fn in diff.tail(3).index[::-1]:
    print(f"      {fn}  granite {g[fn]:>6.1f}%   paddle {p[fn]:>6.1f}%")

# ---------- 7. 저장 ----------
d.round(2).to_csv(OUT / "text_accuracy.csv", index=False, encoding="utf-8-sig")
summary = {
    "n_cases": len(d) // 2,
    "models": {m: {k: round(float(d[d.model == m][c].mean()), 2)
                   for k, c in COLS} for m in MODELS},
    "diff_pp": {k: round(float(d[d.model == "granite"][c].mean()
                               - d[d.model == "paddle"][c].mean()), 2)
                for k, c in COLS},
    "note": "순서 반영 지표는 출력 배치 차이에 민감하다. "
            "Granite는 고객 주소를 맨 뒤에 출력하므로 시각적 읽기 순서인 "
            "정답 텍스트와 어긋나 불리하게 측정된다. "
            "내용 인식 정확도는 순서 무시(bag_*) 지표로 본다.",
}
(OUT / "text_accuracy.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[7] 저장 -> outputs/text_accuracy.csv, outputs/text_accuracy.json")