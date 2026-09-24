"""1단계 - Kaggle 송장 CSV 구조 실측"""
import sys, json, re
from pathlib import Path
from collections import Counter
import pandas as pd

here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next((p for p in [here, *here.parents] if (p / "src").is_dir()), here)
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

base = next((p for p in DATA.iterdir() if p.is_dir() and list(p.rglob("*.csv"))), None)
assert base, f"CSV를 품은 폴더가 없습니다: {DATA}"
print(f"[0] 데이터 폴더: {base.name}")

# ---------- 1. CSV 로드 ----------
csvs = sorted(base.rglob("*.csv"))
print(f"\n[1] CSV {len(csvs)}개")
frames = []
for c in csvs:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            d = pd.read_csv(c, encoding=enc); break
        except UnicodeDecodeError:
            continue
    d["__src"] = c.stem
    frames.append(d)
    print(f"    {c.name:<18} {len(d):>5,}행 x {len(d.columns)}열  enc={enc}")
df = pd.concat(frames, ignore_index=True)
print(f"    합계 {len(df):,}행   (이미지 1,489장 대비)")
print(f"    컬럼: {[c for c in df.columns if c != '__src']}")

# ---------- 2. JSON 컬럼 탐지 ----------
def is_json(v):
    if not isinstance(v, str) or "{" not in v: return False
    try: return isinstance(json.loads(v), dict)
    except Exception: return False

jcol = None
for c in df.columns:
    if c == "__src": continue
    hit = df[c].dropna().head(30).map(is_json).mean()
    if hit > 0.5: jcol = c; break
print(f"\n[2] JSON 컬럼: {jcol or '없음 <- 수동 확인 필요'}")
if not jcol:
    print("    각 컬럼 첫 값 미리보기:")
    for c in df.columns:
        if c == "__src": continue
        v = df[c].dropna()
        print(f"      {c:<20} = {str(v.iloc[0])[:100] if len(v) else '(비어있음)'}")
    sys.exit(1)

# ---------- 3. JSON 파싱 ----------
recs, bad = [], 0
for _, row in df.iterrows():
    try: d = json.loads(row[jcol])
    except Exception: bad += 1; continue
    inv = d.get("invoice", {}) or {}
    sub = d.get("subtotal", {}) or {}
    recs.append({
        "src": row["__src"],
        "invoice_number": inv.get("invoice_number"),
        "invoice_date":   inv.get("invoice_date"),
        "due_date":       inv.get("due_date") or None,
        "seller_name":    inv.get("seller_name"),
        "client_name":    inv.get("client_name"),
        "total":          sub.get("total"),
        "tax":            sub.get("tax"),
        "n_items":        len(d.get("items") or []),
    })
g = pd.DataFrame(recs)
print(f"\n[3] 파싱 {len(g):,}건  (실패 {bad})")

# ---------- 4. 규칙 3종 필드 채움 비율 ----------
def pct(s): return s.notna().mean()*100 if len(s) else 0
print(f"\n[4] 규칙 3종 대응 필드 채움 비율")
print(f"    amount_matches <- total        : {pct(g['total']):6.2f}%")
print(f"    name_matches   <- seller_name  : {pct(g['seller_name']):6.2f}%")
print(f"    date_before    <- invoice_date : {pct(g['invoice_date']):6.2f}%   <- 100%여야 함")
print(f"    (참고)            due_date     : {pct(g['due_date']):6.2f}%")

# ---------- 5. 마감 간격 실측 (UCP 600 21일 대조) ----------
print(f"\n[5] 발행일~지급기한 실측 간격  (UCP 600 제14조(c) = 21일)")
g["d_inv"] = pd.to_datetime(g["invoice_date"], format="%m/%d/%Y", errors="coerce")
g["d_due"] = pd.to_datetime(g["due_date"],     format="%m/%d/%Y", errors="coerce")
both = g.dropna(subset=["d_inv", "d_due"])
if len(both):
    gap = (both["d_due"] - both["d_inv"]).dt.days
    print(f"    대상 {len(both):,}건")
    print(gap.describe(percentiles=[.25,.5,.75]).round(1).to_string())
    print(f"    최빈값 {gap.mode().iloc[0]}일 / 중앙값 {gap.median():.0f}일")
    print(f"    -> 21일과의 차이: {abs(gap.median()-21):.0f}일")
else:
    print("    due_date 파싱 가능 건 없음 -> 교차확인 불가. 21일 그대로 사용")

print(f"\n    발행일 파싱 성공 {g['d_inv'].notna().mean()*100:.2f}%")
if g["d_inv"].notna().any():
    print(f"    기간 {g['d_inv'].min():%Y-%m-%d} ~ {g['d_inv'].max():%Y-%m-%d}")

# ---------- 6. 금액 분포 (층화용) ----------
g["amt"] = pd.to_numeric(g["total"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                         errors="coerce")
print(f"\n[6] 총액 분포  (층화 기준 후보)")
print(g["amt"].describe(percentiles=[.25,.5,.75]).round(2).to_string())
print(f"\n    품목 수: 평균 {g['n_items'].mean():.1f} / 최대 {g['n_items'].max()}")

# ---------- 7. 이미지 매칭 ----------
imgs = [p for p in base.rglob("*.jpg")]
b1 = {p.stem for p in imgs if p.stem.startswith("batch1-")}
print(f"\n[7] 이미지 매칭")
print(f"    전체 jpg {len(imgs):,}장 (중복 포함)")
print(f"    batch1 고유 {len(b1):,}장   <- 1,489 근처여야 함")
idcols = [c for c in df.columns if any(k in c.lower() for k in ["file","image","name","id","path"])]
print(f"    CSV 파일명 후보 컬럼: {idcols or '없음'}")
for c in idcols:
    v = df[c].dropna().astype(str).map(lambda s: Path(s).stem)
    m = len(set(v) & b1)
    print(f"      '{c}' 매칭 {m:,} / {v.nunique():,}")

# ---------- 8. 저장 ----------
g.drop(columns=["d_inv","d_due"]).to_csv(OUT / "kaggle_ground_truth.csv",
                                          index=False, encoding="utf-8-sig")
summary = {
    "csv_files": [c.name for c in csvs],
    "rows": int(len(df)), "parsed": int(len(g)), "parse_failed": int(bad),
    "json_column": jcol,
    "fill_pct": {k: round(pct(g[k]), 2) for k in
                 ["total","seller_name","invoice_date","due_date","client_name"]},
    "gap_days": ({"n": int(len(both)), "median": float(gap.median()),
                  "mode": int(gap.mode().iloc[0])} if len(both) else None),
    "ucp600_reference_days": 21,
    "amount": {"median": float(g["amt"].median()), "max": float(g["amt"].max())},
    "images_batch1_unique": len(b1),
}
(OUT / "kaggle_schema.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
print(f"\n[8] 저장 -> outputs/kaggle_ground_truth.csv, outputs/kaggle_schema.json")
