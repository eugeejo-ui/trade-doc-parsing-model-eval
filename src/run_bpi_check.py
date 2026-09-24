"""0단계 - BPI Challenge 2019 금액 대조 규칙 검증"""
import sys, json
from pathlib import Path
import pandas as pd

here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next((p for p in [here, *here.parents] if (p / "src").is_dir()), here)
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from rules import amount_matches, amount_gap, DEFAULT_EPS

# --- 1. 로드 ---
cands = sorted(DATA.glob("*.csv.gz")) + sorted(DATA.glob("*.csv"))
assert cands, f"data 폴더에 CSV가 없습니다: {DATA}"
src = cands[0]
print(f"[1] 읽는 파일: {src.name}  ({src.stat().st_size/1e6:,.0f} MB)")

df = pd.read_csv(src, low_memory=False, encoding="latin-1")
df.columns = [c.strip() for c in df.columns]
V, A, C = "event Cumulative net worth (EUR)", "event concept:name", "case concept:name"
print(f"    이벤트 {len(df):>9,}  (공식 1,595,923)")
print(f"    케이스 {df[C].nunique():>9,}  (공식   251,734)")
print(f"    활동   {df[A].nunique():>9}  (공식        42)")

# --- 2. A 방법 필터 ---
ACTS = ["Create Purchase Order Item", "Record Goods Receipt", "Record Invoice Receipt"]
f1 = df[(df["case GR-Based Inv. Verif."] == True) & (df["case Goods Receipt"] == True)]
n3 = f1[C].nunique()
g = f1[f1[A].isin(ACTS)]
cnt = g.pivot_table(index=C, columns=A, values=V, aggfunc="size").fillna(0)
simple = cnt[(cnt[ACTS[0]] == 1) & (cnt[ACTS[1]] == 1) & (cnt[ACTS[2]] == 1)].index
w = g[g[C].isin(simple)].pivot_table(index=C, columns=A, values=V, aggfunc="first")
w.columns = ["PO", "GR", "INV"]
w = w[w["PO"].abs() > 1]
print(f"\n[2] 필터")
print(f"    3-way 대상    {n3:>7,}")
print(f"    1:1:1 단순건  {len(w):>7,}   (제외율 {(1-len(w)/n3)*100:.1f}%)")

# --- 3. 규칙 검증 ---
w["match"] = w.apply(lambda r: amount_matches(r["PO"], r["GR"], r["INV"]), axis=1)
w["gap"] = w.apply(lambda r: amount_gap(r["PO"], r["GR"], r["INV"]), axis=1)
truth = (w[["PO","GR","INV"]].max(axis=1) - w[["PO","GR","INV"]].min(axis=1)) <= DEFAULT_EPS
n_bad = int((w["match"] != truth).sum())
print(f"\n[3] 규칙 검증")
print("    규칙 vs 정의상 정답 : " + ("100% 일치" if n_bad == 0 else f"{n_bad}건 불일치 <- 버그"))
print(f"    불일치 판정         : {(~w['match']).sum():,}건 ({(~w['match']).mean()*100:.2f}%)")

# --- 4. 허용오차 민감도 ---
print(f"\n[4] 허용오차 민감도")
for e in [0.001, 0.01, 0.1, 1.0, 10.0]:
    n = (~w.apply(lambda r: amount_matches(r["PO"], r["GR"], r["INV"], eps=e), axis=1)).sum()
    print(f"    eps={e:>7}   -> 불일치 {n:,}건")
for rl in [0.05, 0.10]:
    n = (~w.apply(lambda r: amount_matches(r["PO"], r["GR"], r["INV"], rel=rl), axis=1)).sum()
    print(f"    rel={rl:>6.0%}   -> 불일치 {n:,}건")

# --- 5. 재작업 교차확인 ---
REWORK = ["Remove Payment Block", "Change Quantity", "Change Price",
          "Cancel Invoice Receipt", "Vendor creates debit memo", "Cancel Goods Receipt"]
rw = f1[f1[A].isin(REWORK)].groupby(C).size()
w["rework"] = w.index.map(rw).fillna(0) > 0
print(f"\n[5] 재작업 교차확인")
print(pd.crosstab(w["match"].map({False:"규칙=불일치", True:"규칙=일치"}),
                  w["rework"].map({True:"재작업 있음", False:"재작업 없음"})).to_string())
print(f"    불일치 중 재작업 {w[~w['match']]['rework'].mean()*100:5.1f}%   /   일치 중 재작업 {w[w['match']]['rework'].mean()*100:5.1f}%")
print("    * 불일치 쪽이 낮은 것이 정상 - 재작업은 하자의 증상이 아니라 치료")

# --- 6. 미정산 제외 ---
cleared = set(f1[f1[A] == "Clear Invoice"][C])
w["cleared"] = w.index.isin(cleared)
final = w[(~w["match"]) & w["cleared"]]
print(f"\n[6] 미정산 제외")
print(pd.crosstab(w["match"].map({False:"규칙=불일치", True:"규칙=일치"}),
                  w["cleared"].map({True:"정산완료", False:"미정산"})).to_string())
print(f"\n    최종 하자 후보 : {len(final):,}건")
print(final[["PO","GR","INV","gap"]].head(10).round(2).to_string())

# --- 7. 저장 ---
w.round(2).to_csv(OUT / "bpi_case_a.csv", encoding="utf-8-sig")
summary = {
    "method": "A (1:1:1 simple cases)",
    "source": "BPI Challenge 2019, 4TU.ResearchData, doi:10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1",
    "cases_3way": int(n3),
    "cases_analyzed": int(len(w)),
    "exclusion_rate_pct": round((1 - len(w)/n3)*100, 1),
    "rule_vs_definition": "100% match" if n_bad == 0 else f"{n_bad} disagreements",
    "mismatched": int((~w["match"]).sum()),
    "mismatched_cleared": int(len(final)),
    "eps": DEFAULT_EPS,
    "eps_invariant_range": "0.001 ~ 10.0 EUR",
}
(OUT / "bpi_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n[7] 저장 완료 -> outputs/bpi_case_a.csv, outputs/bpi_summary.json")
