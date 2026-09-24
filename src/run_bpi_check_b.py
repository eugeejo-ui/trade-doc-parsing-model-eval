"""0단계 B - 분할 납품 건 확장 (누적값 최종값 사용)"""
import sys, json
from pathlib import Path
import pandas as pd

here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next((p for p in [here, *here.parents] if (p / "src").is_dir()), here)
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from rules import amount_matches, amount_gap, DEFAULT_EPS

MAX_GR = 21   # 입고 횟수 상한 (95퍼센타일). 넘으면 짝짓기 필요 -> 제외

# --- 1. 로드 ---
cands = sorted(DATA.glob("*.csv.gz")) + sorted(DATA.glob("*.csv"))
assert cands, f"data 폴더에 CSV가 없습니다: {DATA}"
df = pd.read_csv(cands[0], low_memory=False, encoding="latin-1")
df.columns = [c.strip() for c in df.columns]
V, A, C, T = ("event Cumulative net worth (EUR)", "event concept:name",
              "case concept:name", "event time:timestamp")
df[T] = pd.to_datetime(df[T], dayfirst=True, errors="coerce")
print(f"[1] 로드 완료  이벤트 {len(df):,}")

ACTS = ["Create Purchase Order Item", "Record Goods Receipt", "Record Invoice Receipt"]
f1 = df[(df["case GR-Based Inv. Verif."] == True) & (df["case Goods Receipt"] == True)]
n3 = f1[C].nunique()

# --- 2. 분할 건 선별 ---
g = f1[f1[A].isin(ACTS)]
cnt = g.pivot_table(index=C, columns=A, values=V, aggfunc="size").fillna(0)
is_simple = (cnt[ACTS[0]] == 1) & (cnt[ACTS[1]] == 1) & (cnt[ACTS[2]] == 1)
has_all   = (cnt[ACTS[0]] >= 1) & (cnt[ACTS[1]] >= 1) & (cnt[ACTS[2]] >= 1)
split = cnt[(~is_simple) & has_all & (cnt[ACTS[1]] <= MAX_GR)].index

print(f"\n[2] 분할 건 선별")
print(f"    3-way 대상        {n3:>7,}")
print(f"    단순건(A 처리분)  {int(is_simple.sum()):>7,}")
print(f"    분할건 전체       {int(((~is_simple) & has_all).sum()):>7,}")
print(f"    입고<={MAX_GR}회 만       {len(split):>7,}   <- B 대상")

# --- 3. 누적값 최종값 추출 ---
sub = g[g[C].isin(split)].sort_values(T)
w = sub.groupby([C, A])[V].last().unstack()
w = w.reindex(columns=ACTS)
w.columns = ["PO", "GR", "INV"]
w = w.dropna()
w = w[w["PO"].abs() > 1]
print(f"\n[3] 최종값 추출    {len(w):,}건")

# --- 4. Clear Invoice 필수 조건 ---
cleared = set(f1[f1[A] == "Clear Invoice"][C])
before = len(w)
w = w[w.index.isin(cleared)]
print(f"[4] 정산완료만     {len(w):,}건  (미정산 {before-len(w):,}건 제외)")
assert len(w) > 0, "정산완료 건이 없습니다 - B 중단"

# --- 5. 규칙 검증 ---
w["match"] = w.apply(lambda r: amount_matches(r["PO"], r["GR"], r["INV"]), axis=1)
w["gap"]   = w.apply(lambda r: amount_gap(r["PO"], r["GR"], r["INV"]), axis=1)
truth = (w[["PO","GR","INV"]].max(axis=1) - w[["PO","GR","INV"]].min(axis=1)) <= DEFAULT_EPS
n_bad = int((w["match"] != truth).sum())
print(f"\n[5] 규칙 검증")
print("    규칙 vs 정의상 정답 : " + ("100% 일치" if n_bad == 0 else f"{n_bad}건 불일치 <- 전처리 오류"))
print(f"    불일치 판정         : {(~w['match']).sum():,}건 ({(~w['match']).mean()*100:.2f}%)")
print(f"    (A 방법 비교        : 128건 / 1.50%)")

# --- 6. eps 민감도 ---
print(f"\n[6] 허용오차 민감도")
for e in [0.001, 0.01, 1.0, 10.0]:
    n = (~w.apply(lambda r: amount_matches(r["PO"], r["GR"], r["INV"], eps=e), axis=1)).sum()
    print(f"    eps={e:>7}   -> 불일치 {n:,}건")

# --- 7. 하자 유형 ---
fin = w[~w["match"]].copy()
if len(fin):
    fin["r_gr"] = (fin["GR"] / fin["PO"]).round(3)
    t1 = ((fin["PO"] - fin["INV"]).abs() < 0.01).sum()
    t2 = (((fin["GR"] - fin["INV"]).abs() < 0.01) & ((fin["PO"] - fin["INV"]).abs() >= 0.01)).sum()
    print(f"\n[7] 하자 유형  (총 {len(fin)}건)")
    print(f"    유형1 PO=INV, GR만 어긋남       : {t1:>4}건")
    print(f"    유형2 GR=INV, 둘 다 PO와 어긋남 : {t2:>4}건")
    print(f"    GR/PO 정수배                    : {int((fin['r_gr'] % 1 == 0).sum()):>4}건")
    print(fin[["PO","GR","INV","gap"]].head(10).round(2).to_string())

# --- 8. 저장 ---
w.round(2).to_csv(OUT / "bpi_case_b.csv", encoding="utf-8-sig")
summary = {
    "method": "B (split deliveries, cumulative last value)",
    "max_gr_events": MAX_GR,
    "cases_analyzed": int(len(w)),
    "rule_vs_definition": "100% match" if n_bad == 0 else f"{n_bad} disagreements",
    "mismatched": int((~w["match"]).sum()),
    "mismatch_rate_pct": round((~w["match"]).mean()*100, 2),
    "compare_method_a": {"cases": 8564, "mismatched_cleared": 128, "rate_pct": 1.50},
}
(OUT / "bpi_summary_b.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n[8] 저장 완료 -> outputs/bpi_case_b.csv, outputs/bpi_summary_b.json")
