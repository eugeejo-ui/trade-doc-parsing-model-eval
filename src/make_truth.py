"""1.5단계 - 조건별 정답지 생성 (모델 실행 전에 확정)"""
import sys, json, random, re
from pathlib import Path
import pandas as pd

here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next((p for p in [here, *here.parents] if (p / "src").is_dir()), here)
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from rules import DEFAULT_EPS

SEED         = 42
DEFECT_RATE  = 0.70
CURRENCY_DP  = 2
DEADLINE_DAY = 21
AMOUNT_RATES = [0.001, 0.01, 0.10, 0.50]
DATE_DAYS    = [1, 3, 7, 30]

rng = random.Random(SEED)


def parse_amount(s):
    """이 데이터셋의 금액 표기를 파싱한다.
    천 단위 구분 = 공백.  소수점 = 점 또는 쉼표 (건마다 다름).
    예) '70 577,35' -> 70577.35 / '3 838.87' -> 3838.87
    """
    t = re.sub(r"[\s\u00a0\u202f]", "", str(s))          # 공백류 제거
    if "," in t and "." in t:                            # 둘 다 있으면 뒤쪽이 소수점
        t = (t.replace(".", "") if t.rfind(",") > t.rfind(".")
             else t.replace(",", "")).replace(",", ".")
    else:
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return float("nan")


# ---------- 1. 원본 CSV 로드 ----------
base = next(p for p in DATA.iterdir() if p.is_dir() and list(p.rglob("*.csv")))
rows = []
for c in sorted(base.rglob("*.csv")):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            d = pd.read_csv(c, encoding=enc); break
        except UnicodeDecodeError:
            continue
    rows.append(d)
raw = pd.concat(rows, ignore_index=True)
print(f"[1] CSV {len(raw):,}행")

recs = []
for _, r in raw.iterrows():
    try:
        j = json.loads(r["Json Data"])
    except Exception:
        continue
    inv, sub = j.get("invoice", {}) or {}, j.get("subtotal", {}) or {}
    recs.append({"file_name": Path(str(r["File Name"])).stem,
                 "invoice_date": inv.get("invoice_date"),
                 "seller_name": inv.get("seller_name"),
                 "total_raw": sub.get("total")})
g = pd.DataFrame(recs)

before = len(g)
g = g.drop_duplicates(subset="file_name", keep="first")
print(f"    파일명 중복 제거: {before - len(g)}건")

g["amount"] = g["total_raw"].map(parse_amount)
g["d_inv"] = pd.to_datetime(g["invoice_date"], format="%m/%d/%Y", errors="coerce")
g = g.dropna(subset=["amount", "d_inv", "seller_name", "file_name"]).reset_index(drop=True)
print(f"    최종 대상 {len(g):,}건   예시 {g['file_name'].iloc[0]}")

print(f"\n[1b] 금액 분포 (파싱 수정 후)")
print(g["amount"].describe(percentiles=[.25, .5, .75]).round(2).to_string())

# ---------- 2. 하자 대상 확정 ----------
n_def = int(round(len(g) * DEFECT_RATE))
idx = list(range(len(g))); rng.shuffle(idx)
target = set(idx[:n_def])
g["target"] = [i in target for i in range(len(g))]
g["sign"]  = [rng.choice([1, -1]) for _ in range(len(g))]
g["ntier"] = [rng.choice(["weak", "mid", "strong"]) for _ in range(len(g))]
print(f"\n[2] 하자 대상 {g['target'].sum():,}건 ({g['target'].mean()*100:.1f}%)  seed={SEED}")

summary = {"seed": SEED, "defect_rate": DEFECT_RATE, "n_cases": int(len(g)),
           "n_target": int(g["target"].sum()), "conditions": {}}

def save(name, df, note=""):
    df.to_csv(OUT / f"truth_{name}.csv", index=False, encoding="utf-8-sig")
    van = int((df["injected"] & ~df["is_defect"]).sum())
    summary["conditions"][name] = {"defects": int(df["is_defect"].sum()),
                                   "vanished": van, "n": int(len(df))}
    print(f"    truth_{name:<12} 하자 {df['is_defect'].sum():>4,}건  소멸 {van:>3}건  {note}")

# ---------- 3. 금액 ----------
print(f"\n[3] 금액 조건  (소수 {CURRENCY_DP}자리 반올림)")
for r in AMOUNT_RATES:
    doc  = g["amount"].round(CURRENCY_DP)
    cond = doc.where(~g["target"], (g["amount"] * (1 + g["sign"] * r)).round(CURRENCY_DP))
    save(f"amount_{r*100:g}", pd.DataFrame({
        "file_name": g["file_name"], "field": "total",
        "doc_value": doc, "cond_value": cond,
        "is_defect": g["target"] & ((cond - doc).abs() > DEFAULT_EPS),
        "injected": g["target"], "amount": g["amount"]}))

# ---------- 4. 기한 ----------
print(f"\n[4] 기한 조건  (마감 = 발행일 + {DEADLINE_DAY}일, UCP 600 14(c))")
for k in DATE_DAYS:
    dl = (g["d_inv"] + pd.Timedelta(days=DEADLINE_DAY)).where(
         ~g["target"], g["d_inv"] - pd.Timedelta(days=k))
    save(f"date_{k}", pd.DataFrame({
        "file_name": g["file_name"], "field": "invoice_date",
        "doc_value": g["d_inv"].dt.strftime("%m/%d/%Y"),
        "cond_value": dl.dt.strftime("%m/%d/%Y"),
        "is_defect": g["d_inv"] > dl, "injected": g["target"],
        "amount": g["amount"]}), f"(-{k}일)")

# ---------- 5. 명칭 ----------
print(f"\n[5] 명칭 조건  (약/중/강 3단계)")
SUF = {"ltd": "Limited", "limited": "Ltd.", "inc": "Incorporated",
       "incorporated": "Inc.", "co": "Company", "corp": "Corporation"}
ALT = ["Zenith Global Corp", "Pinnacle Trading Ltd.", "Meridian Supplies Inc."]

def perturb(name, tier):
    s = str(name)
    if tier == "weak":
        for k, v in SUF.items():
            m = re.search(rf"\b{k}\b\.?", s, re.I)
            if m:
                return s[:m.start()] + v + s[m.end():]
        return s + " Ltd."
    if tier == "mid":
        pos = [i for i, ch in enumerate(s) if ch.isalpha()]
        if not pos:
            return s + "X"
        i = pos[len(pos) // 2]
        return s[:i] + ("B" if s[i].upper() != "B" else "D") + s[i+1:]
    alt = [a for a in ALT if a != s]
    return rng.choice(alt)

doc = g["seller_name"].astype(str)
cond = pd.Series([perturb(n, t) if tg else n
                  for n, t, tg in zip(doc, g["ntier"], g["target"])], index=g.index)
save("name", pd.DataFrame({
    "file_name": g["file_name"], "field": "seller_name",
    "doc_value": doc, "cond_value": cond,
    "is_defect": doc != cond, "injected": g["target"],
    "tier": g["ntier"].where(g["target"], ""), "amount": g["amount"]}))

# ---------- 6. 실행 목록 ----------
q = pd.qcut(g["amount"], 5, labels=False, duplicates="drop")
buckets = [g.index[q == b].tolist() for b in sorted(pd.unique(q.dropna()))]
for b in buckets:
    rng.shuffle(b)
order = []
while any(buckets):
    for b in buckets:
        if b:
            order.append(b.pop())
run = g.loc[order, ["file_name", "amount"]].reset_index(drop=True)
run["seq"] = range(1, len(run) + 1)
run.to_csv(OUT / "run_list.csv", index=False, encoding="utf-8-sig")
print(f"\n[6] 실행 목록 {len(run):,}건  (금액 5분위 층화 후 셔플)")

summary["amount"] = {k: round(float(v), 2) for k, v in
                     g["amount"].describe().items()}
(OUT / "truth_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[7] 저장 완료")