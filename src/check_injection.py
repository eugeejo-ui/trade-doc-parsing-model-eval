"""0.1% 주입 시 반올림으로 차이가 소멸하는 건수 실측"""
import json
from pathlib import Path
import pandas as pd

here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next((p for p in [here, *here.parents] if (p / "src").is_dir()), here)
OUT = ROOT / "outputs"

g = pd.read_csv(OUT / "kaggle_ground_truth.csv", encoding="utf-8-sig")
g["amt"] = pd.to_numeric(g["total"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                         errors="coerce")
g = g.dropna(subset=["amt"])
n = len(g)
print(f"대상 {n:,}건\n")

# --- 1. 금액 하위 분포 ---
print("[1] 금액 하위 분포")
for t in [1, 5, 10, 50, 100]:
    c = (g["amt"] < t).sum()
    print(f"    {t:>4}원 미만 : {c:>5,}건 ({c/n*100:5.2f}%)")

# --- 2. 비율별 주입 소멸 건수 (소수 둘째 자리 반올림) ---
print("\n[2] 주입 후 반올림으로 차이 소멸하는 건수")
print("    (판정 기준 eps = 0.01)")
for r in [0.001, 0.01, 0.10, 0.50]:
    dead = (abs(round(g["amt"] * (1 + r), 2) - g["amt"]) <= 0.01).sum()
    print(f"    {r:>6.1%} 주입 -> 소멸 {dead:>5,}건 ({dead/n*100:5.2f}%)")

# --- 3. 0.1%에서 소멸하는 건들의 금액 ---
dead = g[abs(round(g["amt"] * 1.001, 2) - g["amt"]) <= 0.01]
print(f"\n[3] 0.1%에서 소멸: {len(dead):,}건")
if len(dead):
    print(f"    금액 범위 {dead['amt'].min():.2f} ~ {dead['amt'].max():.2f}")
    print(f"    실제 금액: {sorted(dead['amt'].round(2).tolist())[:20]}")
